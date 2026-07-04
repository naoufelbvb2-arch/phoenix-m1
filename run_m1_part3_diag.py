"""
M1-Part3 DIAGNOSTIC -- observation only.  NO fixes, NO redesign, NO aux floor.

Purpose: Attempt-20 PASSED at ep40 but acc_A is a slow-decay pass (held ~1.0
through ep38, drifted to 0.8635 by ep40).  Before freezing the addressing space
in Part 4 we must know WHAT is decaying:

  KEY_SIM flat (~0.29) while av0/acc_A drift  -> foundation sound, downstream bleed
                                                 -> DIAGNOSIS=FOUNDATION_STABLE
  KEY_SIM rises as acc_A falls                -> the addressing space itself drifts
                                                 -> DIAGNOSIS=ADDRESSING_DRIFT

Method: run_m1_part3.py never saved a checkpoint, so we REPRODUCE Attempt-20
exactly (identical seed, module-creation order, data, loss, clamp, label
smoothing, post-anneal clip) and continue to ep55.  The anneal is PINNED to the
40-epoch horizon via anneal_factor(epoch, 40): AUX is already 0 from ep20, so
epochs 1-40 are bit-identical to Attempt-20 (verified against its ep40 row) and
41-55 is the genuine continuation under the same AUX=0 loss.  Every epoch we log
KEY_SIM (Zone B write-keys) + av0/av1/av2 + acc/gates.  Reuses everything.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from task_m1 import generate_batch, key_marker, value_marker, VOCAB_SIZE
from model_m1 import PhoenixM1, _extract, D_LOCAL
from slot_analysis import analyze_slots
from run_m1_part3 import (
    N_TRAIN, BATCH_SIZE, LR, AUX_ROUTE, AUX_CASC, _tok_after, anneal_factor,
)

warnings.filterwarnings("ignore", category=UserWarning)

ANNEAL_HORIZON = 40   # pin anneal to Attempt-20's schedule (half = 20)
BASE_EPOCHS    = 40   # Attempt-20 endpoint -> diagnosis baseline
DIAG_EPOCHS    = 55   # continue to here
KEY_SIM_RISE_TH = 0.05   # a >5-point cosine rise counts as the space drifting


def run(seed: int = 42):
    torch.manual_seed(seed)

    # --- identical data + seeding order as run_m1_part3.run() -----------------
    xa_tr, xb_tr, ya_tr, yb_tr = generate_batch(N_TRAIN, seed=0)
    loader = DataLoader(
        TensorDataset(xa_tr, xb_tr, ya_tr, yb_tr),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xa_te, xb_te, ya_te, yb_te = generate_batch(4096, seed=9999)

    # --- identical module-creation order (so weight-init RNG matches) ---------
    model = PhoenixM1()
    aux_v0 = nn.Linear(D_LOCAL, VOCAB_SIZE)
    aux_v1 = nn.Linear(D_LOCAL, VOCAB_SIZE)
    aux_v2 = nn.Linear(D_LOCAL, VOCAB_SIZE)
    probe_s0 = nn.Sequential(nn.Linear(D_LOCAL, 128), nn.GELU(), nn.Linear(128, VOCAB_SIZE))
    probe_s1 = nn.Sequential(nn.Linear(D_LOCAL, 128), nn.GELU(), nn.Linear(128, VOCAB_SIZE))
    head_a_p3 = nn.Sequential(
        nn.Linear(2 * VOCAB_SIZE, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
    )

    all_params = (
        list(model.parameters())
        + list(aux_v0.parameters())
        + list(aux_v1.parameters())
        + list(aux_v2.parameters())
        + list(probe_s0.parameters())
        + list(probe_s1.parameters())
        + list(head_a_p3.parameters())
    )
    opt = torch.optim.Adam(all_params, lr=LR)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def get_logits_a(h_a2_0, h_a2_1, xa):
        r0 = _extract(h_a2_0, xa, key_marker(0))
        r1 = _extract(h_a2_1, xa, key_marker(1))
        s0_l = probe_s0(r0)
        s1_l = probe_s1(r1)
        rep = torch.cat([F.softmax(s0_l, -1), F.softmax(s1_l, -1)], -1)
        return head_a_p3(rep), s0_l, s1_l, r0, r1

    def eval_all():
        model.eval()
        with torch.no_grad():
            _, logits_b, _, _ = model(xa_te, xb_te)
            logits_a, s0_l, s1_l, r0, r1 = get_logits_a(model.h_a2_0, model.h_a2_1, xa_te)
            cascade_a = (logits_a.argmax(1) == ya_te).float().mean().item()
            direct_a  = (((s0_l.argmax(1) + s1_l.argmax(1)) % VOCAB_SIZE) == ya_te).float().mean().item()
            acc_a = max(cascade_a, direct_a)
            acc_b = (logits_b.argmax(1) == yb_te).float().mean().item()
            v0_te = _tok_after(xb_te, value_marker(0))
            v1_te = _tok_after(xb_te, value_marker(1))
            v2_te = _tok_after(xa_te, value_marker(2))
            r2 = _extract(model.h_b2, xb_te, key_marker(2))
            av0 = (aux_v0(r0).argmax(1) == v0_te).float().mean().item()
            av1 = (aux_v1(r1).argmax(1) == v1_te).float().mean().item()
            av2 = (aux_v2(r2).argmax(1) == v2_te).float().mean().item()
            # KEY_SIM (Zone B write-keys) -- the addressing-space stability signal.
            # no_grad forward through encode+write; dropout=0 so it is RNG-neutral
            # and does not perturb the training trajectory.
            key_sim = analyze_slots(model, xa_te, xb_te)["sim_B"]
        model.train()
        return key_sim, av0, av1, av2, acc_a, acc_b

    rows = {}
    print(f"  {'ep':>3}  {'KEY_SIM':>7}  {'av0':>7}  {'av1':>7}  {'av2':>7}  "
          f"{'acc_A':>7}  {'acc_B':>7}  {'gate_A':>7}  {'gate_B':>7}")

    for epoch in range(1, DIAG_EPOCHS + 1):
        af = anneal_factor(epoch, ANNEAL_HORIZON)   # pinned to Attempt-20 schedule
        wr = AUX_ROUTE * af
        wc = AUX_CASC * af
        model.train()

        for xa, xb, ya, yb in loader:
            opt.zero_grad()

            logits_a_direct, logits_b, _, _ = model(xa, xb)
            logits_a, s0_logits, s1_logits, r_a0, r_a1 = get_logits_a(model.h_a2_0, model.h_a2_1, xa)
            r_b2 = _extract(model.h_b2, xb, key_marker(2))

            loss = ce(logits_a, ya) + ce(logits_b, yb)

            if wr > 0.0:
                v0 = _tok_after(xb, value_marker(0))
                v1 = _tok_after(xb, value_marker(1))
                v2 = _tok_after(xa, value_marker(2))
                loss = (loss
                        + wr * ce(aux_v0(r_a0), v0)
                        + wr * ce(aux_v1(r_a1), v1)
                        + wr * ce(aux_v2(r_b2), v2))

            if wc > 0.0:
                k0 = _tok_after(xa, key_marker(0))
                v0 = _tok_after(xb, value_marker(0))
                k1 = _tok_after(xa, key_marker(1))
                v1 = _tok_after(xb, value_marker(1))
                s0 = (k0 + v0) % VOCAB_SIZE
                s1 = (k1 + v1) % VOCAB_SIZE
                loss = (loss
                        + wc * ce(s0_logits, s0)
                        + wc * ce(s1_logits, s1))

            loss.backward()
            if wr == 0.0 and wc == 0.0:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            opt.step()

            with torch.no_grad():
                model.zone_a.alpha.clamp_(min=0.35)
                model.zone_b.alpha.clamp_(min=0.35)

        key_sim, av0, av1, av2, acc_a, acc_b = eval_all()
        ga = model.zone_a.alpha.item()
        gb = model.zone_b.alpha.item()
        rows[epoch] = dict(key_sim=key_sim, av0=av0, av1=av1, av2=av2,
                           acc_a=acc_a, acc_b=acc_b, ga=ga, gb=gb)

        if epoch >= BASE_EPOCHS:
            print(f"  {epoch:>3}  {key_sim:>7.4f}  {av0:>7.4f}  {av1:>7.4f}  {av2:>7.4f}  "
                  f"{acc_a:>7.4f}  {acc_b:>7.4f}  {ga:>7.4f}  {gb:>7.4f}")

    b = rows[BASE_EPOCHS]
    e = rows[DIAG_EPOCHS]

    # KEY_SIM flat while acc_A/av0 bleed  -> foundation stable (downstream decay).
    # KEY_SIM rises materially              -> the addressing space itself drifting.
    if (e["key_sim"] - b["key_sim"]) > KEY_SIM_RISE_TH:
        diagnosis = "ADDRESSING_DRIFT"
    else:
        diagnosis = "FOUNDATION_STABLE"

    print()
    print(f"M1-Part3-Diag: KEY_SIM_40={b['key_sim']:.4f} KEY_SIM_55={e['key_sim']:.4f} "
          f"ACC_A_40={b['acc_a']:.4f} ACC_A_55={e['acc_a']:.4f} "
          f"AV0_40={b['av0']:.4f} AV0_55={e['av0']:.4f} DIAGNOSIS={diagnosis}")

    return rows, diagnosis


if __name__ == "__main__":
    run()
