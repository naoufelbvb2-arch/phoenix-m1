"""
M1-FIXEDWRITE -- frozen addresses + FIXED write-assignment (concentration fix).

Tests whether pinning each value to ONE fixed input-independent slot makes the
board blind-readable, fixing the combo's failure (input-dependent placement ->
NOT_BUILDABLE).  Keeps frozen keys + the annealed per-value scaffold.

Zone C is a MEASUREMENT PROBE ONLY: never trained jointly, never gets gradient
into the model, attached fresh AFTER the model is frozen.  Training the evaluator
on the metric it measures would invalidate the metric.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from task_m1 import generate_batch, key_marker, value_marker, VOCAB_SIZE
from model_m1_fixedwrite import PhoenixM1FixedWrite, fixed_slot_assignment
from model_m1_frozen import _extract, D_LOCAL
from board import make_board, K
from run_m1_part3 import _tok_after, anneal_factor
from zone_c import ZoneC


def _value_sim(vals):
    """Mean off-diagonal cosine among the given (B, n, D) payload slots -- works
    for any n (slot_analysis._pairwise_cosine hardcodes K=8 and is frozen)."""
    normed = F.normalize(vals, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2)).mean(0)     # (n, n)
    n = sim.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=vals.device)
    return sim[off].mean().item()

warnings.filterwarnings("ignore", category=UserWarning)

N_TRAIN    = 20_000
N_EPOCHS   = 30
BATCH_SIZE = 128
LR         = 1e-3
AUX_WEIGHT = 2.0
ALPHA_FLOOR = 0.10
CKPT_PATH  = "m1_fixedwrite_ckpt.pt"

N_TRAIN_C  = 20_000
N_TEST_C   = 4_096
N_EPOCHS_C = 40
LR_C       = 1e-3
N_QUERY_C  = 4
ZONEC_TH   = 0.90


def board_of(model, x_a, x_b):
    with torch.no_grad():
        h_a = model.zone_a.encode(x_a)
        h_b = model.zone_b.encode(x_b)
        keys_a, vals_a = model.zone_a.write(h_a, x_a)
        keys_b, vals_b = model.zone_b.write(h_b, x_b)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])
    return bk, bv


def _permute_slots(bk, bv, gen):
    perm = torch.randperm(bk.size(1), generator=gen)
    return bk[:, perm, :], bv[:, perm, :]


def assess_write_scaling():
    ok = True
    for n in range(1, K + 1):
        a = fixed_slot_assignment(n, K)
        ok = ok and (a == list(range(n))) and (len(set(a)) == n)
    tag = "CLEAN_GENERAL" if ok else "HAND_WIRED"
    just = ("value i -> zone slot i (disjoint, input-independent) via "
            "fixed_slot_assignment(n,K); computed for any n<=K, no per-value "
            "constants; 3-value task is a special case")
    return tag, just


def train_main(seed=42):
    torch.manual_seed(seed)
    xa_tr, xb_tr, ya_tr, yb_tr = generate_batch(N_TRAIN, seed=0)
    loader = DataLoader(
        TensorDataset(xa_tr, xb_tr, ya_tr, yb_tr),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xa_te, xb_te, ya_te, yb_te = generate_batch(4096, seed=9999)

    model = PhoenixM1FixedWrite()
    aux_v0 = nn.Linear(D_LOCAL, VOCAB_SIZE)
    aux_v1 = nn.Linear(D_LOCAL, VOCAB_SIZE)
    aux_v2 = nn.Linear(D_LOCAL, VOCAB_SIZE)

    params = (list(model.parameters())
              + list(aux_v0.parameters()) + list(aux_v1.parameters()) + list(aux_v2.parameters()))
    opt = torch.optim.Adam(params, lr=LR)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"Zone B writes v0->slot{model.zone_b.slots[0]}, v1->slot{model.zone_b.slots[1]} (fixed); "
          f"Zone A writes v2->slot{model.zone_a.slots[0]} (fixed). "
          f"slot_keys.requires_grad={model.zone_b.slot_keys.requires_grad}")

    def eval_all():
        model.eval()
        with torch.no_grad():
            _, _, _, vals_b = model(xa_te, xb_te)
            r0 = _extract(model.h_a2, xa_te, key_marker(0))
            r1 = _extract(model.h_a2, xa_te, key_marker(1))
            r2 = _extract(model.h_b2, xb_te, key_marker(2))
            v0 = _tok_after(xb_te, value_marker(0))
            v1 = _tok_after(xb_te, value_marker(1))
            v2 = _tok_after(xa_te, value_marker(2))
            av0 = (aux_v0(r0).argmax(1) == v0).float().mean().item()
            av1 = (aux_v1(r1).argmax(1) == v1).float().mean().item()
            av2 = (aux_v2(r2).argmax(1) == v2).float().mean().item()
            # payload differentiation over Zone B's ASSIGNED (non-zero) slots
            value_sim = _value_sim(vals_b[:, :model.zone_b.n_w, :])
        model.train()
        return av0, av1, av2, value_sim

    print(f"  {'ep':>3}  {'av0':>7}  {'av1':>7}  {'av2':>7}  {'VALUE_SIM':>9}  "
          f"{'gate_A':>7}  {'gate_B':>7}  {'w_aux':>6}")

    checked = False
    for epoch in range(1, N_EPOCHS + 1):
        af = anneal_factor(epoch, N_EPOCHS)
        wa = AUX_WEIGHT * af
        model.train()
        for xa, xb, ya, yb in loader:
            opt.zero_grad()
            logits_a, logits_b, _, _ = model(xa, xb)
            loss = ce(logits_a, ya) + ce(logits_b, yb)
            if wa > 0.0:
                r0 = _extract(model.h_a2, xa, key_marker(0))
                r1 = _extract(model.h_a2, xa, key_marker(1))
                r2 = _extract(model.h_b2, xb, key_marker(2))
                v0 = _tok_after(xb, value_marker(0))
                v1 = _tok_after(xb, value_marker(1))
                v2 = _tok_after(xa, value_marker(2))
                loss = (loss
                        + wa * ce(aux_v0(r0), v0)
                        + wa * ce(aux_v1(r1), v1)
                        + wa * ce(aux_v2(r2), v2))
            loss.backward()
            if not checked:
                print(f"FROZEN_KEYS_GRAD_CHECK: zone_a.slot_keys.grad={model.zone_a.slot_keys.grad} "
                      f"zone_b.slot_keys.grad={model.zone_b.slot_keys.grad} (None => addresses untrained)")
                checked = True
            if wa == 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            with torch.no_grad():
                model.zone_a.alpha.clamp_(min=ALPHA_FLOOR)
                model.zone_b.alpha.clamp_(min=ALPHA_FLOOR)

        if epoch % 3 == 0 or epoch == N_EPOCHS:
            av0, av1, av2, vsim = eval_all()
            print(f"  {epoch:>3}  {av0:>7.4f}  {av1:>7.4f}  {av2:>7.4f}  {vsim:>9.4f}  "
                  f"{model.zone_a.alpha.item():>7.4f}  {model.zone_b.alpha.item():>7.4f}  {wa:>6.2f}")

    av0, av1, av2, vsim = eval_all()
    return model, av0, av1, av2, vsim


def zone_c_probe(model, seed=1234):
    """Zone C is attached AFTER freezing; it never influences the model."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    xa_c, xb_c, _, _ = generate_batch(N_TRAIN_C, seed=111)
    xa_t, xb_t, _, _ = generate_batch(N_TEST_C, seed=222)

    bk_tr, bv_tr = board_of(model, xa_c, xb_c)      # no_grad: model is frozen
    bk_te, bv_te = board_of(model, xa_t, xb_t)
    v0_tr = _tok_after(xb_c, value_marker(0)); v1_tr = _tok_after(xb_c, value_marker(1))
    v0_te = _tok_after(xb_t, value_marker(0)); v1_te = _tok_after(xb_t, value_marker(1))

    zc = ZoneC(n_query=N_QUERY_C)
    opt_c = torch.optim.Adam(zc.parameters(), lr=LR_C)   # ONLY Zone C params
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(bk_tr, bv_tr, v0_tr, v1_tr),
        batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    def eval_zc(permute):
        zc.eval()
        with torch.no_grad():
            bk, bv = bk_te, bv_te
            if permute:
                bk, bv = _permute_slots(bk, bv, gen)
            l0, l1 = zc(bk, bv)
            a0 = (l0.argmax(1) == v0_te).float().mean().item()
            a1 = (l1.argmax(1) == v1_te).float().mean().item()
        zc.train()
        return a0, a1

    print("\nZone C (independent blind probe -- NOT trained into the model) under PERMUTED slots")
    print(f"  {'ep':>3}  {'v0(perm)':>9}  {'v1(perm)':>9}")
    for ep in range(1, N_EPOCHS_C + 1):
        zc.train()
        for bk, bv, v0, v1 in loader:
            bkp, bvp = _permute_slots(bk, bv, gen)
            l0, l1 = zc(bkp, bvp)
            loss = ce(l0, v0) + ce(l1, v1)
            opt_c.zero_grad(); loss.backward(); opt_c.step()
        if ep % 4 == 0 or ep == N_EPOCHS_C:
            a0p, a1p = eval_zc(permute=True)
            print(f"  {ep:>3}  {a0p:>9.4f}  {a1p:>9.4f}")

    return eval_zc(permute=True), eval_zc(permute=False)


def main():
    wtag, wjust = assess_write_scaling()
    print(f"WRITE_ASSIGN_SCALING={wtag}: {wjust}\n")

    model, av0, av1, av2, vsim = train_main()

    # freeze everything, then checkpoint
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\ncheckpoint saved -> {CKPT_PATH}")

    (a0p, a1p), (a0u, a1u) = zone_c_probe(model)

    if a0p >= ZONEC_TH and a1p >= ZONEC_TH:
        verdict = "BUILDABLE"
    elif (a0p >= ZONEC_TH) != (a1p >= ZONEC_TH):
        which = "v0" if a0p >= ZONEC_TH else "v1"
        verdict = f"PARTIAL(only_{which})"
    else:
        verdict = "NOT_BUILDABLE"

    print("\n" + "=" * 72)
    print(f"  routing: av0={av0:.4f} av1={av1:.4f} av2={av2:.4f}  VALUE_SIM(ZoneB)={vsim:.4f}")
    print(f"  Zone C permuted   : v0={a0p:.4f}  v1={a1p:.4f}")
    print(f"  Zone C unpermuted : v0={a0u:.4f}  v1={a1u:.4f}   (index-reliance contrast)")
    print(f"  WRITE_ASSIGN_SCALING={wtag}")
    print("=" * 72)
    print(f"\nM1-FixedWrite: AV0={av0:.4f} AV1={av1:.4f} AV2={av2:.4f} VALUE_SIM={vsim:.4f} "
          f"ZONEC_V0={a0p:.4f} ZONEC_V1={a1p:.4f} WRITE_ASSIGN_SCALING={wtag} VERDICT={verdict}")


if __name__ == "__main__":
    main()
