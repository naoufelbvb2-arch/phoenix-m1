"""
M1-Part4 -- freeze the M1 addressing space + forward-compatibility go/no-go.

STEP 1  Deliberate snapshot selection.  run_m1_part3.py saved no checkpoint, so
        we reproduce the Attempt-20 training BIT-EXACTLY (verified by the ep40
        row) and sweep ep30..50, picking the LOWEST-KEY_SIM epoch that still
        solves the task (acc_A>=0.80 AND acc_B>=0.80).  We do NOT modify the
        training (no crystallization loss); reproduction only regenerates the
        frozen artifact that was never saved.
STEP 2  Bit-exact freeze of BOTH zones + board-writing machinery; verify the
        written (key,value) slots are a deterministic function of input.
STEP 3  Attach Zone C -- a brand-new BLIND consumer (frozen board is its ONLY
        input) -- and train ONLY Zone C to recover v0 AND v1 under PERMUTED slot
        order (forces key-matching, not slot-index).  Verdict:
        BUILDABLE / PARTIAL / NOT_BUILDABLE.

FROZEN & untouched: task_m1.py, model_m1.py, slot_analysis.py, run_m1_part3.py.
"""

import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from task_m1 import generate_batch, key_marker, value_marker, VOCAB_SIZE
from model_m1 import PhoenixM1, _extract, D_LOCAL
from board import make_board, K
from slot_analysis import analyze_slots
from run_m1_part3 import (
    N_TRAIN, BATCH_SIZE, LR, AUX_ROUTE, AUX_CASC, _tok_after, anneal_factor,
)
from zone_c import ZoneC

warnings.filterwarnings("ignore", category=UserWarning)

ANNEAL_HORIZON = 40        # pin anneal to Attempt-20 schedule (half = 20)
SWEEP_LO, SWEEP_HI = 30, 50
TASK_TH  = 0.80           # task-solved gate for snapshot eligibility
ZONEC_TH = 0.90           # per-value recovery bar for BUILDABLE

N_TRAIN_C  = 20_000
N_TEST_C   = 4_096
N_EPOCHS_C = 15
LR_C       = 1e-3
N_QUERY_C  = 4


# ─────────────────────────────────────────────────────────────────────────────
# Frozen-board extraction: the deterministic (key,value) slots for an input pair.
# ─────────────────────────────────────────────────────────────────────────────
def board_of(model, x_a, x_b):
    with torch.no_grad():
        h_a = model.zone_a.encode(x_a)
        h_b = model.zone_b.encode(x_b)
        keys_a, vals_a = model.zone_a.write(h_a)
        keys_b, vals_b = model.zone_b.write(h_b)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])
    return bk, bv


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- reproduce Attempt-20 exactly, sweep ep30..50, pick lowest KEY_SIM
#           among task-solved snapshots.
# ─────────────────────────────────────────────────────────────────────────────
def reproduce_and_select():
    torch.manual_seed(42)

    xa_tr, xb_tr, ya_tr, yb_tr = generate_batch(N_TRAIN, seed=0)
    loader = DataLoader(
        TensorDataset(xa_tr, xb_tr, ya_tr, yb_tr),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    xa_te, xb_te, ya_te, yb_te = generate_batch(4096, seed=9999)

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
        + list(aux_v0.parameters()) + list(aux_v1.parameters()) + list(aux_v2.parameters())
        + list(probe_s0.parameters()) + list(probe_s1.parameters())
        + list(head_a_p3.parameters())
    )
    opt = torch.optim.Adam(all_params, lr=LR)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Verbatim Attempt-20 cascade (returns r0/r1 too) so the graph -- and thus
    # the reproduced trajectory -- is bit-identical.
    def get_logits_a(h_a2_0, h_a2_1, xa):
        r0 = _extract(h_a2_0, xa, key_marker(0))
        r1 = _extract(h_a2_1, xa, key_marker(1))
        s0_l = probe_s0(r0); s1_l = probe_s1(r1)
        rep = torch.cat([F.softmax(s0_l, -1), F.softmax(s1_l, -1)], -1)
        return head_a_p3(rep), s0_l, s1_l, r0, r1

    def eval_task():
        model.eval()
        with torch.no_grad():
            _, logits_b, _, _ = model(xa_te, xb_te)
            logits_a, s0_l, s1_l, _, _ = get_logits_a(model.h_a2_0, model.h_a2_1, xa_te)
            cascade_a = (logits_a.argmax(1) == ya_te).float().mean().item()
            direct_a  = (((s0_l.argmax(1) + s1_l.argmax(1)) % VOCAB_SIZE) == ya_te).float().mean().item()
            acc_a = max(cascade_a, direct_a)
            acc_b = (logits_b.argmax(1) == yb_te).float().mean().item()
            key_sim = analyze_slots(model, xa_te, xb_te)["sim_B"]
        model.train()
        return acc_a, acc_b, key_sim

    print("STEP 1 -- reproduce Attempt-20 and sweep ep30..50")
    print(f"  {'ep':>3}  {'KEY_SIM':>7}  {'acc_A':>7}  {'acc_B':>7}  {'eligible':>8}")

    ckpts = {}          # ep -> deepcopy(model.state_dict())
    metrics = {}        # ep -> (key_sim, acc_a, acc_b)

    for epoch in range(1, SWEEP_HI + 1):
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

        if epoch >= SWEEP_LO:
            acc_a, acc_b, key_sim = eval_task()
            eligible = (acc_a >= TASK_TH and acc_b >= TASK_TH)
            ckpts[epoch] = copy.deepcopy(model.state_dict())
            metrics[epoch] = (key_sim, acc_a, acc_b)
            print(f"  {epoch:>3}  {key_sim:>7.4f}  {acc_a:>7.4f}  {acc_b:>7.4f}  "
                  f"{'yes' if eligible else 'no':>8}")

    eligible_eps = [ep for ep, (ks, a, b) in metrics.items()
                    if a >= TASK_TH and b >= TASK_TH]
    if not eligible_eps:
        raise RuntimeError("no task-solved snapshot in ep30..50 -- cannot freeze")
    freeze_ep = min(eligible_eps, key=lambda ep: metrics[ep][0])
    fk = metrics[freeze_ep][0]
    print(f"\n  selected FREEZE_EPOCH={freeze_ep} (lowest KEY_SIM among solved) "
          f"KEY_SIM={fk:.4f} acc_A={metrics[freeze_ep][1]:.4f} acc_B={metrics[freeze_ep][2]:.4f}")

    # load the chosen snapshot back into `model`
    model.load_state_dict(ckpts[freeze_ep])
    return model, freeze_ep, fk, (xa_te, xb_te)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- bit-exact freeze + determinism verification.
# ─────────────────────────────────────────────────────────────────────────────
def freeze_and_verify(model, xa_te, xb_te):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    # slots must be a deterministic function of input: same batch twice -> identical
    bk1, bv1 = board_of(model, xa_te[:512], xb_te[:512])
    bk2, bv2 = board_of(model, xa_te[:512], xb_te[:512])
    verified = bool(torch.equal(bk1, bk2) and torch.equal(bv1, bv2))
    print(f"\nSTEP 2 -- freeze verified (same input -> bit-identical slots): "
          f"{'yes' if verified else 'no'}")
    return verified


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Zone C forward-compatibility test.
# ─────────────────────────────────────────────────────────────────────────────
def _permute_slots(bk, bv, gen):
    perm = torch.randperm(bk.size(1), generator=gen)
    return bk[:, perm, :], bv[:, perm, :]


def zone_c_test(model, seed=1234):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    xa_c, xb_c, _, _ = generate_batch(N_TRAIN_C, seed=111)
    xa_t, xb_t, _, _ = generate_batch(N_TEST_C, seed=222)

    # Precompute the FROZEN board once (it is a fixed function of input).
    bk_tr, bv_tr = board_of(model, xa_c, xb_c)
    bk_te, bv_te = board_of(model, xa_t, xb_t)
    v0_tr = _tok_after(xb_c, value_marker(0)); v1_tr = _tok_after(xb_c, value_marker(1))
    v0_te = _tok_after(xb_t, value_marker(0)); v1_te = _tok_after(xb_t, value_marker(1))

    zc = ZoneC(n_query=N_QUERY_C)
    opt_c = torch.optim.Adam(zc.parameters(), lr=LR_C)
    ce = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(bk_tr, bv_tr, v0_tr, v1_tr),
        batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    print("\nSTEP 3 -- train Zone C (blind consumer) under PERMUTED slots")
    print(f"  {'ep':>3}  {'v0(perm)':>9}  {'v1(perm)':>9}")

    def eval_zc(permute):
        zc.eval()
        with torch.no_grad():
            bk, bv = (bk_te, bv_te)
            if permute:
                bk, bv = _permute_slots(bk, bv, gen)
            l0, l1 = zc(bk, bv)
            a0 = (l0.argmax(1) == v0_te).float().mean().item()
            a1 = (l1.argmax(1) == v1_te).float().mean().item()
        zc.train()
        return a0, a1

    for ep in range(1, N_EPOCHS_C + 1):
        zc.train()
        for bk, bv, v0, v1 in loader:
            # per-batch slot permutation -> Zone C cannot use slot index
            bkp, bvp = _permute_slots(bk, bv, gen)
            l0, l1 = zc(bkp, bvp)
            loss = ce(l0, v0) + ce(l1, v1)
            opt_c.zero_grad(); loss.backward(); opt_c.step()
        a0p, a1p = eval_zc(permute=True)
        print(f"  {ep:>3}  {a0p:>9.4f}  {a1p:>9.4f}")

    a0_perm, a1_perm = eval_zc(permute=True)
    a0_unp,  a1_unp  = eval_zc(permute=False)
    return (a0_perm, a1_perm), (a0_unp, a1_unp)


def main():
    model, freeze_ep, key_sim, (xa_te, xb_te) = reproduce_and_select()
    verified = freeze_and_verify(model, xa_te, xb_te)
    (a0p, a1p), (a0u, a1u) = zone_c_test(model)

    # Verdict on PERMUTED accuracies (the honest key-matching numbers).
    # An index-reliant consumer would pass unpermuted but fail permuted.
    if a0p >= ZONEC_TH and a1p >= ZONEC_TH:
        verdict = "BUILDABLE"
    elif (a0p >= ZONEC_TH) != (a1p >= ZONEC_TH):
        which = "v0" if a0p >= ZONEC_TH else "v1"
        verdict = f"PARTIAL(only_{which})"
    else:
        verdict = "NOT_BUILDABLE"

    print("\n" + "=" * 72)
    print(f"  Zone C permuted   : v0={a0p:.4f}  v1={a1p:.4f}")
    print(f"  Zone C unpermuted : v0={a0u:.4f}  v1={a1u:.4f}   (contrast: index-reliance check)")
    print("=" * 72)
    print(f"\nM1-Part4: FREEZE_EPOCH={freeze_ep} KEY_SIM={key_sim:.4f} "
          f"FREEZE_VERIFIED={'yes' if verified else 'no'} "
          f"ZONEC_V0={a0p:.4f} ZONEC_V1={a1p:.4f} PERMUTED=yes VERDICT={verdict}")


if __name__ == "__main__":
    main()
