"""
M1-REBUILD -- test whether FROZEN ADDRESSES fix the addressing space.

Diagnosis being corrected: with learned keys, addresses never crystallize
(ADDRESSING_DRIFT) and a blind consumer could not decode the frozen board
(Part-4 = NOT_BUILDABLE).  Fix under test: slot KEYS are fixed orthogonal
buffers (never trained); zones learn only WHAT payload to place at each fixed
address and HOW to read.

Main training (model_m1_frozen.PhoenixM1Frozen):
  * losses = CE_A(head_a->label_A) + CE_B(head_b->label_B)
             + annealed per-value scaffold aux_v0/v1/v2 (Part-3 scaffold, ->0
               over first half).  Two heads exactly as model_m1.py.
  * track av0/av1/av2 (routing) and VALUE_SIM (Zone B payload differentiation;
    KEY_SIM is now trivially fixed so we report payload similarity instead).
  * confirm the frozen key buffers receive ZERO gradient.

Verdict (same blind Zone C as Part-4, permuted slots):
  v0>=0.90 AND v1>=0.90 -> BUILDABLE ; one>=0.90 -> PARTIAL ;
  both<0.90 -> STILL_NOT_BUILDABLE (report av / VALUE_SIM to localize the cause).
"""

import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from task_m1 import generate_batch, key_marker, value_marker, VOCAB_SIZE
from model_m1_frozen import PhoenixM1Frozen, _extract, D_LOCAL
from board import make_board, K
from slot_analysis import _pairwise_cosine
from run_m1_part3 import _tok_after, anneal_factor
from zone_c import ZoneC

warnings.filterwarnings("ignore", category=UserWarning)

# ── main training ────────────────────────────────────────────────────────────
N_TRAIN    = 20_000
N_EPOCHS   = 30            # anneal scaffold over first 15
BATCH_SIZE = 128
LR         = 1e-3
AUX_WEIGHT = 2.0
ALPHA_FLOOR = 0.10        # keep the read gate alive (Safe-Zero ~0.1)

# ── Zone C forward-compat test ───────────────────────────────────────────────
N_TRAIN_C  = 20_000
N_TEST_C   = 4_096
N_EPOCHS_C = 40           # train to convergence (Part-4 under-trained v1 at 15)
LR_C       = 1e-3
N_QUERY_C  = 4
ZONEC_TH   = 0.90


def board_of(model, x_a, x_b):
    with torch.no_grad():
        h_a = model.zone_a.encode(x_a)
        h_b = model.zone_b.encode(x_b)
        keys_a, vals_a = model.zone_a.write(h_a)
        keys_b, vals_b = model.zone_b.write(h_b)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])
    return bk, bv


def _permute_slots(bk, bv, gen):
    perm = torch.randperm(bk.size(1), generator=gen)
    return bk[:, perm, :], bv[:, perm, :]


def train_main(seed=42):
    torch.manual_seed(seed)
    xa_tr, xb_tr, ya_tr, yb_tr = generate_batch(N_TRAIN, seed=0)
    loader = DataLoader(
        TensorDataset(xa_tr, xb_tr, ya_tr, yb_tr),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xa_te, xb_te, ya_te, yb_te = generate_batch(4096, seed=9999)

    model = PhoenixM1Frozen()

    # reader-side routing aux (annealed) -- the Part-3 scaffold
    aux_v0 = nn.Linear(D_LOCAL, VOCAB_SIZE)
    aux_v1 = nn.Linear(D_LOCAL, VOCAB_SIZE)
    aux_v2 = nn.Linear(D_LOCAL, VOCAB_SIZE)

    params = (list(model.parameters())
              + list(aux_v0.parameters()) + list(aux_v1.parameters()) + list(aux_v2.parameters()))
    opt = torch.optim.Adam(params, lr=LR)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── frozen-key sanity: orthogonality + non-trainable ─────────────────────
    all_keys = torch.cat([model.zone_a.slot_keys, model.zone_b.slot_keys], 0)  # (2K, D_ADDR)
    kn = torch.nn.functional.normalize(all_keys, dim=-1)
    cos = kn @ kn.T
    off = cos[~torch.eye(2 * K, dtype=torch.bool)]
    print(f"frozen keys: shape={tuple(all_keys.shape)} "
          f"max|off-diag cos|={off.abs().max().item():.2e} "
          f"(orthogonal), zone_a.slot_keys.requires_grad={model.zone_a.slot_keys.requires_grad}")

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
            value_sim = _pairwise_cosine(vals_b)[0]   # Zone B payload similarity
        model.train()
        return av0, av1, av2, value_sim

    print(f"  {'ep':>3}  {'av0':>7}  {'av1':>7}  {'av2':>7}  {'VALUE_SIM':>9}  "
          f"{'gate_A':>7}  {'gate_B':>7}  {'w_aux':>6}")

    checked_grad = False
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

            if not checked_grad:
                ga = model.zone_a.slot_keys.grad
                gb = model.zone_b.slot_keys.grad
                print(f"FROZEN_KEYS_GRAD_CHECK: zone_a.slot_keys.grad={ga} "
                      f"zone_b.slot_keys.grad={gb} (both None/zero => addresses untrained)")
                checked_grad = True

            if wa == 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            with torch.no_grad():
                model.zone_a.alpha.clamp_(min=ALPHA_FLOOR)
                model.zone_b.alpha.clamp_(min=ALPHA_FLOOR)

        if epoch % 3 == 0 or epoch == N_EPOCHS:
            av0, av1, av2, vsim = eval_all()
            print(f"  {epoch:>3}  {av0:>7.4f}  {av1:>7.4f}  {av2:>7.4f}  {vsim:>9.4f}  "
                  f"{model.zone_a.alpha.item():>7.4f}  {model.zone_b.alpha.item():>7.4f}  "
                  f"{wa:>6.2f}")

    av0, av1, av2, vsim = eval_all()
    return model, av0, av1, av2, vsim


def zone_c_test(model, seed=1234):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    xa_c, xb_c, _, _ = generate_batch(N_TRAIN_C, seed=111)
    xa_t, xb_t, _, _ = generate_batch(N_TEST_C, seed=222)

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

    print("\nZone C (blind consumer) -- train under PERMUTED slots")
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
    model, av0, av1, av2, vsim = train_main()

    # freeze everything
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    (a0p, a1p), (a0u, a1u) = zone_c_test(model)

    if a0p >= ZONEC_TH and a1p >= ZONEC_TH:
        verdict = "BUILDABLE"
    elif (a0p >= ZONEC_TH) != (a1p >= ZONEC_TH):
        which = "v0" if a0p >= ZONEC_TH else "v1"
        verdict = f"PARTIAL(only_{which})"
    else:
        verdict = "STILL_NOT_BUILDABLE"

    print("\n" + "=" * 72)
    print(f"  routing: av0={av0:.4f} av1={av1:.4f} av2={av2:.4f}  VALUE_SIM(ZoneB)={vsim:.4f}")
    print(f"  Zone C permuted   : v0={a0p:.4f}  v1={a1p:.4f}")
    print(f"  Zone C unpermuted : v0={a0u:.4f}  v1={a1u:.4f}   (index-reliance contrast)")
    print("=" * 72)
    print(f"\nM1-Rebuild: AV0={av0:.4f} AV1={av1:.4f} AV2={av2:.4f} VALUE_SIM={vsim:.4f} "
          f"ZONEC_V0={a0p:.4f} ZONEC_V1={a1p:.4f} PERMUTED=yes VERDICT={verdict}")


if __name__ == "__main__":
    main()
