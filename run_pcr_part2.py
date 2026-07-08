"""
PCR-Part2 -- two-zone board on the frozen PCR task; blind-consumer verdict.

Trains PhoenixPCR (Proposer/Checker over a passive board with FROZEN addresses +
FIXED write-assignment) to run the execute->check->repair loop across the board,
then freezes everything and attaches a FRESH blind Zone C (measurement probe
only, never trained into the model) to test whether BOTH crossed values -- x and
T -- are blind-addressable from the frozen board under permuted slot order.

Reports the actual TASK accuracy (does Zone A emit corrected output = T)
separately from Zone C's addressability.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pcr_task import generate_batch, encode, M
from model_pcr import PhoenixPCR, D_LOCAL
from board import make_board, K
from run_m1_part3 import anneal_factor
from zone_c import ZoneC

warnings.filterwarnings("ignore", category=UserWarning)

N_TRAIN    = 40_000
N_EPOCHS   = 25
BATCH_SIZE = 256
LR         = 1e-3
AUX_WEIGHT = 2.0
ALPHA_FLOOR = 0.10
CKPT_PATH  = "pcr_part2_ckpt.pt"

N_TRAIN_C  = 20_000
N_TEST_C   = 4_096
N_EPOCHS_C = 40
LR_C       = 1e-3
N_QUERY_C  = 4
TASK_TH    = 0.90
ZONEC_TH   = 0.90


def _value_sim(vals):
    """Mean off-diagonal cosine among (B, n, D) payloads (any n)."""
    normed = F.normalize(vals, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2)).mean(0)
    n = sim.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=vals.device)
    return sim[off].mean().item()


def board_of(model, toks_a, toks_b):
    with torch.no_grad():
        h_a = model.zone_a.trunk_encode(toks_a)
        h_b = model.zone_b.trunk_encode(toks_b)
        keys_a, vals_a = model.zone_a.write(h_a)
        keys_b, vals_b = model.zone_b.write(h_b)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])
    return bk, bv


def _permute_slots(bk, bv, gen):
    perm = torch.randperm(bk.size(1), generator=gen)
    return bk[:, perm, :], bv[:, perm, :]


def train_main(seed=42):
    torch.manual_seed(seed)
    s0, ops, T, x, r = generate_batch(N_TRAIN, seed=0)
    ta = encode(s0, ops, T, "prop")     # proposer view (T masked)
    tb = encode(s0, ops, T, "check")    # checker view  (s0, ops masked)
    loader = DataLoader(
        TensorDataset(ta, tb, T, x, r),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    s0v, opsv, Tv, xv, rv = generate_batch(4096, seed=999)
    tav = encode(s0v, opsv, Tv, "prop")
    tbv = encode(s0v, opsv, Tv, "check")

    model = PhoenixPCR()
    aux_T = nn.Linear(D_LOCAL, M)   # A recovers T (from read of B's slot)
    aux_x = nn.Linear(D_LOCAL, M)   # B recovers x (from read of A's slot)

    params = list(model.parameters()) + list(aux_T.parameters()) + list(aux_x.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"frozen keys grad-check: zone_a.slot_keys.requires_grad="
          f"{model.zone_a.slot_keys.requires_grad}")

    def evaluate():
        model.eval()
        with torch.no_grad():
            r_logits = model(tav, tbv)
            r_pred = r_logits.argmax(1)
            task_acc = (((xv + r_pred) % M) == Tv).float().mean().item()
            av_T = (aux_T(model.h_a2.mean(1)).argmax(1) == Tv).float().mean().item()
            av_x = (aux_x(model.h_b2.mean(1)).argmax(1) == xv).float().mean().item()
            vsim = _value_sim(model.board_vals[:, [0, K], :])   # x-slot, T-slot
        model.train()
        return task_acc, av_x, av_T, vsim

    print(f"  {'ep':>3}  {'task':>7}  {'av_x':>7}  {'av_T':>7}  {'VALUE_SIM':>9}  "
          f"{'gate_A':>7}  {'gate_B':>7}  {'w_aux':>6}")

    checked = False
    for epoch in range(1, N_EPOCHS + 1):
        af = anneal_factor(epoch, N_EPOCHS)
        wa = AUX_WEIGHT * af
        model.train()
        for tba, tbb, Tb, xb, rb in loader:
            opt.zero_grad()
            r_logits = model(tba, tbb)
            loss = ce(r_logits, rb)
            if wa > 0.0:
                loss = (loss
                        + wa * ce(aux_T(model.h_a2.mean(1)), Tb)
                        + wa * ce(aux_x(model.h_b2.mean(1)), xb))
            loss.backward()
            if not checked:
                print(f"FROZEN_KEYS_GRAD_CHECK: zone_a.slot_keys.grad={model.zone_a.slot_keys.grad} "
                      f"zone_b.slot_keys.grad={model.zone_b.slot_keys.grad} (None => untrained)")
                checked = True
            if wa == 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            with torch.no_grad():
                model.zone_a.alpha.clamp_(min=ALPHA_FLOOR)
                model.zone_b.alpha.clamp_(min=ALPHA_FLOOR)

        if epoch % 3 == 0 or epoch == N_EPOCHS:
            task_acc, av_x, av_T, vsim = evaluate()
            print(f"  {epoch:>3}  {task_acc:>7.4f}  {av_x:>7.4f}  {av_T:>7.4f}  {vsim:>9.4f}  "
                  f"{model.zone_a.alpha.item():>7.4f}  {model.zone_b.alpha.item():>7.4f}  {wa:>6.2f}")

    task_acc, av_x, av_T, vsim = evaluate()
    return model, task_acc, av_x, av_T, vsim


def zone_c_probe(model, seed=1234):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    s0c, opsc, Tc, xc, _ = generate_batch(N_TRAIN_C, seed=111)
    s0t, opst, Tt, xt, _ = generate_batch(N_TEST_C, seed=222)
    tac, tbc = encode(s0c, opsc, Tc, "prop"), encode(s0c, opsc, Tc, "check")
    tat, tbt = encode(s0t, opst, Tt, "prop"), encode(s0t, opst, Tt, "check")

    bk_tr, bv_tr = board_of(model, tac, tbc)
    bk_te, bv_te = board_of(model, tat, tbt)

    zc = ZoneC(n_query=N_QUERY_C)
    opt_c = torch.optim.Adam(zc.parameters(), lr=LR_C)   # ONLY Zone C
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(bk_tr, bv_tr, xc, Tc),
        batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    def eval_zc(permute):
        zc.eval()
        with torch.no_grad():
            bk, bv = bk_te, bv_te
            if permute:
                bk, bv = _permute_slots(bk, bv, gen)
            lx, lt = zc(bk, bv)          # head0 -> x, head1 -> T
            ax = (lx.argmax(1) == xt).float().mean().item()
            at = (lt.argmax(1) == Tt).float().mean().item()
        zc.train()
        return ax, at

    print("\nZone C (blind probe -- NOT trained into the model) under PERMUTED slots")
    print(f"  {'ep':>3}  {'x(perm)':>9}  {'T(perm)':>9}")
    for ep in range(1, N_EPOCHS_C + 1):
        zc.train()
        for bk, bv, xb, Tb in loader:
            bkp, bvp = _permute_slots(bk, bv, gen)
            lx, lt = zc(bkp, bvp)
            loss = ce(lx, xb) + ce(lt, Tb)
            opt_c.zero_grad(); loss.backward(); opt_c.step()
        if ep % 5 == 0 or ep == N_EPOCHS_C:
            ax, at = eval_zc(permute=True)
            print(f"  {ep:>3}  {ax:>9.4f}  {at:>9.4f}")

    return eval_zc(permute=True), eval_zc(permute=False)


def main():
    model, task_acc, av_x, av_T, vsim = train_main()

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\ncheckpoint saved -> {CKPT_PATH}")

    (zx_p, zt_p), (zx_u, zt_u) = zone_c_probe(model)

    if task_acc < TASK_TH:
        verdict = f"NOT_BUILDABLE(task_fail={task_acc:.3f})"
    elif zx_p >= ZONEC_TH and zt_p >= ZONEC_TH:
        verdict = "BUILDABLE"
    elif (zx_p >= ZONEC_TH) != (zt_p >= ZONEC_TH):
        which = "x" if zx_p >= ZONEC_TH else "T"
        verdict = f"PARTIAL(only_{which})"
    else:
        verdict = "NOT_BUILDABLE"

    print("\n" + "=" * 72)
    print(f"  TASK_ACC={task_acc:.4f}  (Zone A corrected output == T on held-out)")
    print(f"  native routing: av_x={av_x:.4f} av_T={av_T:.4f}  VALUE_SIM={vsim:.4f}")
    print(f"  Zone C permuted   : x={zx_p:.4f}  T={zt_p:.4f}")
    print(f"  Zone C unpermuted : x={zx_u:.4f}  T={zt_u:.4f}   (index-reliance contrast)")
    print("=" * 72)
    print(f"\nPCR-Part2: TASK_ACC={task_acc:.4f} AV_X={av_x:.4f} AV_T={av_T:.4f} "
          f"VALUE_SIM={vsim:.4f} ZONEC_X={zx_p:.4f} ZONEC_T={zt_p:.4f} VERDICT={verdict}")


if __name__ == "__main__":
    main()
