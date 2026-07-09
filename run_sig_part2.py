"""
SIG-Part2 -- structured (3-field) value on the two-zone board; blind per-field
selective-extraction verdict.

Trains PhoenixSig (Zone A writes each signature field to its own fixed slot;
Zone B query-conditionally reads the right field and emits the binary match),
then freezes everything and attaches a FRESH blind Zone C (measurement probe
only) with THREE field-heads.  Each head must reconstruct one field's value from
the frozen board by key-matching its fixed slot, under permuted slot order.

Reports TASK_ACC (does Zone B produce the correct match) separately from the
per-field blind reconstruction (ZONEC_NAME/RET/PARAM).
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sig_task import generate_batch, encode, MASK_ID, SEQ_LEN, NAME_V
from model_sig import PhoenixSig, D_LOCAL
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
CKPT_PATH  = "sig_part2_ckpt.pt"

N_TRAIN_C  = 20_000
N_TEST_C   = 4_096
N_EPOCHS_C = 40
LR_C       = 1e-3
N_QUERY_C  = 4
TASK_TH    = 0.90
ZONEC_TH   = 0.90


def encode_zoneA(name, ret, param):
    """Zone A view: [name, ret, param, MASK, MASK] (query/probe hidden)."""
    n = name.size(0)
    toks = torch.full((n, SEQ_LEN), MASK_ID, dtype=torch.long)
    toks[:, 0] = name
    toks[:, 1] = ret
    toks[:, 2] = param
    return toks


def _value_sim(vals):
    normed = F.normalize(vals, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2)).mean(0)
    n = sim.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=vals.device)
    return sim[off].mean().item()


def board_of(model, toks_a, toks_b):
    with torch.no_grad():
        h_a = model.zone_a.trunk_encode(toks_a)
        h_b = model.zone_b.trunk_encode(toks_b)
        keys_a, vals_a = model.zone_a.write(h_a, model.FIELD_POS)
        keys_b, vals_b = model.zone_b.write(h_b, model.QP_POS)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])
    return bk, bv


def _permute_slots(bk, bv, gen):
    perm = torch.randperm(bk.size(1), generator=gen)
    return bk[:, perm, :], bv[:, perm, :]


def train_main(seed=42):
    torch.manual_seed(seed)
    name, ret, param, query, probe, label = generate_batch(N_TRAIN, seed=0)
    ta = encode_zoneA(name, ret, param)
    tb = encode(name, ret, param, query, probe, "nosig")   # querier view
    sig = torch.stack([name, ret, param], 1)
    sq = sig[torch.arange(N_TRAIN), query]                 # queried field value
    loader = DataLoader(
        TensorDataset(ta, tb, label, query, sq),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    nv, rv, pv, qv, prv, lv = generate_batch(4096, seed=999)
    tav = encode_zoneA(nv, rv, pv)
    tbv = encode(nv, rv, pv, qv, prv, "nosig")
    sigv = torch.stack([nv, rv, pv], 1); sqv = sigv[torch.arange(4096), qv]

    model = PhoenixSig()
    aux_field = nn.Linear(D_LOCAL, NAME_V)   # recover the queried field's value

    params = list(model.parameters()) + list(aux_field.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"frozen keys grad-check: zone_a.slot_keys.requires_grad="
          f"{model.zone_a.slot_keys.requires_grad}")

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = model(tav, tbv)
            task_acc = (logits.argmax(1) == lv).float().mean().item()
            af = aux_field(model.h_b2.mean(1)).argmax(1)
            av = []
            for q in range(3):
                mq = qv == q
                av.append((af[mq] == sqv[mq]).float().mean().item())
            vsim = _value_sim(model.board_vals[:, [0, 1, 2], :])   # 3 field slots
        model.train()
        return task_acc, av, vsim

    print(f"  {'ep':>3}  {'task':>7}  {'AV_name':>7}  {'AV_ret':>7}  {'AV_param':>8}  "
          f"{'VALUE_SIM':>9}  {'gA':>6}  {'gB':>6}  {'w_aux':>6}")

    checked = False
    for epoch in range(1, N_EPOCHS + 1):
        af = anneal_factor(epoch, N_EPOCHS)
        wa = AUX_WEIGHT * af
        model.train()
        for tba, tbb, lb, qb, sqb in loader:
            opt.zero_grad()
            logits = model(tba, tbb)
            loss = ce(logits, lb)
            if wa > 0.0:
                loss = loss + wa * ce(aux_field(model.h_b2.mean(1)), sqb)
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
            task_acc, av, vsim = evaluate()
            print(f"  {epoch:>3}  {task_acc:>7.4f}  {av[0]:>7.4f}  {av[1]:>7.4f}  {av[2]:>8.4f}  "
                  f"{vsim:>9.4f}  {model.zone_a.alpha.item():>6.3f}  {model.zone_b.alpha.item():>6.3f}  {wa:>6.2f}")

    return model, *evaluate()


def zone_c_probe(model, seed=1234):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    nc, rc, pc, qc, prc, lc = generate_batch(N_TRAIN_C, seed=111)
    nt, rt, pt, qt, prt, lt = generate_batch(N_TEST_C, seed=222)
    tac, tbc = encode_zoneA(nc, rc, pc), encode(nc, rc, pc, qc, prc, "nosig")
    tat, tbt = encode_zoneA(nt, rt, pt), encode(nt, rt, pt, qt, prt, "nosig")

    bk_tr, bv_tr = board_of(model, tac, tbc)
    bk_te, bv_te = board_of(model, tat, tbt)
    tgt_tr = torch.stack([nc, rc, pc], 1)      # (N,3) field values
    tgt_te = torch.stack([nt, rt, pt], 1)

    zc = ZoneC(n_query=N_QUERY_C, n_values=3, out_dim=NAME_V)
    opt_c = torch.optim.Adam(zc.parameters(), lr=LR_C)   # ONLY Zone C
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(bk_tr, bv_tr, tgt_tr),
        batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    def eval_zc(permute):
        zc.eval()
        with torch.no_grad():
            bk, bv = bk_te, bv_te
            if permute:
                bk, bv = _permute_slots(bk, bv, gen)
            outs = zc(bk, bv)          # 3 field logits
            accs = [(outs[f].argmax(1) == tgt_te[:, f]).float().mean().item() for f in range(3)]
        zc.train()
        return accs

    print("\nZone C (blind probe -- NOT trained into the model) under PERMUTED slots")
    print(f"  {'ep':>3}  {'name':>7}  {'ret':>7}  {'param':>7}")
    for ep in range(1, N_EPOCHS_C + 1):
        zc.train()
        for bk, bv, tg in loader:
            bkp, bvp = _permute_slots(bk, bv, gen)
            outs = zc(bkp, bvp)
            loss = sum(ce(outs[f], tg[:, f]) for f in range(3))
            opt_c.zero_grad(); loss.backward(); opt_c.step()
        if ep % 5 == 0 or ep == N_EPOCHS_C:
            a = eval_zc(permute=True)
            print(f"  {ep:>3}  {a[0]:>7.4f}  {a[1]:>7.4f}  {a[2]:>7.4f}")

    return eval_zc(permute=True), eval_zc(permute=False)


def main():
    model, task_acc, av, vsim = train_main()

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\ncheckpoint saved -> {CKPT_PATH}")

    perm, unperm = zone_c_probe(model)
    zn, zr, zp = perm
    field_names = ["name", "ret", "param"]
    below = [f for f, acc in zip(field_names, perm) if acc < ZONEC_TH]

    if task_acc < TASK_TH:
        verdict = f"NOT_BUILDABLE(task_fail={task_acc:.3f})"
    elif len(below) == 0:
        verdict = "BUILDABLE"
    elif len(below) == 1:
        verdict = f"PARTIAL(only_{below[0]}_resists)"
    else:
        verdict = f"NOT_BUILDABLE(fields<{ZONEC_TH}:{'+'.join(below)})"

    print("\n" + "=" * 72)
    print(f"  TASK_ACC={task_acc:.4f}  (Zone B match label on held-out)")
    print(f"  native routing: AV_NAME={av[0]:.4f} AV_RET={av[1]:.4f} AV_PARAM={av[2]:.4f}  VALUE_SIM={vsim:.4f}")
    print(f"  Zone C permuted   : name={zn:.4f} ret={zr:.4f} param={zp:.4f}")
    print(f"  Zone C unpermuted : name={unperm[0]:.4f} ret={unperm[1]:.4f} param={unperm[2]:.4f}   (index-reliance contrast)")
    print("=" * 72)
    print(f"\nSIG-Part2: TASK_ACC={task_acc:.4f} ZONEC_NAME={zn:.4f} ZONEC_RET={zr:.4f} "
          f"ZONEC_PARAM={zp:.4f} VALUE_SIM={vsim:.4f} VERDICT={verdict}")


if __name__ == "__main__":
    main()
