"""
SIG-Part3 -- VARIABLE-ARITY structured value on the board; blind consumer must
respect VALIDITY (present vs empty).

Zone A holds a variable-arity signature (name + 1..4 params) and writes each
param to its fixed ordinal value-slot plus a per-param validity slot; empty
params get a zero value + an EMPTY validity vector.  Zone B queries (param index
+ probe) and emits the binary "exists AND matches" label.  Then a fresh blind
Zone C (per-param heads, out = value 0..7 or EMPTY=8) must, for each param slot,
recover the value if present or say EMPTY if absent -- under permuted slots.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sig_task_var import (generate_batch, encode_a, encode_b, N_PARAM, PARAM_V)
from model_sig_var import PhoenixSigVar, D_LOCAL, VAL_SLOTS
from board import make_board
from run_m1_part3 import anneal_factor
from zone_c import ZoneC

warnings.filterwarnings("ignore", category=UserWarning)

N_TRAIN    = 40_000
N_EPOCHS   = 25
BATCH_SIZE = 256
LR         = 1e-3
AUX_WEIGHT = 2.0
ALPHA_FLOOR = 0.10
CKPT_PATH  = "sig_part3_ckpt.pt"

N_TRAIN_C  = 20_000
N_TEST_C   = 4_096
N_EPOCHS_C = 40
LR_C       = 1e-3
N_QUERY_C  = 4
EMPTY_CLS  = PARAM_V          # Zone C's "EMPTY" class index (= 8)
TH         = 0.90


def _value_sim(vals):
    normed = F.normalize(vals, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2)).mean(0)
    n = sim.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=vals.device)
    return sim[off].mean().item()


def board_of(model, toks_a, toks_b):
    with torch.no_grad():
        h_a = model.zone_a.encode(toks_a)
        h_b = model.zone_b.encode(toks_b)
        keys_a, vals_a = model.zone_a.write(h_a, toks_a)
        keys_b, vals_b = model.zone_b.write(h_b)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])
    return bk, bv


def _permute_slots(bk, bv, gen):
    perm = torch.randperm(bk.size(1), generator=gen)
    return bk[:, perm, :], bv[:, perm, :]


def train_main(seed=42):
    torch.manual_seed(seed)
    ar = torch.arange(N_TRAIN)
    name, params, present, query, probe, label = generate_batch(N_TRAIN, seed=0)
    ta = encode_a(name, params, present)
    tb = encode_b(query, probe)
    in_range = present[ar, query]
    vq = torch.where(in_range, params[ar, query], torch.full_like(query, -1))   # value target (ignore -1)
    valid_t = in_range.long()
    loader = DataLoader(
        TensorDataset(ta, tb, label, vq, valid_t),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    # held-out
    arv = torch.arange(4096)
    nv, pv, prv, qv, prbv, lv = generate_batch(4096, seed=999)
    tav, tbv = encode_a(nv, pv, prv), encode_b(qv, prbv)
    in_range_v = prv[arv, qv]
    vqv = torch.where(in_range_v, pv[arv, qv], torch.full_like(qv, -1))

    model = PhoenixSigVar()
    aux_value = nn.Linear(D_LOCAL, PARAM_V)   # recover queried value (in-range)
    aux_valid = nn.Linear(D_LOCAL, 2)         # present vs empty for the queried slot

    params_all = list(model.parameters()) + list(aux_value.parameters()) + list(aux_valid.parameters())
    opt = torch.optim.Adam(params_all, lr=LR)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_val = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

    print(f"frozen keys grad-check: zone_a.slot_keys.requires_grad="
          f"{model.zone_a.slot_keys.requires_grad}")

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = model(tav, tbv)
            pred = logits.argmax(1)
            task_acc = (pred == lv).float().mean().item()
            oor = ~in_range_v
            oor_acc = (pred[oor] == lv[oor]).float().mean().item() if oor.any() else 1.0
            hbm = model.h_b2.mean(1)
            av_valid = (aux_valid(hbm).argmax(1) == in_range_v.long()).float().mean().item()
            avm = in_range_v
            av_value = (aux_value(hbm).argmax(1)[avm] == vqv[avm]).float().mean().item()
            vsim = _value_sim(model.board_vals[:, VAL_SLOTS, :])
        model.train()
        return task_acc, oor_acc, av_value, av_valid, vsim

    print(f"  {'ep':>3}  {'task':>7}  {'oor':>7}  {'av_val':>7}  {'av_vld':>7}  "
          f"{'VALUE_SIM':>9}  {'gB':>6}  {'w_aux':>6}")

    checked = False
    for epoch in range(1, N_EPOCHS + 1):
        af = anneal_factor(epoch, N_EPOCHS)
        wa = AUX_WEIGHT * af
        model.train()
        for tba, tbb, lb, vqb, vldb in loader:
            opt.zero_grad()
            logits = model(tba, tbb)
            loss = ce(logits, lb)
            if wa > 0.0:
                hbm = model.h_b2.mean(1)
                loss = loss + wa * (ce_val(aux_value(hbm), vqb) + ce(aux_valid(hbm), vldb))
            loss.backward()
            if not checked:
                print(f"FROZEN_KEYS_GRAD_CHECK: zone_a.slot_keys.grad={model.zone_a.slot_keys.grad} "
                      f"zone_b.slot_keys.grad={model.zone_b.slot_keys.grad} (None => untrained)")
                checked = True
            if wa == 0.0:
                torch.nn.utils.clip_grad_norm_(params_all, max_norm=5.0)
            opt.step()
            with torch.no_grad():
                model.zone_b.alpha.clamp_(min=ALPHA_FLOOR)

        if epoch % 3 == 0 or epoch == N_EPOCHS:
            ta_, oor_, avv, avl, vs = evaluate()
            print(f"  {epoch:>3}  {ta_:>7.4f}  {oor_:>7.4f}  {avv:>7.4f}  {avl:>7.4f}  "
                  f"{vs:>9.4f}  {model.zone_b.alpha.item():>6.3f}  {wa:>6.2f}")

    return model, *evaluate()


def zone_c_probe(model, seed=1234):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    nc, pc, prc, qc, prbc, lc = generate_batch(N_TRAIN_C, seed=111)
    nt, pt, prt, qt, prbt, lt = generate_batch(N_TEST_C, seed=222)
    tac, tbc = encode_a(nc, pc, prc), encode_b(qc, prbc)
    tat, tbt = encode_a(nt, pt, prt), encode_b(qt, prbt)

    bk_tr, bv_tr = board_of(model, tac, tbc)
    bk_te, bv_te = board_of(model, tat, tbt)
    # per-param target: value if present else EMPTY_CLS
    tgt_tr = torch.where(prc, pc, torch.full_like(pc, EMPTY_CLS))     # (N,4)
    tgt_te = torch.where(prt, pt, torch.full_like(pt, EMPTY_CLS))
    pres_te = prt                                                     # (N,4) present mask

    zc = ZoneC(n_query=N_QUERY_C, n_values=N_PARAM, out_dim=PARAM_V + 1)   # value 0..7 or EMPTY=8
    opt_c = torch.optim.Adam(zc.parameters(), lr=LR_C)
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
            outs = zc(bk, bv)                                   # N_PARAM heads, each (N, 9)
            pred = torch.stack([o.argmax(1) for o in outs], dim=1)   # (N, N_PARAM)
            present = pres_te
            model_present = pred < EMPTY_CLS                    # said a value
            value_ok = (pred == tgt_te) & present               # value correct AND present
            value_acc = value_ok.float().sum().item() / present.float().sum().item()
            validity_acc = (model_present == present).float().mean().item()
            empty = ~present
            false_present = (model_present & empty).float().sum().item() / empty.float().sum().item()
            false_empty = ((~model_present) & present).float().sum().item() / present.float().sum().item()
        zc.train()
        return value_acc, validity_acc, false_present, false_empty

    print("\nZone C (blind probe -- NOT trained into the model) under PERMUTED slots")
    print(f"  {'ep':>3}  {'value':>7}  {'validity':>8}  {'false_pres':>10}")
    for ep in range(1, N_EPOCHS_C + 1):
        zc.train()
        for bk, bv, tg in loader:
            bkp, bvp = _permute_slots(bk, bv, gen)
            outs = zc(bkp, bvp)
            loss = sum(ce(outs[f], tg[:, f]) for f in range(N_PARAM))
            opt_c.zero_grad(); loss.backward(); opt_c.step()
        if ep % 5 == 0 or ep == N_EPOCHS_C:
            va, vl, fp, fe = eval_zc(permute=True)
            print(f"  {ep:>3}  {va:>7.4f}  {vl:>8.4f}  {fp:>10.4f}")

    return eval_zc(permute=True), eval_zc(permute=False)


def main():
    model, task_acc, oor_acc, av_value, av_valid, vsim = train_main()

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\ncheckpoint saved -> {CKPT_PATH}")

    (zv, zvl, fp, fe), (zv_u, zvl_u, fp_u, fe_u) = zone_c_probe(model)

    if task_acc < TH:
        verdict = f"NOT_BUILDABLE(task_fail={task_acc:.3f})"
    elif zv < TH:
        verdict = "NOT_BUILDABLE"
    elif zvl < TH:
        verdict = "PARTIAL_VALIDITY"
    else:
        verdict = "BUILDABLE"

    print("\n" + "=" * 72)
    print(f"  TASK_ACC={task_acc:.4f}  OOR_ACC={oor_acc:.4f}  (native AV_value={av_value:.4f} AV_valid={av_valid:.4f})")
    print(f"  VALUE_SIM={vsim:.4f}")
    print(f"  Zone C permuted   : value={zv:.4f} validity={zvl:.4f} false_present={fp:.4f} false_empty={fe:.4f}")
    print(f"  Zone C unpermuted : value={zv_u:.4f} validity={zvl_u:.4f} false_present={fp_u:.4f}")
    print("=" * 72)
    print(f"\nSIG-Part3: TASK_ACC={task_acc:.4f} OOR_ACC={oor_acc:.4f} ZONEC_VALUE={zv:.4f} "
          f"ZONEC_VALIDITY={zvl:.4f} FALSE_PRESENT={fp:.4f} VERDICT={verdict}")


if __name__ == "__main__":
    main()
