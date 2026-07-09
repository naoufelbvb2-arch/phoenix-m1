"""
COMP-Part2 -- can a BLIND, MINIMAL consumer COMPOSE a relational decision from
the frozen board (not just retrieve a field)?

Zone A writes ret/param to fixed slots; Zone B (relator) reads both and emits the
same-category label.  Then two fresh MINIMAL Zone Cs (fixed queries + a small MLP
head; NO transformer / processing layers) are attached to the frozen board:
  * ZONEC_RETRIEVE : recover each field's value (sanity).
  * ZONEC_COMPOSE  : output the relational same-category LABEL by combining the
                     two field-slots (the real composability test).
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from comp_task import generate_batch, encode, MASK_ID, T_V
from model_comp import PhoenixComp, D_LOCAL, SEQ_B, A_SLOTS
from board import make_board
from run_m1_part3 import anneal_factor
from zone_c import ZoneC

warnings.filterwarnings("ignore", category=UserWarning)

N_TRAIN    = 30_000
N_EPOCHS   = 20
BATCH_SIZE = 256
LR         = 1e-3
AUX_WEIGHT = 2.0
ALPHA_FLOOR = 0.10
CKPT_PATH  = "comp_part2_ckpt.pt"

N_TRAIN_C  = 20_000
N_TEST_C   = 4_096
N_EPOCHS_C = 40
LR_C       = 1e-3
N_QUERY_C  = 4
TH         = 0.90


def encode_b(n):
    return torch.full((n, SEQ_B), MASK_ID, dtype=torch.long)


def _value_sim(vals):
    normed = F.normalize(vals, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2)).mean(0)
    k = sim.size(0)
    off = ~torch.eye(k, dtype=torch.bool, device=vals.device)
    return sim[off].mean().item()


def board_of(model, toks_a, toks_b):
    with torch.no_grad():
        h_a = model.zone_a.encode(toks_a)
        h_b = model.zone_b.encode(toks_b)
        keys_a, vals_a = model.zone_a.write(h_a)
        keys_b, vals_b = model.zone_b.write(h_b)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])
    return bk, bv


def _permute_slots(bk, bv, gen):
    perm = torch.randperm(bk.size(1), generator=gen)
    return bk[:, perm, :], bv[:, perm, :]


def train_main(seed=42):
    torch.manual_seed(seed)
    ret, param, label = generate_batch(N_TRAIN, seed=0)
    ta = encode(ret, param, "full")
    tb = encode_b(N_TRAIN)
    loader = DataLoader(
        TensorDataset(ta, tb, label, ret, param),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    rv, pv, lv = generate_batch(4096, seed=999)
    tav, tbv = encode(rv, pv, "full"), encode_b(4096)

    model = PhoenixComp()
    aux_ret = nn.Linear(SEQ_B * D_LOCAL, T_V)     # recover ret / param from B's read
    aux_par = nn.Linear(SEQ_B * D_LOCAL, T_V)     # (drives A's write + B's read)

    params_all = list(model.parameters()) + list(aux_ret.parameters()) + list(aux_par.parameters())
    opt = torch.optim.Adam(params_all, lr=LR)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"frozen keys grad-check: zone_a.slot_keys.requires_grad="
          f"{model.zone_a.slot_keys.requires_grad}")

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = model(tav, tbv)
            task_acc = (logits.argmax(1) == lv).float().mean().item()
            hbf = model.h_b2.reshape(model.h_b2.size(0), -1)
            av_ret = (aux_ret(hbf).argmax(1) == rv).float().mean().item()
            av_par = (aux_par(hbf).argmax(1) == pv).float().mean().item()
            vsim = _value_sim(model.board_vals[:, :A_SLOTS, :])
        model.train()
        return task_acc, av_ret, av_par, vsim

    print(f"  {'ep':>3}  {'task':>7}  {'av_ret':>7}  {'av_par':>7}  {'VALUE_SIM':>9}  {'gB':>6}  {'w_aux':>6}")
    checked = False
    for epoch in range(1, N_EPOCHS + 1):
        af = anneal_factor(epoch, N_EPOCHS)
        wa = AUX_WEIGHT * af
        model.train()
        for tba, tbb, lb, rb, pb in loader:
            opt.zero_grad()
            logits = model(tba, tbb)
            loss = ce(logits, lb)
            if wa > 0.0:
                hbf = model.h_b2.reshape(model.h_b2.size(0), -1)
                loss = loss + wa * (ce(aux_ret(hbf), rb) + ce(aux_par(hbf), pb))
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
            ta_, ar_, ap_, vs = evaluate()
            print(f"  {epoch:>3}  {ta_:>7.4f}  {ar_:>7.4f}  {ap_:>7.4f}  {vs:>9.4f}  "
                  f"{model.zone_b.alpha.item():>6.3f}  {wa:>6.2f}")

    return model, *evaluate()


def _train_zc(zc, bk_tr, bv_tr, targets_tr, gen, seed):
    opt_c = torch.optim.Adam(zc.parameters(), lr=LR_C)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(bk_tr, bv_tr, targets_tr),
        batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for ep in range(N_EPOCHS_C):
        zc.train()
        for bk, bv, tg in loader:
            bkp, bvp = _permute_slots(bk, bv, gen)
            outs = zc(bkp, bvp)
            loss = sum(ce(outs[f], tg[:, f]) for f in range(len(outs)))
            opt_c.zero_grad(); loss.backward(); opt_c.step()


def zone_c_probe(model, seed=1234):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    rc, pc, lc = generate_batch(N_TRAIN_C, seed=111)
    rt, pt, lt = generate_batch(N_TEST_C, seed=222)
    bk_tr, bv_tr = board_of(model, encode(rc, pc, "full"), encode_b(N_TRAIN_C))
    bk_te, bv_te = board_of(model, encode(rt, pt, "full"), encode_b(N_TEST_C))

    # RETRIEVE: 2 heads recover ret / param values
    zc_ret = ZoneC(n_query=N_QUERY_C, n_values=2, out_dim=T_V)
    _train_zc(zc_ret, bk_tr, bv_tr, torch.stack([rc, pc], 1), gen, seed)

    # COMPOSE: 1 head outputs the relational same-category label (minimal!)
    zc_comp = ZoneC(n_query=N_QUERY_C, n_values=1, out_dim=2)
    _train_zc(zc_comp, bk_tr, bv_tr, lc.view(-1, 1), gen, seed)

    zc_cap = sum(p.numel() for p in zc_comp.parameters())
    cap_desc = (f"fixed_{N_QUERY_C}q+MLP(256-128-2)_no_transformer_{zc_cap//1000}Kparams")

    def eval_c(permute):
        zc_ret.eval(); zc_comp.eval()
        with torch.no_grad():
            bk, bv = bk_te, bv_te
            if permute:
                bk, bv = _permute_slots(bk, bv, gen)
            r_out = zc_ret(bk, bv)
            rr = (r_out[0].argmax(1) == rt).float().mean().item()
            pp = (r_out[1].argmax(1) == pt).float().mean().item()
            cc = (zc_comp(bk, bv)[0].argmax(1) == lt).float().mean().item()
        zc_ret.train(); zc_comp.train()
        return rr, pp, cc

    print("\nZone C (minimal blind probes, permuted) -- RETRIEVE (2 heads) + COMPOSE (1 head)")
    return eval_c(True), eval_c(False), cap_desc


def main():
    model, task_acc, av_ret, av_par, vsim = train_main()

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\ncheckpoint saved -> {CKPT_PATH}")

    (rr, pp, cc), (rr_u, pp_u, cc_u), cap = zone_c_probe(model)
    retrieve = min(rr, pp)
    compose = cc

    if retrieve < TH:
        verdict = "NOT_BUILDABLE"
    elif compose >= TH and task_acc >= TH:
        verdict = "COMPOSABLE"
    elif compose < TH:
        verdict = "RETRIEVABLE_NOT_COMPOSABLE"
    else:
        verdict = f"NOT_BUILDABLE(task_fail={task_acc:.3f})"

    print("\n" + "=" * 72)
    print(f"  TASK_ACC={task_acc:.4f}  (native AV_ret={av_ret:.4f} AV_param={av_par:.4f})  VALUE_SIM={vsim:.4f}")
    print(f"  Zone C permuted   : retrieve_ret={rr:.4f} retrieve_param={pp:.4f} COMPOSE={cc:.4f}")
    print(f"  Zone C unpermuted : retrieve_ret={rr_u:.4f} retrieve_param={pp_u:.4f} COMPOSE={cc_u:.4f}")
    print(f"  Zone C compose capacity: {cap}")
    print("=" * 72)
    print(f"\nCOMP-Part2: TASK_ACC={task_acc:.4f} ZONEC_RETRIEVE={retrieve:.4f} "
          f"ZONEC_COMPOSE={compose:.4f} ZONEC_CAPACITY={cap} VERDICT={verdict}")


if __name__ == "__main__":
    main()
