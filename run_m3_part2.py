"""
M3-Part2 -- FUNCTIONAL plug-and-play under crowding: can the appended Zone D
IGNITE on the already-populated frozen board and compose a NEW decision that
needs BOTH old zones?

  d_label = category(params[query]) XOR result        (over IN-RANGE queries)
    category(v) = v // (PARAM_V//2)  -- a NEW high/low partition SIG never used
    result      = B's processed decision (== an independent fair coin)

Reports three separate things:
  DTASK_ACC  : D solves its new task on held-out (>=90% = D functions).
  NEEDS_BOTH : ablate A's slots from D -> DACC_NO_A should collapse to chance;
               ablate B's slots from D -> DACC_NO_B should collapse to chance.
  OLD_STABLE : after ALL D training, A/B SIG-3 outputs stay bit-identical.

Verdict:
  DTASK>=.90 & both ablations ~chance & OLD_STABLE(0.0) -> PLUG_AND_PLAY
  DTASK>=.90 but only one ablation collapses            -> USES_ONE_ONLY(which)
  DTASK<.90                                             -> IGNITION_FAILED
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sig_task_var import generate_batch, encode_a, encode_b, N_PARAM
from model_sig_var import PhoenixSigVar
from model_m3_part2 import (
    PhoenixM3Part2, slot_mask, CATEGORY_DIV,
    A_BLOCK, B_BLOCK, D_BLOCK,
)
from run_m1_part3 import anneal_factor

warnings.filterwarnings("ignore", category=UserWarning)

CKPT_IN   = "sig_part3_ckpt.pt"       # frozen SIG-Part3 zones A/B (== A/B in m3_part1)
CKPT_OUT  = "m3_part2_ckpt.pt"
N_TRAIN   = 40_000
N_TEST    = 8_192
N_EPOCHS  = 30
BATCH     = 256
LR        = 1e-3
AUX_W     = 1.0
N_PTR_D   = 6
TH        = 0.90
CHANCE_TH = 0.60          # ablation must fall below this to count as "~chance"
OLD_SEED  = 20260709      # fixed batch for OLD_STABLE bit-exact check


def make_dataset(n, seed):
    """Return only IN-RANGE samples (queried field present on the board)."""
    name, params, present, query, probe, label = generate_batch(n, seed)
    ar = torch.arange(n)
    in_range = present[ar, query]
    idx = in_range.nonzero(as_tuple=True)[0]
    name, params, present = name[idx], params[idx], present[idx]
    query, probe, label = query[idx], probe[idx], label[idx]
    toks_a = encode_a(name, params, present)
    toks_b = encode_b(query, probe)
    pq = params[torch.arange(idx.numel()), query]         # queried value (present)
    category = (pq // CATEGORY_DIV)                         # NEW partition bit
    return toks_a, toks_b, query, pq, category, label      # label(old) == result(match)


def build_frozen_board(model, toks_a, toks_b, chunk=8192):
    bks, bvs, results = [], [], []
    with torch.no_grad():
        for i in range(0, toks_a.size(0), chunk):
            ta, tb = toks_a[i:i + chunk], toks_b[i:i + chunk]
            bk, bv = model.build_board(ta, tb)
            lb, _ = model.old_logits(ta, tb)
            bks.append(bk); bvs.append(bv); results.append(lb.argmax(1))
    return torch.cat(bks), torch.cat(bvs), torch.cat(results)


def main():
    torch.manual_seed(0)

    base = PhoenixSigVar()
    base.load_state_dict(torch.load(CKPT_IN, map_location="cpu"))
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    model = PhoenixM3Part2(base, n_ptr=N_PTR_D)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.zone_d.parameters():
        p.requires_grad_(True)

    # ---- OLD baseline captured BEFORE training D (fixed batch) --------------
    n_o, p_o, pr_o, q_o, pb_o, l_o = generate_batch(4096, OLD_SEED)
    ta_o, tb_o = encode_a(n_o, p_o, pr_o), encode_b(q_o, pb_o)
    with torch.no_grad():
        old_logits_before, _ = model.old_logits(ta_o, tb_o)
        ref_logits = base(ta_o, tb_o)
    base_vs_ref = (old_logits_before - ref_logits).abs().max().item()

    # ---- data / frozen boards ----------------------------------------------
    ta_tr, tb_tr, q_tr, pq_tr, cat_tr, res_tr = make_dataset(N_TRAIN, seed=101)
    ta_te, tb_te, q_te, pq_te, cat_te, res_te = make_dataset(N_TEST, seed=202)

    bk_tr, bv_tr, rb_tr = build_frozen_board(model, ta_tr, tb_tr)
    bk_te, bv_te, rb_te = build_frozen_board(model, ta_te, tb_te)

    # d_label uses the result bit ACTUALLY on the board (B's decision)
    dlab_tr = (cat_tr ^ rb_tr).long()
    dlab_te = (cat_te ^ rb_te).long()
    b_acc = (rb_tr == res_tr).float().mean().item()         # B correctness sanity
    print(f"data: train={bk_tr.size(0)} test={bk_te.size(0)}  "
          f"B_result_acc={b_acc:.4f}  d_label balance={dlab_tr.float().mean():.3f}")

    m_full = slot_mask([A_BLOCK, B_BLOCK, D_BLOCK])
    m_noA  = slot_mask([B_BLOCK, D_BLOCK])
    m_noB  = slot_mask([A_BLOCK, D_BLOCK])

    loader = DataLoader(
        TensorDataset(bk_tr, bv_tr, dlab_tr, q_tr, rb_tr, pq_tr),
        batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )

    opt = torch.optim.Adam(model.zone_d.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    def eval_acc(mask):
        model.zone_d.eval()
        with torch.no_grad():
            logits, _ = model.zone_d(bk_te, bv_te, mask)
            acc = (logits.argmax(1) == dlab_te).float().mean().item()
        model.zone_d.train()
        return acc

    print(f"  {'ep':>3}  {'DTASK':>7}  {'NO_A':>6}  {'NO_B':>6}  {'w_aux':>6}")
    for epoch in range(1, N_EPOCHS + 1):
        af = anneal_factor(epoch, N_EPOCHS)
        wa = AUX_W * af
        model.zone_d.train()
        for bk, bv, dl, qy, rs, pv in loader:
            opt.zero_grad()
            logits, reads = model.zone_d(bk, bv, m_full)
            loss = ce(logits, dl)
            if wa > 0.0:
                aux_q, aux_r, aux_v, sel = model.zone_d.aux_logits(reads)
                loss = loss + wa * (ce(aux_q, qy) + ce(aux_r, rs)
                                    + ce(aux_v, pv) + ce(sel, qy))   # sel: teach dereference
            loss.backward()
            opt.step()
        if epoch % 3 == 0 or epoch == N_EPOCHS:
            print(f"  {epoch:>3}  {eval_acc(m_full):>7.4f}  {eval_acc(m_noA):>6.4f}  "
                  f"{eval_acc(m_noB):>6.4f}  {wa:>6.2f}")

    dtask = eval_acc(m_full)
    dacc_noA = eval_acc(m_noA)
    dacc_noB = eval_acc(m_noB)

    # ---- OLD_STABLE re-check LAST, after all D training --------------------
    with torch.no_grad():
        old_logits_after, _ = model.old_logits(ta_o, tb_o)
    old_maxdiff = (old_logits_after - old_logits_before).abs().max().item()

    # ---- verdict -----------------------------------------------------------
    both_collapse = (dacc_noA < CHANCE_TH) and (dacc_noB < CHANCE_TH)
    stable = (old_maxdiff == 0.0)
    if dtask < TH:
        verdict = "IGNITION_FAILED"
    elif both_collapse and stable:
        verdict = "PLUG_AND_PLAY"
    elif stable and (dacc_noA < CHANCE_TH) != (dacc_noB < CHANCE_TH):
        leans = "A" if dacc_noB < CHANCE_TH else "B"   # the zone it still needs
        verdict = f"USES_ONE_ONLY(leans_on_{leans})"
    elif not stable:
        verdict = f"OLD_DRIFT({old_maxdiff:.2e})"
    else:
        verdict = "USES_ONE_ONLY(neither_collapsed)"

    torch.save(model.state_dict(), CKPT_OUT)
    print("\n" + "=" * 72)
    print(f"  baseline vs original SIG-3 model : max|dlogit| = {base_vs_ref:.3e}")
    print(f"  DTASK_ACC (full board)  = {dtask:.4f}")
    print(f"  DACC_NO_A (A masked)    = {dacc_noA:.4f}   (chance=0.5)")
    print(f"  DACC_NO_B (B masked)    = {dacc_noB:.4f}   (chance=0.5)")
    print(f"  OLD_STABLE max|dlogit|  = {old_maxdiff:.3e}  (after all D training)")
    print(f"  checkpoint saved -> {CKPT_OUT}")
    print("=" * 72)
    print(f"\nM3-Part2: DTASK_ACC={dtask:.4f} DACC_NO_A={dacc_noA:.4f} "
          f"DACC_NO_B={dacc_noB:.4f} OLD_MAXDIFF={old_maxdiff:.3e} VERDICT={verdict}")


if __name__ == "__main__":
    main()
