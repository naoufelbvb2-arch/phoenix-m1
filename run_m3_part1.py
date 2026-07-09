"""
M3-Part1 -- first real test of Append-Only Masking: append a new third zone (D)
onto the FROZEN SIG-Part3 board and verify BIT-EXACT zero forgetting of A/B.

  1. Baseline    : run frozen SIG-Part3 (A/B) on a fixed batch; record B's logits
                   and A's written slots EXACTLY.
  2. Append D    : Safe-Zero + append-only masking, NO training.  Re-run same
                   batch -> assert B's logits & A's slots are BIT-IDENTICAL.
  3. Train D     : short run on a trivial placeholder objective (D reconstructs an
                   existing field) so real gradients flow through D.  A/B frozen.
  4. Post-train  : re-run same batch -> assert STILL bit-identical.

Strict: any non-zero drift is a real failure.
  * bit-identical at BOTH init and post-train   -> STABLE
  * identical at init, drifts after training D   -> TRAIN_LEAK
  * not identical even at init                   -> APPEND_LEAK
"""

import warnings

import torch
import torch.nn as nn

from sig_task_var import generate_batch, encode_a, encode_b, NAME_V
from model_sig_var import PhoenixSigVar
from model_m3 import PhoenixM3, dummy_d_input, D_LOCAL, D_SEQ

warnings.filterwarnings("ignore", category=UserWarning)

CKPT_IN   = "sig_part3_ckpt.pt"
CKPT_OUT  = "m3_part1_ckpt.pt"
BATCH_N   = 4096
BATCH_SEED = 20260709      # the FIXED evaluation batch
D_EPOCHS  = 200
D_LR      = 1e-3


def make_batch(n, seed):
    name, params, present, query, probe, label = generate_batch(n, seed)
    toks_a = encode_a(name, params, present)
    toks_b = encode_b(query, probe)
    toks_d = dummy_d_input(n)
    return toks_a, toks_b, toks_d, name


def load_frozen():
    base = PhoenixSigVar()
    state = torch.load(CKPT_IN, map_location="cpu")
    base.load_state_dict(state)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base, state


def main():
    torch.manual_seed(0)
    base, state = load_frozen()

    # capture the OLD address-key buffers (must stay bit-identical after append)
    old_keys_a = state["zone_a.slot_keys"].clone()
    old_keys_b = state["zone_b.slot_keys"].clone()

    model = PhoenixM3(base)
    # Freeze everything except Zone D.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.zone_d.parameters():
        p.requires_grad_(True)
    model.eval()

    toks_a, toks_b, toks_d, name = make_batch(BATCH_N, BATCH_SEED)

    # ---- 1. baseline (frozen SIG-Part3 through A/B) -------------------------
    with torch.no_grad():
        b_logits_base, vals_a_base = model.baseline(toks_a, toks_b)
        # sanity: PhoenixM3.baseline must match the original loaded model exactly
        ref_logits = base(toks_a, toks_b)
    base_vs_ref = (b_logits_base - ref_logits).abs().max().item()

    # ---- 2. append D, untrained (Safe-Zero) --------------------------------
    with torch.no_grad():
        b_logits_init, vals_a_init, _ = model.appended(toks_a, toks_b, toks_d)
    init_maxdiff = (b_logits_init - b_logits_base).abs().max().item()
    init_slots_diff = (vals_a_init - vals_a_base).abs().max().item()

    # ---- 3. train ONLY Zone D on a placeholder objective -------------------
    # D reads the board and reconstructs the NAME field (an existing A field).
    d_head = nn.Linear(D_SEQ * D_LOCAL, NAME_V)
    d_params = list(model.zone_d.parameters()) + list(d_head.parameters())
    opt = torch.optim.Adam(d_params, lr=D_LR)
    ce = nn.CrossEntropyLoss()

    ta_tr, tb_tr, td_tr, name_tr = make_batch(8192, seed=12345)
    model.train()
    d_acc = 0.0
    for ep in range(D_EPOCHS):
        opt.zero_grad()
        _, _, h_d2 = model.appended(ta_tr, tb_tr, td_tr)
        pred = d_head(h_d2.reshape(h_d2.size(0), -1))
        loss = ce(pred, name_tr)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.zone_d.alpha.clamp_(min=0.10)
        if ep == 0:
            # confirm gradients really flow through D but NOT into A/B
            gd = model.zone_d.read_q_proj.weight.grad
            g_a = model.zone_a.name_proj.weight.grad
            g_b = model.zone_b.read_q_proj.weight.grad
            print(f"grad-flow check: zone_d.read_q grad_norm="
                  f"{(gd.norm().item() if gd is not None else 0):.4f}  "
                  f"zone_a.grad={g_a}  zone_b.grad={g_b}  (A/B must be None)")
    model.eval()
    with torch.no_grad():
        _, _, h_d2 = model.appended(ta_tr, tb_tr, td_tr)
        d_acc = (d_head(h_d2.reshape(h_d2.size(0), -1)).argmax(1) == name_tr).float().mean().item()

    # ---- 4. post-training check (same fixed batch) -------------------------
    with torch.no_grad():
        b_logits_post, vals_a_post, _ = model.appended(toks_a, toks_b, toks_d)
    posttrain_maxdiff = (b_logits_post - b_logits_base).abs().max().item()
    post_slots_diff = (vals_a_post - vals_a_base).abs().max().item()

    # ---- old address-key buffers unchanged? --------------------------------
    keys_a_diff = (model.zone_a.slot_keys - old_keys_a).abs().max().item()
    keys_b_diff = (model.zone_b.slot_keys - old_keys_b).abs().max().item()
    keys_unchanged = (keys_a_diff == 0.0 and keys_b_diff == 0.0)

    old_slots_unchanged = (init_slots_diff == 0.0 and post_slots_diff == 0.0 and keys_unchanged)

    # ---- verdict -----------------------------------------------------------
    if init_maxdiff != 0.0:
        verdict = "APPEND_LEAK"
    elif posttrain_maxdiff != 0.0:
        verdict = "TRAIN_LEAK"
    elif old_slots_unchanged:
        verdict = "STABLE"
    else:
        verdict = "TRAIN_LEAK"   # logits ok but a slot/key moved

    y = lambda b: "y" if b else "n"
    torch.save(model.state_dict(), CKPT_OUT)

    print("\n" + "=" * 72)
    print(f"  baseline vs original loaded model : max|dlogit| = {base_vs_ref:.3e}")
    print(f"  Zone D placeholder (reconstruct name) train acc = {d_acc:.4f}  "
          f"(alpha={model.zone_d.alpha.item():.3f})")
    print("  ----------------------------------------------------------------")
    print(f"  INIT      : max|dlogit_B| = {init_maxdiff:.3e}   A-slots dmax = {init_slots_diff:.3e}")
    print(f"  POST-TRAIN: max|dlogit_B| = {posttrain_maxdiff:.3e}   A-slots dmax = {post_slots_diff:.3e}")
    print(f"  old address keys unchanged: zone_a dmax={keys_a_diff:.3e}  "
          f"zone_b dmax={keys_b_diff:.3e}  -> {y(keys_unchanged)}")
    print(f"  checkpoint saved -> {CKPT_OUT}")
    print("=" * 72)
    print(f"\nM3-Part1: INIT_MAXDIFF={init_maxdiff:.3e} "
          f"POSTTRAIN_MAXDIFF={posttrain_maxdiff:.3e} "
          f"OLD_SLOTS_UNCHANGED={y(old_slots_unchanged)} VERDICT={verdict}")


if __name__ == "__main__":
    main()
