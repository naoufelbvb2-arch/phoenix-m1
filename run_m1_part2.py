"""
M1-Part2: board-channel ignition test.

Run 1 (no scaffold): loss = CE(label_A) + CE(label_B) only (AUX_WEIGHT=0).
  Prints gate magnitudes + per-label accuracy every 2 epochs.
  Verdict:
    PASS_NO_SCAFFOLD  -- both acc >= 80% AND both gates clearly grown from init
    PARTIAL           -- one/both acc in 10-80%
    CHANNEL_SILENCE   -- both acc near chance, gates flat

Run 2 (anneal, only if CHANNEL_SILENCE or PARTIAL):
  aux_weight anneals 2.0 -> 0 linearly over the first half of training, then 0.
  Aux targets supervise what each zone must WRITE for the other to READ:
    Zone A slots -> predict v2  (the value Zone B needs from A)
    Zone B slots -> predict v0  (value Zone A needs from B, type 0)
    Zone B slots -> predict v1  (value Zone A needs from B, type 1)
  Reports whether accuracy holds after the scaffold reaches 0.

Output:
  M1-Part2-NoScaffold: ACC_A=<> ACC_B=<> GATE_A=<> GATE_B=<> VERDICT=<>
  M1-Part2-Anneal:     ACC_A=<> ACC_B=<> SURVIVES_ANNEAL=<yes/no>  (if needed)
"""

import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from task_m1 import generate_batch, value_marker, VOCAB_SIZE, N_TYPES
from model_m1 import PhoenixM1

warnings.filterwarnings("ignore", category=UserWarning)

N_TRAIN       = 20_000
N_EPOCHS      = 8
BATCH_SIZE    = 128
LR            = 1e-3
AUX_MAX       = 2.0
INIT_ALPHA    = 0.1     # must match ZoneM1.__init__
GATE_GROWN_TH = 0.5     # gate must exceed this to count as "clearly grown"
PASS_TH       = 0.80
CHANCE_TH     = 0.10    # roughly 6x random chance (1/64 ~ 0.016)


def _tok_after(x: torch.Tensor, marker_id: int) -> torch.Tensor:
    """Raw token ID after marker_id -- for ground-truth aux targets only."""
    is_m = (x == marker_id)
    shifted = torch.zeros_like(is_m)
    shifted[:, 1:] = is_m[:, :-1]
    return (x * shifted.long()).sum(dim=1)


def anneal_w(epoch: int, n_epochs: int) -> float:
    """Linear 2.0 -> 0 over first half, then 0."""
    half = n_epochs // 2
    if epoch > half:
        return 0.0
    return AUX_MAX * (1.0 - (epoch - 1) / half)


def train(aux_schedule=None, seed: int = 42):
    """
    Train PhoenixM1.
    aux_schedule: callable(epoch, n_epochs) -> float, or None (no aux loss).
    Returns (acc_A, acc_B, gate_A, gate_B).
    """
    torch.manual_seed(seed)

    xa_tr, xb_tr, ya_tr, yb_tr = generate_batch(N_TRAIN, seed=0)
    loader = DataLoader(
        TensorDataset(xa_tr, xb_tr, ya_tr, yb_tr),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xa_te, xb_te, ya_te, yb_te = generate_batch(4096, seed=9999)

    model   = PhoenixM1()
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    ce      = nn.CrossEntropyLoss()

    print(f"  {'ep':>3}  {'acc_A':>7}  {'acc_B':>7}  "
          f"{'gate_A':>8}  {'gate_B':>8}  {'aux_w':>6}")

    def eval_acc():
        model.eval()
        with torch.no_grad():
            la, lb, _, _ = model(xa_te, xb_te)
            a = (la.argmax(1) == ya_te).float().mean().item()
            b = (lb.argmax(1) == yb_te).float().mean().item()
        model.train()
        return a, b

    for epoch in range(1, N_EPOCHS + 1):
        w = 0.0 if aux_schedule is None else aux_schedule(epoch, N_EPOCHS)
        model.train()
        for xa, xb, ya, yb in loader:
            opt.zero_grad()
            logits_a, logits_b, vals_a, vals_b = model(xa, xb)
            loss = ce(logits_a, ya) + ce(logits_b, yb)

            if w > 0.0:
                # Aux targets: the cross-stream values each zone must write
                # for the other to read. _tok_after used only for labels.
                v2 = _tok_after(xa, value_marker(2))   # v2 in stream_A -> Zone B needs it
                v0 = _tok_after(xb, value_marker(0))   # v0 in stream_B -> Zone A needs it
                v1 = _tok_after(xb, value_marker(1))   # v1 in stream_B -> Zone A needs it
                ma = vals_a.mean(1)   # (B, D_VAL)
                mb = vals_b.mean(1)
                aux = (ce(model.aux_va(ma),  v2)
                     + ce(model.aux_vb0(mb), v0)
                     + ce(model.aux_vb1(mb), v1))
                loss = loss + w * aux

            loss.backward()
            opt.step()

        if epoch % 2 == 0 or epoch == N_EPOCHS:
            a, b  = eval_acc()
            ga    = model.zone_a.alpha.item()
            gb    = model.zone_b.alpha.item()
            print(f"  {epoch:>3}  {a:>7.4f}  {b:>7.4f}  "
                  f"{ga:>8.4f}  {gb:>8.4f}  {w:>6.2f}")

    acc_a, acc_b = eval_acc()
    return acc_a, acc_b, model.zone_a.alpha.item(), model.zone_b.alpha.item()


def compute_verdict(acc_a, acc_b, gate_a, gate_b):
    gates_grown = gate_a > GATE_GROWN_TH and gate_b > GATE_GROWN_TH
    if acc_a >= PASS_TH and acc_b >= PASS_TH and gates_grown:
        return "PASS_NO_SCAFFOLD"
    if acc_a < CHANCE_TH and acc_b < CHANCE_TH and not gates_grown:
        return "CHANNEL_SILENCE"
    lags = []
    if acc_a < PASS_TH:
        lags.append(f"A={acc_a:.3f}")
    if acc_b < PASS_TH:
        lags.append(f"B={acc_b:.3f}")
    return "PARTIAL(lags:" + ",".join(lags) + ")" if lags else "PARTIAL"


if __name__ == "__main__":
    # ---- Run 1: no scaffold -------------------------------------------------
    print("=" * 65)
    print("Run 1 -- No scaffold  (AUX_WEIGHT=0 throughout)")
    print("=" * 65)
    t0 = time.time()
    acc_a, acc_b, ga, gb = train(aux_schedule=None, seed=42)
    t1 = time.time()

    v = compute_verdict(acc_a, acc_b, ga, gb)
    print(f"\nM1-Part2-NoScaffold: ACC_A={acc_a:.4f} ACC_B={acc_b:.4f} "
          f"GATE_A={ga:.4f} GATE_B={gb:.4f} VERDICT={v}  ({t1-t0:.1f}s)")

    # ---- Run 2: anneal (only if channel didn't self-ignite) -----------------
    need_anneal = v == "CHANNEL_SILENCE" or v.startswith("PARTIAL")
    if need_anneal:
        print()
        print("=" * 65)
        print("Run 2 -- Anneal  (aux 2.0->0 over first half, then 0)")
        print("=" * 65)
        t2 = time.time()
        acc_a2, acc_b2, ga2, gb2 = train(aux_schedule=anneal_w, seed=43)
        t3 = time.time()
        survives = "yes" if (acc_a2 >= PASS_TH and acc_b2 >= PASS_TH) else "no"
        print(f"\nM1-Part2-Anneal: ACC_A={acc_a2:.4f} ACC_B={acc_b2:.4f} "
              f"GATE_A={ga2:.4f} GATE_B={gb2:.4f} "
              f"SURVIVES_ANNEAL={survives}  ({t3-t2:.1f}s)")
