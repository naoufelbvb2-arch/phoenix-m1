"""
M1-Part1 -- Decomposability control (half-info ablations).

A-only: model sees ONLY stream_A, must predict label_A.
  stream_A carries KEY_MARKER(0..n_types-2) and VALUE_MARKER(n_types-1) only.
  VALUE_MARKER(0..n_types-2) never appear in stream_A, so those value
  representations are always the zero vector -- v_0..v_{n-2} are structurally
  inaccessible, and label_A = sum(k+v over group 1) is uniformly random given
  any fixed k0..k_{n-2}. Expect failure (near 1/vocab_size chance).

B-only: model sees ONLY stream_B, must predict label_B.
  stream_B carries KEY_MARKER(n_types-1) but never VALUE_MARKER(n_types-1)
  (that marker lives in stream_A) -- v_{n-1} is inaccessible, so
  label_B = k_{n-1} + v_{n-1} is uniformly random given a fixed k_{n-1}.
  Expect failure (near 1/vocab_size chance).

Same per-stream capacity as the ceiling model (3-layer transformer, d=128),
so any failure is attributable to missing information, not under-capacity.
"""

import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from task_m1 import (
    generate_batch, key_marker, value_marker,
    VOCAB_SIZE, SEQ_LEN, N_TYPES,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---- model (same per-stream capacity as ceiling) -----------------------------

class SingleStreamTransformerM1(nn.Module):
    """target='A' predicts label_A from group-1 markers (key_0..n-2, val_0..n-2).
       target='B' predicts label_B from group-2 markers (key_{n-1}, val_{n-1})."""

    def __init__(self, target, n_types=N_TYPES, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE):
        super().__init__()
        assert target in ("A", "B")
        self.target = target
        d = 128
        total_vocab = vocab_size + 2 * n_types

        self.tok = nn.Embedding(total_vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)

        # Mirror ceiling's pairwise-sum structure: one (key+val) pair per type.
        # Missing markers in a single stream return zero vectors, so the pair-sum
        # collapses to just the available rep -- structurally proving information
        # is absent rather than inferring it.
        if target == "A":
            self.type_pairs = [(key_marker(t), value_marker(t)) for t in range(n_types - 1)]
        else:
            self.type_pairs = [(key_marker(n_types - 1), value_marker(n_types - 1))]

        self.head = nn.Sequential(
            nn.Linear(len(self.type_pairs) * d, 256), nn.GELU(), nn.Linear(256, vocab_size),
        )

    def _extract(self, h, x, marker_id):
        m = (x == marker_id)
        shifted = torch.zeros_like(m)
        shifted[:, 1:] = m[:, :-1]
        return (h * shifted.unsqueeze(-1).float()).sum(1)

    def forward(self, x):                            # x: (B, seq_len)
        pos = torch.arange(x.size(1), device=x.device)
        h   = self.tok(x) + self.pos(pos)
        h   = self.enc(h)
        # Pairwise sum per type; missing markers yield zero vectors automatically.
        pair_reps = [self._extract(h, x, km) + self._extract(h, x, vm)
                     for km, vm in self.type_pairs]
        return self.head(torch.cat(pair_reps, dim=-1))


# ---- training + eval --------------------------------------------------------

def run(target: str, n_train: int = 8_000, n_epochs: int = 3,
        batch_size: int = 128, lr: float = 1e-3):
    torch.manual_seed(42)
    t0 = time.time()

    A_tr, B_tr, yA_tr, yB_tr = generate_batch(n_train, seed=0)
    X_tr = A_tr if target == "A" else B_tr
    y_tr = yA_tr if target == "A" else yB_tr
    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    model   = SingleStreamTransformerM1(target)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        A_te, B_te, yA_te, yB_te = generate_batch(4096, seed=9999)
        X_te = A_te if target == "A" else B_te
        y_te = yA_te if target == "A" else yB_te
        preds = model(X_te).argmax(1)
        acc   = (preds == y_te).float().mean().item()

    return acc, time.time() - t0


if __name__ == "__main__":
    threshold = 1.0 / VOCAB_SIZE + 0.05   # ~0.0656, matches M0 convention

    acc_A, t_A = run("A")
    acc_B, t_B = run("B")

    status_A = "PASS" if acc_A <= threshold else "FAIL"
    status_B = "PASS" if acc_B <= threshold else "FAIL"

    print(f"[CTRL-A] acc={acc_A:.4f}  target<={threshold:.4f}  time={t_A:.1f}s  -> {status_A}")
    print(f"[CTRL-B] acc={acc_B:.4f}  target<={threshold:.4f}  time={t_B:.1f}s  -> {status_B}")
