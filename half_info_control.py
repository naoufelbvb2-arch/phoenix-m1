"""
M0-Part1 -- Half-info control.

Identical architecture and training budget to dense_ceiling.py but the model
receives ONLY stream_A.  KEY_MARKER is present, so key_rep encodes k.
VALUE_MARKER is absent, so val_rep is always the zero vector -- the model
has no access to v.  Because v is uniform over [0, 64), the label (k+v)%64
is uniformly random from the model's perspective for any fixed k.

Acceptance criterion: <= (1/64 + 0.05) ~= 6.56% accuracy.
"""

import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from task import generate_batch, TOTAL_VOCAB, VOCAB_SIZE, SEQ_LEN, KEY_MARKER

warnings.filterwarnings("ignore", category=UserWarning)


# ---- model (same capacity as ceiling) ---------------------------------------

class DenseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 128
        self.tok  = nn.Embedding(TOTAL_VOCAB, d)
        self.pos  = nn.Embedding(SEQ_LEN, d)       # positions 0-15 only
        layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc  = nn.TransformerEncoder(layer, num_layers=2)
        # Head shape matches ceiling (2*d input) but val_rep is always zero
        self.head = nn.Sequential(
            nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )

    def forward(self, x):                           # x: (B, SEQ_LEN)
        T   = x.size(1)
        pos = torch.arange(T, device=x.device)
        h   = self.tok(x) + self.pos(pos)
        h   = self.enc(h)                           # (B, 16, d)

        # Extract k-token rep.  val_rep is the zero vector (no VALUE_MARKER).
        km     = (x == KEY_MARKER)
        k_mask = torch.zeros_like(km)
        k_mask[:, 1:] = km[:, :-1]

        key_rep = (h * k_mask.unsqueeze(-1).float()).sum(1)   # (B, d)
        val_rep = torch.zeros_like(key_rep)                   # (B, d) -- always zero

        return self.head(torch.cat([key_rep, val_rep], dim=-1))


# ---- training + eval --------------------------------------------------------

def run(n_train: int = 10_000, n_epochs: int = 3, batch_size: int = 128,
        lr: float = 1e-3, key_range: int = VOCAB_SIZE):
    torch.manual_seed(42)
    t0 = time.time()

    # Same training split as ceiling -- only stream_A is kept.
    A_tr, _, y_tr = generate_batch(n_train, key_range=key_range, seed=0)
    loader = DataLoader(
        TensorDataset(A_tr, y_tr), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    model   = DenseTransformer()
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
        A_te, _, y_te = generate_batch(4096, key_range=key_range, seed=9999)
        preds = model(A_te).argmax(1)
        acc   = (preds == y_te).float().mean().item()

    return acc, time.time() - t0


if __name__ == "__main__":
    acc, elapsed = run()
    threshold = 1.0 / VOCAB_SIZE + 0.05
    status    = "PASS" if acc <= threshold else "FAIL"
    print(f"[CONTROL] acc={acc:.4f}  target<={threshold:.4f}  time={elapsed:.1f}s  -> {status}")
