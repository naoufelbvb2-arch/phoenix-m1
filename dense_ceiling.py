"""
M0-Part1 -- Dense ceiling.

Two-layer transformer receives BOTH streams concatenated (32 tokens).

Design choices:
  - Marker shift: extract the transformer output at the k-token position
    (one slot right of KEY_MARKER) and at the v-token position (one slot
    right of VALUE_MARKER).  The k/v tokens carry their value directly in
    their initial embedding; after two attention layers they also encode
    context.  Gradient flows to exactly these two positions -- no dilution.
  - Head: concatenate key_rep and val_rep (256 input) for maximum flexibility;
    the 256->64 MLP can learn any function of the two representations.
  - Training: 20 000-sample dataset iterated in mini-batches; with 4096
    possible (k,v) pairs, E[unseen pairs] ~ 31, giving a 99%+ theoretical
    ceiling.  7 epochs converges the training loss to near zero (~0.001).

Acceptance criterion: >= 95% accuracy on held-out 4096-sample set.
"""

import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from task import generate_batch, TOTAL_VOCAB, VOCAB_SIZE, SEQ_LEN, KEY_MARKER, VALUE_MARKER

warnings.filterwarnings("ignore", category=UserWarning)


# ---- model ------------------------------------------------------------------

class DenseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 128
        self.tok  = nn.Embedding(TOTAL_VOCAB, d)
        self.pos  = nn.Embedding(SEQ_LEN * 2, d)   # positions 0-31
        self.seg  = nn.Embedding(2, d)              # 0=stream_A, 1=stream_B
        layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc  = nn.TransformerEncoder(layer, num_layers=2)
        # Concatenated key_rep + val_rep -> 2*d input
        self.head = nn.Sequential(
            nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )

    def forward(self, x):                           # x: (B, 2*SEQ_LEN)
        T   = x.size(1)
        dev = x.device
        pos = torch.arange(T, device=dev)
        seg = torch.cat([
            torch.zeros(SEQ_LEN, dtype=torch.long, device=dev),
            torch.ones( SEQ_LEN, dtype=torch.long, device=dev),
        ])
        h = self.tok(x) + self.pos(pos) + self.seg(seg)
        h = self.enc(h)                             # (B, 32, d)

        # Build masks for the k and v token positions (one slot after each marker).
        km = (x == KEY_MARKER  )                    # (B, 32) bool
        vm = (x == VALUE_MARKER)

        k_mask = torch.zeros_like(km)
        v_mask = torch.zeros_like(vm)
        k_mask[:, 1:] = km[:, :-1]                 # shift right: marker pos -> k pos
        v_mask[:, 1:] = vm[:, :-1]

        key_rep = (h * k_mask.unsqueeze(-1).float()).sum(1)   # (B, d)
        val_rep = (h * v_mask.unsqueeze(-1).float()).sum(1)   # (B, d)

        return self.head(torch.cat([key_rep, val_rep], dim=-1))


# ---- training + eval --------------------------------------------------------

def run(n_train: int = 20_000, n_epochs: int = 7, batch_size: int = 128,
        lr: float = 1e-3, key_range: int = VOCAB_SIZE):
    torch.manual_seed(42)
    t0 = time.time()

    A_tr, B_tr, y_tr = generate_batch(n_train, key_range=key_range, seed=0)
    X_tr = torch.cat([A_tr, B_tr], dim=1)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
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
        A_te, B_te, y_te = generate_batch(4096, key_range=key_range, seed=9999)
        X_te  = torch.cat([A_te, B_te], dim=1)
        preds = model(X_te).argmax(1)
        acc   = (preds == y_te).float().mean().item()

    return acc, time.time() - t0


if __name__ == "__main__":
    acc, elapsed = run()
    status = "PASS" if acc >= 0.95 else "FAIL"
    print(f"[CEILING] acc={acc:.4f}  target>=0.9500  time={elapsed:.1f}s  -> {status}")
