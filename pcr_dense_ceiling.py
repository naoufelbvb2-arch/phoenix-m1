"""
pcr_dense_ceiling.py -- dense ceiling for the PCR task.

One small transformer (<=3 layers, d_model=128) that sees BOTH sides and must
predict the repair r = (T - x) mod M -- i.e. it must EXECUTE (s0,ops)->x and
REPAIR against T.  Proves the task is solvable.  Target >= 95%.

Also exposes PCRModel + train_eval(mode) reused by pcr_controls.py so the exact
same architecture is used for the ceiling and the two half-info controls.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pcr_task import generate_batch, encode, VOCAB_SIZE, SEQ_LEN, M

D_MODEL  = 128
N_LAYERS = 2            # <= 3 as required
N_HEAD   = 4
N_TRAIN  = 50_000       # ~3x cover of the L=2 input space (16,384) -> learns fast
N_TEST   = 4_096
N_EPOCHS = 8            # hits ~100% by ep8 (verified); kept low for the CPU budget
BATCH    = 256
LR       = 1e-3
WD       = 0.0


class PCRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, nhead=N_HEAD, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        # Flatten readout: gives the MLP the full per-position representation so
        # it can memorize/interpolate the (covered) input->repair mapping (M0
        # lesson: don't dilute with mean-pool).  MLP supplies the nonlinearity.
        self.head = nn.Sequential(
            nn.Linear(SEQ_LEN * D_MODEL, 256), nn.GELU(), nn.Linear(256, M),
        )

    def forward(self, toks):
        pos = torch.arange(toks.size(1), device=toks.device)
        h = self.enc(self.tok(toks) + self.pos(pos))
        return self.head(h.reshape(h.size(0), -1))    # flatten -> repair (M classes)


def train_eval(mode: str, seed: int = 0, n_epochs: int = N_EPOCHS, verbose: bool = False):
    """Train a fresh PCRModel on `mode` inputs; return held-out accuracy on r."""
    torch.manual_seed(seed)

    s0, ops, T, _, y = generate_batch(N_TRAIN, seed=1)
    toks = encode(s0, ops, T, mode)
    loader = DataLoader(
        TensorDataset(toks, y), batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    s0v, opsv, Tv, _, yv = generate_batch(N_TEST, seed=999)   # held-out
    toksv = encode(s0v, opsv, Tv, mode)

    model = PCRModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, n_epochs + 1):
        model.train()
        for tb, yb in loader:
            opt.zero_grad()
            loss = ce(model(tb), yb)
            loss.backward()
            opt.step()
        if verbose and (ep % 2 == 0 or ep == n_epochs):
            model.eval()
            with torch.no_grad():
                acc = (model(toksv).argmax(1) == yv).float().mean().item()
                tracc = (model(toks[:4096]).argmax(1) == y[:4096]).float().mean().item()
            print(f"    [{mode}] ep{ep:>2} train={tracc:.4f} test={acc:.4f}")

    model.eval()
    with torch.no_grad():
        acc = (model(toksv).argmax(1) == yv).float().mean().item()
    return acc


if __name__ == "__main__":
    t0 = time.time()
    acc = train_eval("full", verbose=True)
    print(f"\nPCR dense ceiling (full): CEILING={acc:.4f}  chance={1.0/M:.4f}  "
          f"({time.time() - t0:.1f}s)")
