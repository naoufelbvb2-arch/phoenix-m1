"""
sig_dense_ceiling.py -- dense ceiling for the SIG selective-extraction task.

One small transformer (<=3 layers, d_model=128) that sees the FULL signature +
query + probe and predicts match / no-match.  Proves the selective-extraction
task is solvable.  Target >= 95% (chance = 50%).

Exposes SigModel + train_eval(mode) reused by sig_controls.py so the exact same
architecture is used for the ceiling and both controls.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sig_task import generate_batch, encode, VOCAB_SIZE, SEQ_LEN

D_MODEL  = 128
N_LAYERS = 2
N_HEAD   = 4
N_TRAIN  = 30_000       # coverage lesson: enough volume to learn select+compare
N_TEST   = 4_096
N_EPOCHS = 8            # ceiling hits 1.0 by ep3; kept low for the CPU budget
BATCH    = 256
LR       = 1e-3


class SigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, nhead=N_HEAD, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        # flatten readout -> binary head
        self.head = nn.Sequential(
            nn.Linear(SEQ_LEN * D_MODEL, 256), nn.GELU(), nn.Linear(256, 2),
        )

    def forward(self, toks):
        pos = torch.arange(toks.size(1), device=toks.device)
        h = self.enc(self.tok(toks) + self.pos(pos))
        return self.head(h.reshape(h.size(0), -1))


def _encode_mode(batch, mode):
    name, ret, param, query, probe, label = batch
    return encode(name, ret, param, query, probe, mode), label


def train_eval(mode: str, seed: int = 0, n_epochs: int = N_EPOCHS, verbose: bool = False):
    torch.manual_seed(seed)
    toks, y = _encode_mode(generate_batch(N_TRAIN, seed=1), mode)
    loader = DataLoader(
        TensorDataset(toks, y), batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    toksv, yv = _encode_mode(generate_batch(N_TEST, seed=999), mode)   # held-out

    model = SigModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, n_epochs + 1):
        model.train()
        for tb, yb in loader:
            opt.zero_grad()
            loss = ce(model(tb), yb)
            loss.backward()
            opt.step()
        if verbose and (ep % 3 == 0 or ep == n_epochs):
            model.eval()
            with torch.no_grad():
                acc = (model(toksv).argmax(1) == yv).float().mean().item()
            print(f"    [{mode}] ep{ep:>2} acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        acc = (model(toksv).argmax(1) == yv).float().mean().item()
    return acc


if __name__ == "__main__":
    t0 = time.time()
    acc = train_eval("full", verbose=True)
    print(f"\nSIG dense ceiling (full): CEILING={acc:.4f}  chance=0.5  ({time.time()-t0:.1f}s)")
