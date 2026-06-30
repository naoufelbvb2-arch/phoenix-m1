"""
lesion.py -- Slot-mask helpers and masked-eval utility for M0-Part3.

Board slot layout (fixed by board.py):
  slots 0..K-1   = Zone A's owned slots
  slots K..2K-1  = Zone B's owned slots

slot_mask semantics (from zones.py Read layer):
  True  = slot is visible  (participates in softmax)
  False = slot is masked   (-inf before softmax, excluded from renormalisation)

Note: model.forward() passes the same slot_mask to both zones' readers.
Zone A's h_A2 is never used in the output (no head), so masking Zone A's
reader is a no-op for the final predictions.  Only Zone B's reader matters.
"""

import torch
from board import K

TOTAL_SLOTS = 2 * K   # 16


def mask_c1():
    """
    Cross-zone lesion: hide Zone A's slots (0..K-1).
    Zone B can only read its own slots, which lack key k.
    Expected: accuracy collapses to chance.
    """
    m = torch.ones(TOTAL_SLOTS, dtype=torch.bool)
    m[:K] = False
    return m


def mask_c2():
    """
    Negative control: hide Zone B's own slots (K..2K-1).
    Zone B still reads Zone A's slots (which carry key k).
    Expected: accuracy stays near baseline.
    """
    m = torch.ones(TOTAL_SLOTS, dtype=torch.bool)
    m[K:] = False
    return m


@torch.no_grad()
def eval_acc(model, loader, slot_mask=None):
    """
    Evaluate model accuracy with an optional slot mask at inference.

    model     : PhoenixM0P2 (or compatible)
    loader    : DataLoader yielding (xa, xb, y) batches
    slot_mask : (TOTAL_SLOTS,) bool tensor, or None (all visible)

    Returns float accuracy.
    """
    model.eval()
    correct = total = 0
    for xa, xb, y in loader:
        logits, _ = model(xa, xb, slot_mask=slot_mask)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    model.train()
    return correct / total
