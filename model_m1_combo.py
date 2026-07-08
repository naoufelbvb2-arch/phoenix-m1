"""
model_m1_combo.py -- FROZEN addresses + GENERALIZED reader-side sub-masking.

Combines the two half-fixes:
  * frozen addresses (from model_m1_frozen)  -> kills ADDRESSING_DRIFT, clean v0
  * reader-side sub-masking (from Part-3)     -> forces distinct values into
                                                 distinct addressable slots

CRITICAL -- the sub-masking is written GENERAL in the number of routed
attributes n, NOT hardcoded for 2.  `slot_subset_masks` partitions a writer's K
slots into n disjoint contiguous subsets by the formula
    attribute i  ->  slots [(i*K)//n : ((i+1)*K)//n]
for any 1 <= n <= K.  The 2-attribute case (halves) is a special case of the
formula, not the implementation.  A reader that routes n attributes issues n
sub-masked reads (one per attribute, at that attribute's key-marker); the
per-attribute aux gradient then forces the writer to place value i in subset i.
Nothing is wired per-attribute by hand -- both the mask set and the read loop
are generated from n.  This is what lets the mechanism scale to the many-value
programming domain rather than being a 2-attribute trick.
"""

import torch
import torch.nn as nn

from task_m1 import key_marker, VOCAB_SIZE
from board import K, D_ADDR, make_board
from model_m1_frozen import ZoneFrozen, _extract, make_frozen_keys, D_LOCAL


def slot_subset_masks(total_slots: int, base: int, k: int, n_attr: int, device):
    """
    General reader-side sub-masking.  Partition the writer's k contiguous slots
    [base : base+k] into n_attr disjoint contiguous subsets; attribute i gets
    slots [(i*k)//n_attr : ((i+1)*k)//n_attr].  Returns a list of n_attr boolean
    masks over the full board (True = visible to that attribute's read).

    Pure function of n_attr -- no per-attribute constants.  n_attr=1 -> whole
    block (dense); n_attr=2 -> halves; n_attr=k -> one slot each.  Requires
    n_attr <= k (>=1 slot per attribute).
    """
    assert 1 <= n_attr <= k, f"need 1 <= n_attr({n_attr}) <= k({k})"
    masks = []
    for i in range(n_attr):
        lo = base + (i * k) // n_attr
        hi = base + ((i + 1) * k) // n_attr
        m = torch.zeros(total_slots, dtype=torch.bool, device=device)
        m[lo:hi] = True
        masks.append(m)
    return masks


class PhoenixM1Combo(nn.Module):
    """
    a_markers : key-markers of the attributes Zone A routes FROM Zone B's slots.
    b_markers : key-markers of the attributes Zone B routes FROM Zone A's slots.
    For task_m1: Zone A routes v0,v1 (markers 0,1); Zone B routes v2 (marker 2).
    The design is general -- pass longer marker tuples for more attributes.
    """
    def __init__(self, a_markers=(0, 1), b_markers=(2,), key_seed: int = 20260706):
        super().__init__()
        d = D_LOCAL
        all_keys = make_frozen_keys(2 * K, D_ADDR, key_seed)
        self.zone_a = ZoneFrozen("A", all_keys[:K].clone())
        self.zone_b = ZoneFrozen("B", all_keys[K:].clone())

        self.a_markers = tuple(a_markers)
        self.b_markers = tuple(b_markers)
        self.n_a = len(self.a_markers)
        self.n_b = len(self.b_markers)

        self.head_a = nn.Sequential(
            nn.Linear(self.n_a * d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )
        self.head_b = nn.Sequential(
            nn.Linear(self.n_b * d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )

    def forward(self, x_a, x_b):
        h_a = self.zone_a.encode(x_a)
        h_b = self.zone_b.encode(x_b)
        keys_a, vals_a = self.zone_a.write(h_a)
        keys_b, vals_b = self.zone_b.write(h_b)
        board_keys, board_vals = make_board([keys_a, keys_b], [vals_a, vals_b])

        total = 2 * K
        device = x_a.device
        # Zone A reads Zone B's block [K:2K], partitioned into n_a subsets.
        masks_a = slot_subset_masks(total, K, K, self.n_a, device)
        # Zone B reads Zone A's block [0:K], partitioned into n_b subsets.
        masks_b = slot_subset_masks(total, 0, K, self.n_b, device)

        # one sub-masked read per routed attribute, extracted at its marker
        self.r_a = []
        for i, mk in enumerate(self.a_markers):
            h_a2_i = self.zone_a.read(h_a, board_keys, board_vals, masks_a[i])
            self.r_a.append(_extract(h_a2_i, x_a, key_marker(mk)))
        self.r_b = []
        for i, mk in enumerate(self.b_markers):
            h_b2_i = self.zone_b.read(h_b, board_keys, board_vals, masks_b[i])
            self.r_b.append(_extract(h_b2_i, x_b, key_marker(mk)))

        rep_a = torch.cat(self.r_a, dim=-1)
        rep_b = torch.cat(self.r_b, dim=-1)
        return self.head_a(rep_a), self.head_b(rep_b), vals_a, vals_b
