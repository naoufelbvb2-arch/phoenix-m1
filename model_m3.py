"""
model_m3.py -- HORIZONTAL GROWTH: append a THIRD zone (D) onto the FROZEN
SIG-Part3 board (zones A + B) without touching the old zones.

Append-Only Masking (spec §4), the core promise being tested:
  * The OLD board (A's 9 slots + B's 2 slots = 11) is preserved bit-for-bit.
  * Zone D gets its OWN new slots, CONCATENATED AFTER the old ones, with its own
    frozen orthogonal address keys.  The OLD keys are LOADED (not regenerated --
    a different-sized QR would give different vectors); only NEW keys are added.
  * Old readers (here: Zone B) are HARD-MASKED from D's new slots.  Because the
    masked slots become exp(-inf)=0 in the softmax and contribute 0 to the
    weighted context (x + 0 == x exactly), B's read is BIT-IDENTICAL to the
    pre-append board -- the denominator never changes.  Zone A is write-only, so
    it is unaffected by construction.
  * Zone D may read EVERYTHING (old slots + its own).

Zone D is randomly initialised with Safe-Zero (zero read_out_proj), so at init
it also cannot perturb anything.  Nothing about A/B is trainable here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sig_task_var import VOCAB_SIZE, MASK_ID
from board import D_ADDR, D_VAL, make_board
from model_m1_frozen import make_frozen_keys
from model_sig_var import (
    PhoenixSigVar, ZoneSigHolder, ZoneQuerier,
    _trunk, _encode, D_LOCAL,
    A_SLOTS, B_SLOTS, TOTAL_SLOTS as BASE_SLOTS,   # 9, 2, 11
)

D_SEQ   = 2      # Zone D view: [MASK, MASK] -- it just reads the board
D_SLOTS = 2      # Zone D's own appended slots


class ZoneAppendD(nn.Module):
    """Zone D: appended reader with its own slots.  Reads the whole board."""
    def __init__(self, slot_keys):
        super().__init__()
        self.trunk = _trunk(D_SEQ)
        self.write_projs = nn.ModuleList([nn.Linear(D_LOCAL, D_VAL) for _ in range(D_SLOTS)])
        self.register_buffer("slot_keys", slot_keys)     # (D_SLOTS, D_ADDR) frozen
        self.read_q_proj   = nn.Linear(D_LOCAL, D_ADDR)
        self.read_out_proj = nn.Linear(D_VAL, D_LOCAL)
        self.alpha         = nn.Parameter(torch.tensor([0.1]))
        nn.init.zeros_(self.read_out_proj.weight)        # Safe-Zero ignition
        nn.init.zeros_(self.read_out_proj.bias)

    def encode(self, toks):
        return _encode(self.trunk, toks)

    def write(self, h):
        B = h.size(0)
        vals = torch.stack([self.write_projs[i](h[:, i, :]) for i in range(D_SLOTS)], dim=1)
        keys = self.slot_keys.unsqueeze(0).expand(B, -1, -1)
        return keys, vals

    def read(self, h, board_keys, board_vals, slot_mask):
        Q = self.read_q_proj(h)
        scores = torch.bmm(Q, board_keys.transpose(1, 2)) / (D_ADDR ** 0.5)
        invisible = ~slot_mask.to(dtype=torch.bool, device=h.device)
        scores = scores.masked_fill(invisible.view(1, 1, -1), float('-inf'))
        w = F.softmax(scores, dim=-1)
        ctx = torch.bmm(w, board_vals)
        return h + self.alpha * self.read_out_proj(ctx)


class PhoenixM3(nn.Module):
    """Frozen A/B (from SIG-Part3) + appended Zone D under append-only masking."""
    def __init__(self, base: PhoenixSigVar, d_key_seed: int = 20260712):
        super().__init__()
        # reuse the FROZEN modules -- same objects, same params, same key buffers
        self.zone_a = base.zone_a
        self.zone_b = base.zone_b
        self.head_b = base.head_b

        # NEW frozen keys for D, appended -- old keys are left exactly as loaded
        d_keys = make_frozen_keys(D_SLOTS, D_ADDR, d_key_seed)
        self.zone_d = ZoneAppendD(d_keys)

        self.total_slots = BASE_SLOTS + D_SLOTS          # 11 + 2 = 13

    # -- Zone A + B writes (identical code path in both forwards) --------------
    def _ab_writes(self, toks_a, toks_b):
        h_a = self.zone_a.encode(toks_a)
        h_b = self.zone_b.encode(toks_b)
        keys_a, vals_a = self.zone_a.write(h_a, toks_a)
        keys_b, vals_b = self.zone_b.write(h_b)
        return h_b, keys_a, vals_a, keys_b, vals_b

    def baseline(self, toks_a, toks_b):
        """EXACT SIG-Part3 behaviour: 11-slot board, no Zone D.  Reference truth."""
        h_b, keys_a, vals_a, keys_b, vals_b = self._ab_writes(toks_a, toks_b)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])   # BASE_SLOTS = 11
        mask_b = torch.zeros(BASE_SLOTS, dtype=torch.bool, device=toks_a.device)
        mask_b[:A_SLOTS] = True                                   # B reads A's block
        h_b2 = self.zone_b.read(h_b, bk, bv, mask_b)
        return self.head_b(h_b2.mean(1)), vals_a

    def appended(self, toks_a, toks_b, toks_d):
        """13-slot board with Zone D appended; B hard-masked from D's slots."""
        h_b, keys_a, vals_a, keys_b, vals_b = self._ab_writes(toks_a, toks_b)
        h_d = self.zone_d.encode(toks_d)
        keys_d, vals_d = self.zone_d.write(h_d)
        bk, bv = make_board([keys_a, keys_b, keys_d],
                            [vals_a, vals_b, vals_d])            # 13 slots

        device = toks_a.device
        # APPEND-ONLY MASK for the old reader B: still ONLY A's original block.
        # D's appended slots (and B's own) stay invisible -> B's read is unchanged.
        mask_b = torch.zeros(self.total_slots, dtype=torch.bool, device=device)
        mask_b[:A_SLOTS] = True
        h_b2 = self.zone_b.read(h_b, bk, bv, mask_b)
        b_logits = self.head_b(h_b2.mean(1))

        # Zone D reads EVERYTHING (old + its own).
        mask_d = torch.ones(self.total_slots, dtype=torch.bool, device=device)
        h_d2 = self.zone_d.read(h_d, bk, bv, mask_d)

        return b_logits, vals_a, h_d2


def dummy_d_input(n, device=None):
    return torch.full((n, D_SEQ), MASK_ID, dtype=torch.long, device=device)
