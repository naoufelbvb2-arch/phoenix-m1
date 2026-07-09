"""
model_comp.py -- COMP two-zone board: test COMPOSABILITY of the frozen board.

Zone A (Signature-holder): sees (ret, param); writes each to its own fixed slot
  (frozen orthogonal key, ordinal placement).
Zone B (Relator): reads BOTH field-slots densely and emits the same-category
  (relational) label.  It has no query input -- its job is purely to combine the
  two fields off the board.

The relation label = (cat(ret) == cat(param)) is XNOR of the two categories,
NOT linearly separable, so producing it genuinely requires composing both fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from comp_task import VOCAB_SIZE, MASK_ID
from board import D_ADDR, D_VAL, make_board
from model_m1_frozen import make_frozen_keys

D_LOCAL = 128
A_SLOTS = 2      # ret -> slot 0, param -> slot 1
B_SLOTS = 2      # Zone B relator-context slots (distractors for Zone C)
TOTAL_SLOTS = A_SLOTS + B_SLOTS   # 4
SEQ_A = 2        # [ret, param]
SEQ_B = 2        # [MASK, MASK] -- two read positions, no query content


def _trunk(seq_len):
    return nn.ModuleDict(dict(
        tok=nn.Embedding(VOCAB_SIZE, D_LOCAL),
        pos=nn.Embedding(seq_len, D_LOCAL),
        enc=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_LOCAL, nhead=4, dim_feedforward=256,
                                       dropout=0.0, batch_first=True, norm_first=True),
            num_layers=2),
    ))


def _encode(trunk, toks):
    pos = torch.arange(toks.size(1), device=toks.device)
    return trunk["enc"](trunk["tok"](toks) + trunk["pos"](pos))


class ZoneCompHolder(nn.Module):
    """Zone A: writes ret -> slot 0, param -> slot 1 (fixed ordinal)."""
    def __init__(self, slot_keys):
        super().__init__()
        self.trunk = _trunk(SEQ_A)
        self.val_projs = nn.ModuleList([nn.Linear(D_LOCAL, D_VAL) for _ in range(A_SLOTS)])
        self.register_buffer("slot_keys", slot_keys)

    def encode(self, toks):
        return _encode(self.trunk, toks)

    def write(self, h):
        B = h.size(0)
        vals = torch.stack([self.val_projs[i](h[:, i, :]) for i in range(A_SLOTS)], dim=1)
        keys = self.slot_keys.unsqueeze(0).expand(B, -1, -1)
        return keys, vals


class ZoneCompRelator(nn.Module):
    """Zone B: reads A's field-slots and emits the relational label."""
    def __init__(self, slot_keys):
        super().__init__()
        self.trunk = _trunk(SEQ_B)
        self.write_projs = nn.ModuleList([nn.Linear(D_LOCAL, D_VAL) for _ in range(B_SLOTS)])
        self.register_buffer("slot_keys", slot_keys)
        self.read_q_proj   = nn.Linear(D_LOCAL, D_ADDR)
        self.read_out_proj = nn.Linear(D_VAL, D_LOCAL)
        self.alpha         = nn.Parameter(torch.tensor([0.1]))
        nn.init.zeros_(self.read_out_proj.weight)
        nn.init.zeros_(self.read_out_proj.bias)

    def encode(self, toks):
        return _encode(self.trunk, toks)

    def write(self, h):
        B = h.size(0)
        vals = torch.stack([self.write_projs[i](h[:, i, :]) for i in range(B_SLOTS)], dim=1)
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


class PhoenixComp(nn.Module):
    def __init__(self, key_seed: int = 20260711):
        super().__init__()
        all_keys = make_frozen_keys(TOTAL_SLOTS, D_ADDR, key_seed)
        self.zone_a = ZoneCompHolder(all_keys[:A_SLOTS].clone())
        self.zone_b = ZoneCompRelator(all_keys[A_SLOTS:].clone())
        self.head_b = nn.Sequential(
            nn.Linear(SEQ_B * D_LOCAL, 256), nn.GELU(), nn.Linear(256, 2),   # same-category
        )

    def forward(self, toks_a, toks_b):
        h_a = self.zone_a.encode(toks_a)
        h_b = self.zone_b.encode(toks_b)
        keys_a, vals_a = self.zone_a.write(h_a)
        keys_b, vals_b = self.zone_b.write(h_b)
        board_keys, board_vals = make_board([keys_a, keys_b], [vals_a, vals_b])

        device = toks_a.device
        mask_b = torch.zeros(TOTAL_SLOTS, dtype=torch.bool, device=device)
        mask_b[:A_SLOTS] = True                          # B reads A's field-slots
        h_b2 = self.zone_b.read(h_b, board_keys, board_vals, mask_b)

        self.h_b2 = h_b2
        self.board_vals = board_vals
        return self.head_b(h_b2.reshape(h_b2.size(0), -1))   # flatten both read positions
