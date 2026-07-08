"""
model_pcr.py -- PCR two-zone board, built on the M1-VALIDATED fixed-write recipe.

Symmetric Proposer/Checker zones over a passive board:
  * FROZEN addresses  : orthogonal slot-key buffers, requires_grad=False.
  * FIXED write-assign: each zone writes its ONE value to its fixed slot 0,
    input-independently (the M1 concentration fix -- what made it BUILDABLE).
  * Dense cross-stream read + Safe-Zero ignition (zero-init read_out + small gate).
  NO learned keys, NO reader-side sub-masking (both proven to fail in M1).

Zone A (Proposer): trunk sees (s0, ops) [T masked]; EXECUTES -> x internally;
                   writes a payload encoding x into A's fixed slot.
Zone B (Checker) : trunk sees (T) [s0,ops masked]; writes a payload encoding T
                   into B's fixed slot.
Round 2: A reads B's block (gets T) and, combining its internal x with the read
         T, emits the repair r=(T-x)%M -> corrected output = T.  B reads A's block
         (gets x).  Only A carries the label head.

Two values cross the board: x (A writes, B reads) and T (B writes, A reads),
each pinned to a fixed slot so a blind fixed-query consumer can address them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcr_task import VOCAB_SIZE, SEQ_LEN, M
from board import K, D_ADDR, D_VAL, make_board
from model_m1_frozen import make_frozen_keys

D_LOCAL = 128


class ZonePCR(nn.Module):
    def __init__(self, name: str, slot_keys: torch.Tensor):
        super().__init__()
        self.name = name
        d = D_LOCAL

        self.tok = nn.Embedding(VOCAB_SIZE, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        enc = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(enc, num_layers=2)

        # WHAT to write (value -> payload); WHERE is fixed (slot 0).
        self.write_proj = nn.Linear(d, D_VAL)
        # FROZEN addresses for this zone's K slots.
        self.register_buffer("slot_keys", slot_keys)     # (K, D_ADDR)

        # Dense read + Safe-Zero.
        self.read_q_proj   = nn.Linear(d, D_ADDR)
        self.read_out_proj = nn.Linear(D_VAL, d)
        self.alpha         = nn.Parameter(torch.tensor([0.1]))
        nn.init.zeros_(self.read_out_proj.weight)
        nn.init.zeros_(self.read_out_proj.bias)

    def trunk_encode(self, toks):
        pos = torch.arange(toks.size(1), device=toks.device)
        return self.trunk(self.tok(toks) + self.pos(pos))

    def write(self, h):
        """Fixed assignment: this zone's value (pooled trunk rep) -> slot 0."""
        B = h.size(0)
        payload = self.write_proj(h.mean(1))              # (B, D_VAL)
        zero = h.new_zeros(B, D_VAL)
        vals = torch.stack([payload] + [zero] * (K - 1), dim=1)   # (B, K, D_VAL)
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


class PhoenixPCR(nn.Module):
    def __init__(self, key_seed: int = 20260708):
        super().__init__()
        all_keys = make_frozen_keys(2 * K, D_ADDR, key_seed)
        self.zone_a = ZonePCR("A", all_keys[:K].clone())   # proposer: writes x
        self.zone_b = ZonePCR("B", all_keys[K:].clone())   # checker : writes T
        # label head sits on A only (only A can execute the repair)
        self.head_a = nn.Sequential(
            nn.Linear(D_LOCAL, 256), nn.GELU(), nn.Linear(256, M),
        )

    def forward(self, toks_a, toks_b):
        h_a = self.zone_a.trunk_encode(toks_a)      # proposer trunk (s0, ops)
        h_b = self.zone_b.trunk_encode(toks_b)      # checker  trunk (T)

        keys_a, vals_a = self.zone_a.write(h_a)     # A: x -> A slot 0
        keys_b, vals_b = self.zone_b.write(h_b)     # B: T -> B slot 0
        board_keys, board_vals = make_board([keys_a, keys_b], [vals_a, vals_b])

        device = toks_a.device
        mask_a = torch.zeros(2 * K, dtype=torch.bool, device=device); mask_a[K:] = True  # A reads B
        mask_b = torch.zeros(2 * K, dtype=torch.bool, device=device); mask_b[:K] = True  # B reads A
        h_a2 = self.zone_a.read(h_a, board_keys, board_vals, mask_a)
        h_b2 = self.zone_b.read(h_b, board_keys, board_vals, mask_b)

        self.h_a2 = h_a2      # A: internal x + read T  (task head + aux_T)
        self.h_b2 = h_b2      # B: read x               (aux_x)
        self.board_vals = board_vals

        # A emits the repair r=(T-x)%M ; corrected output (x+r)%M == T
        return self.head_a(h_a2.mean(1))
