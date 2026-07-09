"""
model_sig_var.py -- VARIABLE-ARITY two-zone board (SIG-Part3).

Board (frozen orthogonal keys, fixed ordinal placement):
  slot 0        : name value
  slots 1..4    : param VALUE slots  (param i -> slot 1+i)
  slots 5..8    : param VALIDITY slots (param i -> slot 5+i; present/empty bit)
  slots 9,10    : Zone B's query / probe
Only the NUMBER of occupied param slots varies per sample (up to cap 4);
placement is the proven CLEAN_GENERAL "i -> slot i".  Empty value slots get a
ZERO payload; their validity slot carries a learned EMPTY vector so a blind
consumer can tell present from empty (a value-only reader would be fooled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sig_task_var import VOCAB_SIZE, SEQ_A, SEQ_B, N_PARAM, PAD_ID
from board import D_ADDR, D_VAL, make_board
from model_m1_frozen import make_frozen_keys

D_LOCAL = 128

# board slot layout
NAME_SLOT   = 0
VAL_SLOTS   = list(range(1, 1 + N_PARAM))              # 1..4
VALID_SLOTS = list(range(1 + N_PARAM, 1 + 2 * N_PARAM))  # 5..8
A_SLOTS     = 1 + 2 * N_PARAM                          # 9
B_SLOTS     = 2                                        # query, probe
TOTAL_SLOTS = A_SLOTS + B_SLOTS                        # 11


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


class ZoneSigHolder(nn.Module):
    """Zone A: holds the variable-arity signature; writes value + validity slots."""
    def __init__(self, slot_keys):
        super().__init__()
        self.trunk = _trunk(SEQ_A)
        self.name_proj  = nn.Linear(D_LOCAL, D_VAL)
        self.val_projs  = nn.ModuleList([nn.Linear(D_LOCAL, D_VAL) for _ in range(N_PARAM)])
        self.validity_emb = nn.Embedding(2, D_VAL)     # 0=empty, 1=present (learned)
        self.register_buffer("slot_keys", slot_keys)   # (A_SLOTS, D_ADDR) frozen

    def encode(self, toks):
        return _encode(self.trunk, toks)

    def write(self, h, toks):
        B = h.size(0)
        present = (toks[:, 1:1 + N_PARAM] != PAD_ID)    # (B, N_PARAM) bool
        zero = h.new_zeros(B, D_VAL)
        slots = [self.name_proj(h[:, 0, :])]           # slot 0 = name
        for i in range(N_PARAM):                        # value slots
            pay = self.val_projs[i](h[:, 1 + i, :])
            slots.append(torch.where(present[:, i:i + 1], pay, zero))
        for i in range(N_PARAM):                        # validity slots
            slots.append(self.validity_emb(present[:, i].long()))
        vals = torch.stack(slots, dim=1)               # (B, A_SLOTS, D_VAL)
        keys = self.slot_keys.unsqueeze(0).expand(B, -1, -1)
        return keys, vals


class ZoneQuerier(nn.Module):
    """Zone B: holds the query; writes query/probe, reads A's slots, emits label."""
    def __init__(self, slot_keys):
        super().__init__()
        self.trunk = _trunk(SEQ_B)
        self.write_projs = nn.ModuleList([nn.Linear(D_LOCAL, D_VAL) for _ in range(B_SLOTS)])
        self.register_buffer("slot_keys", slot_keys)   # (B_SLOTS, D_ADDR) frozen
        self.read_q_proj   = nn.Linear(D_LOCAL, D_ADDR)
        self.read_out_proj = nn.Linear(D_VAL, D_LOCAL)
        self.alpha         = nn.Parameter(torch.tensor([0.1]))
        nn.init.zeros_(self.read_out_proj.weight)
        nn.init.zeros_(self.read_out_proj.bias)

    def encode(self, toks):
        return _encode(self.trunk, toks)

    def write(self, h):
        B = h.size(0)
        slots = [self.write_projs[i](h[:, i, :]) for i in range(B_SLOTS)]
        vals = torch.stack(slots, dim=1)
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


class PhoenixSigVar(nn.Module):
    def __init__(self, key_seed: int = 20260710):
        super().__init__()
        all_keys = make_frozen_keys(TOTAL_SLOTS, D_ADDR, key_seed)
        self.zone_a = ZoneSigHolder(all_keys[:A_SLOTS].clone())
        self.zone_b = ZoneQuerier(all_keys[A_SLOTS:].clone())
        self.head_b = nn.Sequential(
            nn.Linear(D_LOCAL, 256), nn.GELU(), nn.Linear(256, 2),   # binary label
        )

    def forward(self, toks_a, toks_b):
        h_a = self.zone_a.encode(toks_a)
        h_b = self.zone_b.encode(toks_b)
        keys_a, vals_a = self.zone_a.write(h_a, toks_a)
        keys_b, vals_b = self.zone_b.write(h_b)
        board_keys, board_vals = make_board([keys_a, keys_b], [vals_a, vals_b])

        device = toks_a.device
        mask_b = torch.zeros(TOTAL_SLOTS, dtype=torch.bool, device=device)
        mask_b[:A_SLOTS] = True                         # B reads A's slots (cross-stream)
        h_b2 = self.zone_b.read(h_b, board_keys, board_vals, mask_b)

        self.h_b2 = h_b2
        self.board_vals = board_vals
        return self.head_b(h_b2.mean(1))
