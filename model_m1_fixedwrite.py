"""
model_m1_fixedwrite.py -- FROZEN addresses + FIXED WRITE-ASSIGNMENT.

Root cause the combo exposed: the write placed values INPUT-DEPENDENTLY (learned
write-queries decided WHERE each value went, per sample), so a fixed-query blind
consumer could not decode them -- even though native routing (av~1.0) and payload
orthogonality (VALUE_SIM~0.02) looked perfect.

Fix under test -- freeze the value->slot assignment on the WRITE side:
  * Each routable value is written to ONE pre-assigned, fixed slot, independent
    of input.  Value i (of a zone's write list) -> that zone's slot i.
  * The payload is produced by a learned projection of the hidden state AT that
    value's marker position (marker-shift on hidden states -- the approved
    mechanism, not raw token IDs).  So the writer learns WHAT to place; WHERE is
    fixed.  No input-dependent query decides placement.
  * Unassigned slots carry a zero payload.
  * Keys remain FROZEN orthogonal buffers (addresses fixed at birth).

Because value i's slot is fixed and its payload is a deterministic function of v_i,
that slot is the SAME function of v_i for every input -> a blind fixed-query
consumer can decode it.  This is the concentration the combo lacked.

Generalized in the number of values n (fixed_slot_assignment): value i -> slot i,
disjoint, for any n <= K.  The 3-value task is a special case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from task_m1 import TOTAL_VOCAB, SEQ_LEN, VOCAB_SIZE, key_marker, value_marker
from board import K, D_ADDR, D_VAL, make_board
from model_m1_frozen import _extract, make_frozen_keys, D_LOCAL


def fixed_slot_assignment(n_values: int, k: int):
    """value i -> zone-local slot i (disjoint, input-independent). Requires n<=k."""
    assert 1 <= n_values <= k, f"need 1 <= n_values({n_values}) <= k({k})"
    return list(range(n_values))


class ZoneFixedWrite(nn.Module):
    def __init__(self, name: str, slot_keys: torch.Tensor, write_markers):
        super().__init__()
        self.name = name
        d = D_LOCAL
        self.write_markers = list(write_markers)          # value-marker ids to write
        self.n_w = len(self.write_markers)
        self.slots = fixed_slot_assignment(self.n_w, K)   # value i -> slot i

        self.tok_emb = nn.Embedding(TOTAL_VOCAB, d)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        enc_layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # WHAT to write: one payload projection per assigned value (WHERE is fixed).
        self.write_projs = nn.ModuleList([nn.Linear(d, D_VAL) for _ in range(self.n_w)])

        # FROZEN addresses
        self.register_buffer("slot_keys", slot_keys)      # (K, D_ADDR) non-trainable

        # Read (dense, Safe-Zero) -- only used to give the scaffold a training path
        self.read_q_proj   = nn.Linear(d, D_ADDR)
        self.read_out_proj = nn.Linear(D_VAL, d)
        self.alpha         = nn.Parameter(torch.tensor([0.1]))
        nn.init.zeros_(self.read_out_proj.weight)
        nn.init.zeros_(self.read_out_proj.bias)

    def encode(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        return self.encoder(self.tok_emb(x) + self.pos_emb(pos))

    def write(self, h, x):
        """FIXED assignment: value i's payload -> slot i; other slots = 0."""
        B = h.size(0)
        zero = h.new_zeros(B, D_VAL)
        payloads = [zero] * K
        for i, m in enumerate(self.write_markers):
            e = _extract(h, x, m)                 # (B, D_LOCAL) hidden at value_marker m
            payloads[self.slots[i]] = self.write_projs[i](e)   # WHAT; slot is fixed
        vals = torch.stack(payloads, dim=1)       # (B, K, D_VAL)
        keys = self.slot_keys.unsqueeze(0).expand(B, -1, -1)   # frozen addresses
        return keys, vals

    def read(self, h, board_keys, board_vals, slot_mask=None):
        Q      = self.read_q_proj(h)
        scores = torch.bmm(Q, board_keys.transpose(1, 2)) / (D_ADDR ** 0.5)
        if slot_mask is not None:
            invisible = ~slot_mask.to(dtype=torch.bool, device=h.device)
            scores = scores.masked_fill(invisible.view(1, 1, -1), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, board_vals)
        return h + self.alpha * self.read_out_proj(context)


class PhoenixM1FixedWrite(nn.Module):
    def __init__(self, key_seed: int = 20260707):
        super().__init__()
        d = D_LOCAL
        all_keys = make_frozen_keys(2 * K, D_ADDR, key_seed)
        # Zone A holds v2 (stream_A value); Zone B holds v0, v1 (stream_B values).
        self.zone_a = ZoneFixedWrite("A", all_keys[:K].clone(), write_markers=[value_marker(2)])
        self.zone_b = ZoneFixedWrite("B", all_keys[K:].clone(),
                                     write_markers=[value_marker(0), value_marker(1)])

        self.head_a = nn.Sequential(
            nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )
        self.head_b = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )

    def forward(self, x_a, x_b):
        h_a = self.zone_a.encode(x_a)
        h_b = self.zone_b.encode(x_b)
        keys_a, vals_a = self.zone_a.write(h_a, x_a)
        keys_b, vals_b = self.zone_b.write(h_b, x_b)
        board_keys, board_vals = make_board([keys_a, keys_b], [vals_a, vals_b])

        # DENSE cross-stream read (reader is native; the board's blind-readability
        # now comes from the FIXED write, not from the read).
        device = x_a.device
        mask_for_a = torch.zeros(2 * K, dtype=torch.bool, device=device); mask_for_a[K:] = True
        mask_for_b = torch.zeros(2 * K, dtype=torch.bool, device=device); mask_for_b[:K] = True
        h_a2 = self.zone_a.read(h_a, board_keys, board_vals, mask_for_a)
        h_b2 = self.zone_b.read(h_b, board_keys, board_vals, mask_for_b)
        self.h_a2 = h_a2
        self.h_b2 = h_b2

        rep_a = torch.cat([
            _extract(h_a2, x_a, key_marker(0)),
            _extract(h_a2, x_a, key_marker(1)),
        ], dim=-1)
        rep_b = _extract(h_b2, x_b, key_marker(2))
        return self.head_a(rep_a), self.head_b(rep_b), vals_a, vals_b
