"""
model_sig.py -- SIG two-zone board: STRUCTURED (3-field) value on the board.

Built on the M1/PCR-validated recipe (frozen orthogonal addresses + FIXED
input-independent write-assignment + dense read + Safe-Zero).  What's new: the
crossed value is a 3-field signature, so Zone A writes ONE fixed slot PER FIELD.

Zone A (Signature-holder): trunk sees (name, ret, param) [query/probe masked];
  writes each field into its own fixed slot (name->slot0, ret->slot1,
  param->slot2), input-independent placement.
Zone B (Querier): trunk sees (query, probe) [signature masked]; writes its
  query/probe to its own fixed slots; does a query-conditioned DENSE read of A's
  field-slots (the query in B's input shapes the read query to attend the right
  field-slot -- selective extraction, NOT sub-masking); label head emits the
  binary match = (sig[query] == probe).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sig_task import VOCAB_SIZE, SEQ_LEN
from board import K, D_ADDR, D_VAL, make_board
from model_m1_frozen import make_frozen_keys

D_LOCAL = 128


class ZoneSig(nn.Module):
    def __init__(self, name: str, slot_keys: torch.Tensor, n_write: int):
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

        # one payload projection per written value; WHERE (slot) is fixed.
        self.write_projs = nn.ModuleList([nn.Linear(d, D_VAL) for _ in range(n_write)])
        self.register_buffer("slot_keys", slot_keys)      # (K, D_ADDR) frozen

        self.read_q_proj   = nn.Linear(d, D_ADDR)
        self.read_out_proj = nn.Linear(D_VAL, d)
        self.alpha         = nn.Parameter(torch.tensor([0.1]))
        nn.init.zeros_(self.read_out_proj.weight)
        nn.init.zeros_(self.read_out_proj.bias)

    def trunk_encode(self, toks):
        pos = torch.arange(toks.size(1), device=toks.device)
        return self.trunk(self.tok(toks) + self.pos(pos))

    def write(self, h, positions):
        """Fixed assignment: value at input position positions[i] -> slot i."""
        B = h.size(0)
        zero = h.new_zeros(B, D_VAL)
        payloads = [zero] * K
        for i, p in enumerate(positions):
            payloads[i] = self.write_projs[i](h[:, p, :])
        vals = torch.stack(payloads, dim=1)               # (B, K, D_VAL)
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


class PhoenixSig(nn.Module):
    # input positions of the 3 fields (name/ret/param) and of query/probe
    FIELD_POS = (0, 1, 2)
    QP_POS    = (3, 4)

    def __init__(self, key_seed: int = 20260709):
        super().__init__()
        all_keys = make_frozen_keys(2 * K, D_ADDR, key_seed)
        self.zone_a = ZoneSig("A", all_keys[:K].clone(), n_write=3)   # name, ret, param
        self.zone_b = ZoneSig("B", all_keys[K:].clone(), n_write=2)   # query, probe
        # label head on B only (binary match)
        self.head_b = nn.Sequential(
            nn.Linear(D_LOCAL, 256), nn.GELU(), nn.Linear(256, 2),
        )

    def forward(self, toks_a, toks_b):
        h_a = self.zone_a.trunk_encode(toks_a)
        h_b = self.zone_b.trunk_encode(toks_b)

        keys_a, vals_a = self.zone_a.write(h_a, self.FIELD_POS)   # 3 fields -> A slots 0,1,2
        keys_b, vals_b = self.zone_b.write(h_b, self.QP_POS)      # query,probe -> B slots 0,1
        board_keys, board_vals = make_board([keys_a, keys_b], [vals_a, vals_b])

        device = toks_a.device
        mask_b = torch.zeros(2 * K, dtype=torch.bool, device=device); mask_b[:K] = True  # B reads A
        h_b2 = self.zone_b.read(h_b, board_keys, board_vals, mask_b)

        self.h_b2 = h_b2
        self.board_vals = board_vals
        return self.head_b(h_b2.mean(1))          # binary match label
