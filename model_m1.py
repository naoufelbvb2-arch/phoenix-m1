"""
PhoenixM1 -- M1-Part2 symmetric two-zone board model.

Reuses board.py primitives (make_board, K, D_ADDR, D_VAL).
ZoneM1 mirrors M0's Zone design (write + Safe-Zero read) but targets task_m1
constants and carries NO ignition crutches beyond the mandated Safe-Zero:
  - read_out_proj: zero-init (Safe-Zero, the only allowed zero-init)
  - alpha: 0.1 (small positive so out_proj receives gradient from step 1)
  - read_q_proj, write_val_proj: standard random init (no extra crutches)

Two-round protocol:
  Round 1  encode + write : both zones process their stream, push K slots.
  Round 2  read + head    : both zones cross-attend the whole board (alpha-
                            gated residual), then emit per-label logits.

Heads use marker-shift extraction on the zone's post-read hidden state:
  head_A : cat(_extract(h_A2, x_a, km0), _extract(h_A2, x_a, km1)) -> label_A
  head_B : _extract(h_B2, x_b, km2) -> label_B

Rationale: the key-token positions in stream_A (resp. stream_B) carry the
locally-observable k values after encoding; after the board read those same
positions also receive the cross-stream v context, giving head_A/B a 2d (resp.
d) representation that encodes both sides of each pair without mean-pool
dilution.

Aux heads (anneal run only -- weight=0 in no-scaffold run):
  aux_va  : vals_a.mean(1) -> predict v2   (what Zone B needs to read from A)
  aux_vb0 : vals_b.mean(1) -> predict v0   (what Zone A needs to read from B)
  aux_vb1 : vals_b.mean(1) -> predict v1   (what Zone A needs to read from B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from task_m1 import TOTAL_VOCAB, SEQ_LEN, VOCAB_SIZE, key_marker
from board import K, D_ADDR, D_VAL, make_board

D_LOCAL = 128


def _extract(h: torch.Tensor, x: torch.Tensor, marker_id: int) -> torch.Tensor:
    """Transformer hidden state at the position immediately after marker_id."""
    m = (x == marker_id)
    shifted = torch.zeros_like(m)
    shifted[:, 1:] = m[:, :-1]
    return (h * shifted.unsqueeze(-1).float()).sum(1)


class ZoneM1(nn.Module):
    def __init__(self, name: str = "zone"):
        super().__init__()
        self.name = name
        d = D_LOCAL

        self.tok_emb = nn.Embedding(TOTAL_VOCAB, d)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        enc_layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Write: K learned queries cross-attend zone hidden states
        self.write_queries  = nn.Parameter(torch.randn(K, d))
        self.write_attn     = nn.MultiheadAttention(d, num_heads=4, dropout=0.0,
                                                     batch_first=True)
        self.write_key_proj = nn.Linear(d, D_ADDR)
        self.write_val_proj = nn.Linear(d, D_VAL)   # standard init (no crutch)

        # Read: cross-attend full board, inject via Safe-Zero gated residual
        self.read_q_proj   = nn.Linear(d, D_ADDR)   # standard init (no crutch)
        self.read_out_proj = nn.Linear(D_VAL, d)    # Safe-Zero: zero-init only
        # Learnable so it can rise freely during bootstrap (attempt-11 showed
        # alpha spikes 0.3->0.5 at ep2, which the fragile cascade symmetry-break
        # needs).  A post-step floor clamp (in run loop) stops it decaying below
        # 0.35 — attempt-11's decay to ~0 is what killed acc_B and drifted keys.
        self.alpha         = nn.Parameter(torch.tensor([0.3]))

        nn.init.zeros_(self.read_out_proj.weight)
        nn.init.zeros_(self.read_out_proj.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return self.encoder(self.tok_emb(x) + self.pos_emb(pos))

    def write(self, h: torch.Tensor):
        B = h.size(0)
        Q = self.write_queries.unsqueeze(0).expand(B, -1, -1)
        # No detach: the encoder MUST receive the AUX_ROUTE gradient so it learns
        # to represent v0/v1/v2 clearly for the write to route (attempt-13 showed
        # detaching here starves routing -> av0 collapses post-anneal).
        summaries, _ = self.write_attn(Q, h, h)
        return self.write_key_proj(summaries), self.write_val_proj(summaries)

    def read(self, h: torch.Tensor, board_keys: torch.Tensor,
             board_vals: torch.Tensor, slot_mask=None) -> torch.Tensor:
        """
        Cross-attend full board, add alpha-gated context to h.
        slot_mask: optional (total_slots,) bool, True=visible  [Part-3 hook].
        """
        Q      = self.read_q_proj(h)
        scores = torch.bmm(Q, board_keys.transpose(1, 2)) / (D_ADDR ** 0.5)
        if slot_mask is not None:
            invisible = ~slot_mask.to(dtype=torch.bool, device=h.device)
            scores = scores.masked_fill(invisible.view(1, 1, -1), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, board_vals)
        return h + self.alpha * self.read_out_proj(context)


class PhoenixM1(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_LOCAL
        self.zone_a = ZoneM1("A")
        self.zone_b = ZoneM1("B")

        # head_A: concat of k0-pos and k1-pos extractions from h_A2 (2d input)
        self.head_a = nn.Sequential(
            nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )
        # head_B: k2-pos extraction from h_B2 (d input)
        self.head_b = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )

        # Aux heads for anneal run (unused when AUX_WEIGHT=0)
        self.aux_va  = nn.Linear(D_VAL, VOCAB_SIZE)
        self.aux_vb0 = nn.Linear(D_VAL, VOCAB_SIZE)
        self.aux_vb1 = nn.Linear(D_VAL, VOCAB_SIZE)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, slot_mask=None):
        """
        Returns (logits_a, logits_b, vals_a, vals_b).
        vals_a/vals_b are (B, K, D_VAL) slot value tensors exposed for aux loss.
        """
        # Round 1: encode + write
        h_a = self.zone_a.encode(x_a)
        h_b = self.zone_b.encode(x_b)
        keys_a, vals_a = self.zone_a.write(h_a)
        keys_b, vals_b = self.zone_b.write(h_b)
        board_keys, board_vals = make_board([keys_a, keys_b], [vals_a, vals_b])

        # Round 2: cross-stream read -- each zone sees ONLY the other zone's slots
        # (prevents self-referential gradient loops through the shared board).
        #
        # Zone A additionally uses SUB-MASKED reads to break the v0/v1 symmetry
        # deterministically: the k0-position read (r0) sees only Zone B's first
        # half of slots, the k1-position read (r1) sees only the second half.
        # v0 must therefore route to the first subset and v1 to the second --
        # they physically cannot compete for the same slots, eliminating the
        # random symmetry-breaking that starved one sub-problem (attempts 11/15/16).
        device = x_a.device
        H = K // 2
        mask_a_full = torch.zeros(2 * K, dtype=torch.bool, device=device)
        mask_a_full[K:] = True             # all Zone B slots (for model head_a)
        mask_a0 = torch.zeros(2 * K, dtype=torch.bool, device=device)
        mask_a0[K:K + H] = True            # Zone B first half  -> r0 / v0
        mask_a1 = torch.zeros(2 * K, dtype=torch.bool, device=device)
        mask_a1[K + H:] = True             # Zone B second half -> r1 / v1
        mask_for_b = torch.zeros(2 * K, dtype=torch.bool, device=device)
        mask_for_b[:K] = True              # Zone B sees only Zone A's slots

        h_a2   = self.zone_a.read(h_a, board_keys, board_vals, mask_a_full)
        h_a2_0 = self.zone_a.read(h_a, board_keys, board_vals, mask_a0)
        h_a2_1 = self.zone_a.read(h_a, board_keys, board_vals, mask_a1)
        h_b2   = self.zone_b.read(h_b, board_keys, board_vals, mask_for_b)
        self.h_a2   = h_a2     # full-mask read (model's own head_a)
        self.h_a2_0 = h_a2_0   # r0/v0 sub-masked read  (Part-3)
        self.h_a2_1 = h_a2_1   # r1/v1 sub-masked read  (Part-3)
        self.h_b2   = h_b2

        rep_a = torch.cat([
            _extract(h_a2, x_a, key_marker(0)),
            _extract(h_a2, x_a, key_marker(1)),
        ], dim=-1)
        rep_b = _extract(h_b2, x_b, key_marker(2))

        logits_a = self.head_a(rep_a)
        logits_b = self.head_b(rep_b)

        return logits_a, logits_b, vals_a, vals_b
