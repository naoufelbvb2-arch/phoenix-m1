"""
Zone module -- M0-Part2.

Each zone embeds and processes its own stream with 2 transformer blocks,
then writes K content-aware (key, val) slots and reads from the full board.

Write  (spec 2.3): K learned queries cross-attend the zone's hidden states
                   -> K summaries -> projected to (D_ADDR, D_VAL) slot pairs.

Read   (spec 2.4): zone hidden state cross-attends the whole board
                   (queries from h, keys = slot keys, values = slot values),
                   output projected back to D_LOCAL and injected via a
                   Safe-Zero scalar gate.

Safe-Zero design:
  out_proj is ZERO-INITIALIZED, so output = alpha * 0 = 0 at init exactly.
  alpha starts at 0.1 (not 0) so that out_proj.weight receives a non-zero
  gradient (alpha * dL/dh * context) from step 1, bootstrapping the channel.
  If alpha=0 at init AND out_proj=0, both gradients are exactly zero --
  Zone A's write never trains because the board read path is fully dead.

slot_mask hook: Read accepts an optional (total_slots,) bool mask where
                True = visible.  Default is all visible.  Used by Part 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from task  import TOTAL_VOCAB, SEQ_LEN
from board import K, D_ADDR, D_VAL

D_LOCAL = 128   # hidden dimension per zone


class Zone(nn.Module):
    def __init__(self, name="zone"):
        super().__init__()
        self.name = name
        d = D_LOCAL

        # ---- encoder --------------------------------------------------------
        self.tok_emb = nn.Embedding(TOTAL_VOCAB, d)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        enc_layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # ---- write ----------------------------------------------------------
        # K learned query vectors; each cross-attends zone hidden states.
        self.write_queries  = nn.Parameter(torch.randn(K, d) * 0.02)
        self.write_attn     = nn.MultiheadAttention(d, num_heads=4, dropout=0.0,
                                                    batch_first=True)
        self.write_key_proj = nn.Linear(d, D_ADDR)
        self.write_val_proj = nn.Linear(d, D_VAL)

        # ---- read -----------------------------------------------------------
        self.read_q_proj   = nn.Linear(d, D_ADDR, bias=True)
        self.read_out_proj = nn.Linear(D_VAL, d, bias=True)
        self.alpha         = nn.Parameter(torch.ones(1))   # gate init=1; output still 0

        # Safe-Zero: alpha=1 * W_out=0 -> exactly 0 contribution at init.
        # W_out gets gradient from step 1 (alpha=1 non-zero * dL/dh * context).
        # Gradient to alpha = dL/dh · W_out(ctx) = 0 at step 1, non-zero from step 2.
        nn.init.zeros_(self.read_out_proj.weight)
        nn.init.zeros_(self.read_out_proj.bias)

        # Uniform-attention init: read_q_proj = 0 -> all scores = 0 -> softmax = 1/N.
        # Context = mean(board_vals) from step 1, no key-query alignment required.
        # Once W_out has been trained (step 2+) and Zone A has written k to the board,
        # the mean context already contains k; alpha grows; then read_q_proj can
        # gradually learn to focus on Zone A's slots for higher signal-to-noise.
        nn.init.zeros_(self.read_q_proj.weight)
        nn.init.zeros_(self.read_q_proj.bias)

    # -------------------------------------------------------------------------

    def encode(self, x):
        """x: (B, SEQ_LEN) -> h: (B, SEQ_LEN, D_LOCAL)"""
        pos = torch.arange(x.size(1), device=x.device)
        h   = self.tok_emb(x) + self.pos_emb(pos)
        return self.encoder(h)

    def write(self, h):
        """
        h: (B, T, D_LOCAL) -> slot_keys (B, K, D_ADDR), slot_vals (B, K, D_VAL)
        K learned queries cross-attend h to produce K content-aware summaries.
        """
        B = h.size(0)
        Q            = self.write_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        summaries, _ = self.write_attn(Q, h, h)                            # (B, K, d)
        return self.write_key_proj(summaries), self.write_val_proj(summaries)

    def read(self, h, board_keys, board_vals, slot_mask=None):
        """
        Cross-attend the full board and inject into h via residual + alpha gate.

        h          : (B, T, D_LOCAL)
        board_keys : (B, total_slots, D_ADDR)
        board_vals : (B, total_slots, D_VAL)
        slot_mask  : optional (total_slots,) bool, True = visible  [Part 3 hook]

        Returns h + alpha * out_proj(context).
        At init: out_proj = 0 -> contribution is exactly 0.
        """
        Q      = self.read_q_proj(h)                                         # (B, T, D_ADDR)
        scores = torch.bmm(Q, board_keys.transpose(1, 2)) / (D_ADDR ** 0.5) # (B, T, total_slots)

        if slot_mask is not None:
            invisible = ~slot_mask.to(dtype=torch.bool, device=h.device)
            scores = scores.masked_fill(invisible.view(1, 1, -1), float('-inf'))

        weights = F.softmax(scores, dim=-1)                   # (B, T, total_slots)
        context = torch.bmm(weights, board_vals)               # (B, T, D_VAL)
        delta   = self.alpha * self.read_out_proj(context)    # (B, T, D_LOCAL)
        return h + delta
