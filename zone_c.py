"""
Zone C -- a brand-new BLIND consumer of the frozen M1 board (M1-Part4).

Forward-compatibility probe.  Zone C shares NO weights with Zone A/B and never
sees stream_A or stream_B.  Its ONLY input is the frozen board (keys + values).
It must reconstruct the two separated values v0 and v1 -- the pair Part-3
struggled to keep distinct -- through two INDEPENDENT learned-query cross-
attention read heads + two INDEPENDENT MLP classifiers.

Design guarantees the test is honest:
  * Pure content addressing: each head scores its learned queries against the
    board KEYS (softmax over slots) and pulls a weighted sum of the board VALUES.
  * Permutation-invariant over slots by construction, and NO positional / slot-
    index signal is ever fed in -> Zone C structurally CANNOT index-memorize.
  * Safe-Zero: each classifier's output layer is zero-init (neutral logits at
    start; gradient flows in after one warmup step, exactly like read_out_proj).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from board import D_ADDR, D_VAL
from task_m1 import VOCAB_SIZE


class ZoneC(nn.Module):
    def __init__(self, n_query: int = 4):
        super().__init__()
        self.n_query = n_query
        # Separate learned query sets (addresses) for v0 and v1, in D_ADDR space
        # so they dot directly against the frozen board keys.  randn init (std 1)
        # so the queries start differentiated (slot-collapse lesson from Zone B).
        self.q0 = nn.Parameter(torch.randn(n_query, D_ADDR))
        self.q1 = nn.Parameter(torch.randn(n_query, D_ADDR))
        self.head0 = self._make_head(n_query * D_VAL)
        self.head1 = self._make_head(n_query * D_VAL)

    @staticmethod
    def _make_head(in_dim: int) -> nn.Sequential:
        h = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, VOCAB_SIZE),
        )
        nn.init.zeros_(h[-1].weight)   # Safe-Zero: neutral logits at init
        nn.init.zeros_(h[-1].bias)
        return h

    def _read(self, q: torch.Tensor, board_keys: torch.Tensor,
              board_vals: torch.Tensor) -> torch.Tensor:
        """Content-addressed read.  Invariant to slot order (softmax over slots)."""
        B = board_keys.size(0)
        Q = q.unsqueeze(0).expand(B, -1, -1)                    # (B, n_query, D_ADDR)
        scores = torch.bmm(Q, board_keys.transpose(1, 2))       # (B, n_query, S)
        scores = scores / (D_ADDR ** 0.5)
        w = F.softmax(scores, dim=-1)                           # over slots
        ctx = torch.bmm(w, board_vals)                          # (B, n_query, D_VAL)
        return ctx.reshape(B, -1)                               # (B, n_query*D_VAL)

    def forward(self, board_keys: torch.Tensor, board_vals: torch.Tensor):
        c0 = self._read(self.q0, board_keys, board_vals)
        c1 = self._read(self.q1, board_keys, board_vals)
        return self.head0(c0), self.head1(c1)
