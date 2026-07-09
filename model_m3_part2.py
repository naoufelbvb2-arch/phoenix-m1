"""
model_m3_part2.py -- FUNCTIONAL plug-and-play: the appended Zone D must IGNITE on
an already-populated frozen board and compose a NEW decision that needs BOTH old
zones (A: signature values, B: query + processed result).

Setup (all of A/B stays FROZEN and bit-exact -- verified in run_m3_part2.py):
  * Board (13 slots), same crowded layout as M3-Part1:
      slots 0..8   : Zone A frozen signature (name, 4 param VALUES, 4 VALIDITY)
      slots 9,10   : Zone B FIXED-ASSIGNMENT write of its query + result
                     (a frozen embedding of the raw query index + B's decision;
                     B never reads its own slots, so this cannot change B's logits)
      slots 11,12  : Zone D's own slots (empty payload -- D is a pure consumer here)
  * Zone D's NEW task (over IN-RANGE queries so the value is really on the board):
        d_label = category(params[query])  XOR  result
      category(v) = v // (PARAM_V//2)  (a NEW high/low partition the SIG task never
      used); result = B's processed decision (the old SIG label == an independent
      fair coin, so XOR forces both ablations to EXACTLY chance -- no majority leak).
    To solve it D MUST (a) read B's slot 9 to know WHICH field is queried,
    (b) read A's value slots to get that field's value, (c) read B's slot 10 for
    the result bit, (d) apply its own rule.  Neither old zone alone suffices.

Zone D is a blind content-addressed consumer (fixed learned queries dotted against
the frozen board keys, Safe-Zero head) -- it shares no weights with A/B and reads
the board purely by key-matching.  A slot_mask supports the NEEDS_BOTH ablations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sig_task_var import (
    VOCAB_SIZE, N_PARAM, PARAM_V, QUERY_BASE, MASK_ID,
)
from board import D_ADDR, D_VAL, make_board
from model_m1_frozen import make_frozen_keys
from model_sig_var import (
    PhoenixSigVar, A_SLOTS, B_SLOTS, TOTAL_SLOTS as BASE_SLOTS,   # 9, 2, 11
)

D_OWN_SLOTS  = 2
TOTAL_SLOTS  = BASE_SLOTS + D_OWN_SLOTS          # 13
CATEGORY_DIV = PARAM_V // 2                       # 4 -> high/low partition

# board slot index blocks
A_BLOCK = list(range(0, A_SLOTS))                 # 0..8
B_BLOCK = list(range(A_SLOTS, A_SLOTS + B_SLOTS)) # 9,10
D_BLOCK = list(range(A_SLOTS + B_SLOTS, TOTAL_SLOTS))  # 11,12


class ZoneDTask(nn.Module):
    """Blind POINTER-DEREFERENCING consumer of the frozen board (two-hop read).

    The task needs indirection -- read A's value at the field B points to -- which
    a single fixed-query read cannot do (its queries are constant across samples).
    So Zone D reads in two hops, purely by content-addressing the board:
      hop-1 : fixed queries gather B's block  -> a POINTER representation (which q)
      deref : map that pointer -> an ADDRESS  -> hop-2 content-reads A's value[q]
      result: a fixed query reads B's result slot
    A small head then applies D's own rule (category XOR result) on the two clean
    reads.  Shares no weights with A/B; obeys the ablation slot_mask on every hop.

    Annealed aux (reconstruct q, result, and the dereferenced value) bootstraps the
    two-hop addressing on the crowded frozen board, then anneals to 0."""
    def __init__(self, n_ptr: int = 6, n_result: int = 4):
        super().__init__()
        self.q_val    = nn.Parameter(torch.randn(N_PARAM, D_ADDR))   # read the 4 value slots
        self.q_ptr    = nn.Parameter(torch.randn(n_ptr, D_ADDR))     # read B's pointer slot
        self.q_result = nn.Parameter(torch.randn(n_result, D_ADDR))  # read B's result slot
        # learnable read temperature: soft start (exp(1)~2.7) so addressing can
        # form, then sharpens to isolate individual slots on the crowded board.
        self.log_temp = nn.Parameter(torch.tensor(1.0))
        self.sel_head = nn.Sequential(                               # pointer -> field selection
            nn.Linear(n_ptr * D_VAL, 128), nn.GELU(), nn.Linear(128, N_PARAM),
        )
        feat_dim = D_VAL + n_result * D_VAL                          # value[q] + result
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, 2),
        )
        nn.init.zeros_(self.head[-1].weight)     # Safe-Zero: neutral logits at init
        nn.init.zeros_(self.head[-1].bias)
        # aux scaffold heads (annealed)
        self.aux_query  = nn.Linear(n_ptr * D_VAL, N_PARAM)    # pointer -> q
        self.aux_result = nn.Linear(n_result * D_VAL, 2)       # result read -> result bit
        self.aux_value  = nn.Linear(D_VAL, PARAM_V)            # gathered read -> value[q]

    def _attn(self, q, board_keys, board_vals, slot_mask):
        """Content-addressed read.  q: (B, nq, D_ADDR).  Returns (B, nq, D_VAL)."""
        scores = torch.bmm(q, board_keys.transpose(1, 2)) * self.log_temp.exp()
        if slot_mask is not None:
            invisible = ~slot_mask.to(dtype=torch.bool, device=board_keys.device)
            scores = scores.masked_fill(invisible.view(1, 1, -1), float('-inf'))
        w = F.softmax(scores, dim=-1)
        return torch.bmm(w, board_vals)

    def _reads(self, board_keys, board_vals, slot_mask=None):
        B = board_keys.size(0)
        # fixed reads of the 4 value slots
        v_stack = self._attn(self.q_val.unsqueeze(0).expand(B, -1, -1),
                             board_keys, board_vals, slot_mask)      # (B, N_PARAM, D_VAL)
        # pointer read from B -> selection distribution over fields
        ptr = self._attn(self.q_ptr.unsqueeze(0).expand(B, -1, -1),
                         board_keys, board_vals, slot_mask).reshape(B, -1)
        sel_logits = self.sel_head(ptr)                             # (B, N_PARAM)
        alpha = F.softmax(sel_logits, dim=-1).unsqueeze(1)          # (B, 1, N_PARAM)
        val = torch.bmm(alpha, v_stack).squeeze(1)                  # (B, D_VAL) = value[q]
        # result read from B
        res = self._attn(self.q_result.unsqueeze(0).expand(B, -1, -1),
                         board_keys, board_vals, slot_mask).reshape(B, -1)
        return ptr, sel_logits, val, res

    def forward(self, board_keys, board_vals, slot_mask=None):
        ptr, sel_logits, val, res = self._reads(board_keys, board_vals, slot_mask)
        logits = self.head(torch.cat([val, res], dim=-1))
        return logits, (ptr, sel_logits, val, res)

    def aux_logits(self, reads):
        ptr, sel_logits, val, res = reads
        # sel_logits supervised by q is the key scaffold (teach the dereference)
        return self.aux_query(ptr), self.aux_result(res), self.aux_value(val), sel_logits


class PhoenixM3Part2(nn.Module):
    """Frozen A/B (SIG-Part3) + B fixed query/result write + appended Zone D task."""
    def __init__(self, base: PhoenixSigVar,
                 b_write_seed: int = 20260713, d_key_seed: int = 20260712,
                 n_ptr: int = 6):
        super().__init__()
        self.zone_a = base.zone_a          # frozen
        self.zone_b = base.zone_b          # frozen
        self.head_b = base.head_b          # frozen

        # B's FIXED-ASSIGNMENT write content: frozen orthonormal embeddings of the
        # raw query index (0..3) and the result bit (0/1).  Deterministic -> does
        # not change B's output (B never reads slots 9,10).
        self.register_buffer("query_emb",  make_frozen_keys(N_PARAM, D_VAL, b_write_seed))
        self.register_buffer("result_emb", make_frozen_keys(2, D_VAL, b_write_seed + 1))
        # D's own (empty) slots keep the board crowded exactly like Part1.
        self.register_buffer("d_keys", make_frozen_keys(D_OWN_SLOTS, D_ADDR, d_key_seed))

        self.zone_d = ZoneDTask(n_ptr=n_ptr)

    # ---- frozen old-zone computation ---------------------------------------
    def old_logits(self, toks_a, toks_b):
        """Exact SIG-Part3 B output (for OLD_STABLE) + B's predicted result bit."""
        h_b = self.zone_b.encode(toks_b)
        h_a = self.zone_a.encode(toks_a)
        keys_a, vals_a = self.zone_a.write(h_a, toks_a)
        keys_b, vals_b = self.zone_b.write(h_b)
        bk, bv = make_board([keys_a, keys_b], [vals_a, vals_b])
        mask_b = torch.zeros(BASE_SLOTS, dtype=torch.bool, device=toks_a.device)
        mask_b[:A_SLOTS] = True
        h_b2 = self.zone_b.read(h_b, bk, bv, mask_b)
        logits = self.head_b(h_b2.mean(1))
        return logits, vals_a

    def build_board(self, toks_a, toks_b):
        """13-slot crowded board: A frozen | B fixed query+result | D empty."""
        B = toks_a.size(0)
        device = toks_a.device
        # A (frozen signature)
        h_a = self.zone_a.encode(toks_a)
        keys_a, vals_a = self.zone_a.write(h_a, toks_a)
        # B's frozen output -> result bit; raw query index from its input tokens
        logits_b, _ = self.old_logits(toks_a, toks_b)
        result = logits_b.argmax(1)                              # B's processed decision
        query = (toks_b[:, 0] - QUERY_BASE).clamp_(0, N_PARAM - 1)
        # B FIXED-ASSIGNMENT write into its owned slot keys
        keys_b = self.zone_b.slot_keys.unsqueeze(0).expand(B, -1, -1)
        vals_b = torch.stack([self.query_emb[query], self.result_emb[result]], dim=1)
        # D's own (empty) slots
        keys_d = self.d_keys.unsqueeze(0).expand(B, -1, -1)
        vals_d = torch.zeros(B, D_OWN_SLOTS, D_VAL, device=device)
        bk, bv = make_board([keys_a, keys_b, keys_d], [vals_a, vals_b, vals_d])
        return bk, bv


def slot_mask(visible_blocks, device=None):
    """Build a (TOTAL_SLOTS,) bool mask; True for slots in the given index blocks."""
    m = torch.zeros(TOTAL_SLOTS, dtype=torch.bool, device=device)
    for blk in visible_blocks:
        m[blk] = True
    return m
