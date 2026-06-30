"""
Two-round Phoenix model -- M0-Part2.

Round 1 (Broadcast):
  Both zones encode their streams independently, then write K slots each
  to the passive workspace board.

Round 2 (Refinement):
  Both zones cross-attend the filled board (Safe-Zero gated), integrating
  cross-zone information into their residual streams.

Output:
  Zone B: marker-shift extraction at the v-token position of h_B2 -> head.
  Using VALUE_MARKER shift (same as Part 1's dense ceiling) avoids the
  1/16-dilution problem of mean-pooling; the gradient is concentrated at
  exactly the position that encodes v in Zone B's stream.
  After Round 2 that position also carries k from the board read, so
  the 128-dim representation at that slot encodes both k and v.

Auxiliary head:
  Zone A has no direct loss connection (no head), so its write parameters
  receive zero gradient through the board read gate when that gate is small.
  An auxiliary k-prediction head on Zone A's slot values (vals_a) provides
  direct gradient to Zone A's write from training step 1, bootstrapping
  the board channel.  Not used at inference.
"""

import torch
import torch.nn as nn

from task  import VOCAB_SIZE, VALUE_MARKER
from board import make_board, D_VAL
from zones import Zone, D_LOCAL


class PhoenixM0P2(nn.Module):
    def __init__(self):
        super().__init__()
        self.zone_a     = Zone("A")
        self.zone_b     = Zone("B")
        self.head       = nn.Sequential(
            nn.Linear(D_LOCAL, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE)
        )   # MLP head: (k+v)%64 from mixed v_enc+k_correction needs nonlinearity
        self.aux_k_head = nn.Linear(D_VAL,   VOCAB_SIZE)   # Zone A slots -> k

        # Safe-write: Zone B's slot VALUES start at zero so the board holds only
        # Zone A's k-encoding at init (no noise from Zone B's own write).
        # Zone B's write_val_proj still receives gradient and learns; it just starts
        # silent so the board channel carries a pure k-signal from epoch 1.
        nn.init.zeros_(self.zone_b.write_val_proj.weight)
        nn.init.zeros_(self.zone_b.write_val_proj.bias)

    def forward(self, x_a, x_b, slot_mask=None):
        """
        x_a      : (B, SEQ_LEN) stream_A tokens
        x_b      : (B, SEQ_LEN) stream_B tokens
        slot_mask: optional (total_slots,) bool [Part 3 lesion hook]

        Returns:
          logits     (B, VOCAB_SIZE)  main prediction (Zone B, v-token rep)
          aux_logits (B, VOCAB_SIZE)  k prediction from Zone A slots
        """
        # -- Round 1: encode + write ------------------------------------------
        h_a = self.zone_a.encode(x_a)
        h_b = self.zone_b.encode(x_b)

        keys_a, vals_a = self.zone_a.write(h_a)
        keys_b, vals_b = self.zone_b.write(h_b)

        board_keys, board_vals = make_board(
            [keys_a, keys_b], [vals_a, vals_b]
        )

        # -- Round 2: read + integrate ----------------------------------------
        h_a = self.zone_a.read(h_a, board_keys, board_vals, slot_mask)
        h_b = self.zone_b.read(h_b, board_keys, board_vals, slot_mask)

        # -- Main output: Zone B, marker-shift at v-token position ------------
        # h_b2[b, vpos+1, :] encodes v (from Zone B encoder) + k (from board read)
        vm     = (x_b == VALUE_MARKER)            # (B, T) bool
        v_mask = torch.zeros_like(vm)
        v_mask[:, 1:] = vm[:, :-1]               # shift: marker -> v-token pos
        val_rep = (h_b * v_mask.unsqueeze(-1).float()).sum(1)  # (B, D_LOCAL)
        logits  = self.head(val_rep)

        # -- Auxiliary: k prediction from Zone A slot values ------------------
        aux_logits = self.aux_k_head(vals_a.mean(dim=1))

        return logits, aux_logits
