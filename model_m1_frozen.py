"""
PhoenixM1Frozen -- M1-REBUILD: FROZEN-ADDRESS two-zone board model.

Correction under test (vs model_m1.py):  the board's slot KEYS (addresses) are
FIXED at birth and NEVER trained.  Zones learn WHAT payload to place at a pre-
assigned address and HOW to read, but not the address itself.

  * Frozen manual slot reservation : Zone A owns board slots 0..K-1, Zone B owns
    K..2K-1 (K=8 each; d_addr=d_val=64).
  * Frozen addressing keys          : one set of 2K approximately-orthogonal
    vectors (random -> QR), seeded once, registered as a NON-TRAINABLE buffer on
    each zone (requires_grad=False).  Zones never learn keys.
  * Write  : each zone's K learned write-queries cross-attend its hidden states
    and emit only the VALUE (payload in d_val); the key is the fixed buffer.
  * Read (DENSE): round 2, each zone cross-attends ALL of the OTHER zone's slots.
    Query = zone hidden state, keys = FROZEN slot-key buffers, values = written
    payloads.  Safe-Zero read (zero-init out_proj + small positive gate ~0.1).
  * Two-round protocol + two heads (head_a on cat(k0,k1 reads), head_b on k2
    read) exactly as model_m1.py.
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


def make_frozen_keys(n_slots: int, d_addr: int, seed: int) -> torch.Tensor:
    """n_slots approximately-orthogonal unit key vectors in d_addr, seeded+fixed."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d_addr, n_slots, generator=g)     # tall (d_addr >= n_slots)
    Q, _ = torch.linalg.qr(M)                          # (d_addr, n_slots) orthonormal cols
    return Q.T.contiguous()                            # (n_slots, d_addr) orthonormal rows


class ZoneFrozen(nn.Module):
    def __init__(self, name: str, slot_keys: torch.Tensor):
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

        # Write: K learned queries cross-attend hidden states -> K PAYLOADS only.
        # No key projection: the address is the fixed buffer below.
        self.write_queries  = nn.Parameter(torch.randn(K, d))
        self.write_attn     = nn.MultiheadAttention(d, num_heads=4, dropout=0.0,
                                                     batch_first=True)
        self.write_val_proj = nn.Linear(d, D_VAL)

        # FROZEN addresses: this zone's K slot keys, non-trainable.
        self.register_buffer("slot_keys", slot_keys)   # (K, D_ADDR), requires_grad=False

        # Read: query proj + Safe-Zero out_proj + small positive gate.
        self.read_q_proj   = nn.Linear(d, D_ADDR)
        self.read_out_proj = nn.Linear(D_VAL, d)        # Safe-Zero: zero-init
        self.alpha         = nn.Parameter(torch.tensor([0.1]))
        nn.init.zeros_(self.read_out_proj.weight)
        nn.init.zeros_(self.read_out_proj.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return self.encoder(self.tok_emb(x) + self.pos_emb(pos))

    def write(self, h: torch.Tensor):
        """Returns (frozen keys, learned payloads), each (B, K, .)."""
        B = h.size(0)
        Q = self.write_queries.unsqueeze(0).expand(B, -1, -1)
        summaries, _ = self.write_attn(Q, h, h)
        vals = self.write_val_proj(summaries)                       # (B, K, D_VAL)
        keys = self.slot_keys.unsqueeze(0).expand(B, -1, -1)        # (B, K, D_ADDR) frozen
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


class PhoenixM1Frozen(nn.Module):
    def __init__(self, key_seed: int = 20260706):
        super().__init__()
        d = D_LOCAL
        all_keys = make_frozen_keys(2 * K, D_ADDR, key_seed)   # (2K, D_ADDR)
        self.zone_a = ZoneFrozen("A", all_keys[:K].clone())
        self.zone_b = ZoneFrozen("B", all_keys[K:].clone())

        self.head_a = nn.Sequential(
            nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )
        self.head_b = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        # Round 1: encode + write (payloads to fixed addresses)
        h_a = self.zone_a.encode(x_a)
        h_b = self.zone_b.encode(x_b)
        keys_a, vals_a = self.zone_a.write(h_a)
        keys_b, vals_b = self.zone_b.write(h_b)
        board_keys, board_vals = make_board([keys_a, keys_b], [vals_a, vals_b])

        # Round 2: DENSE cross-stream read -- each zone reads ALL of the other
        # zone's slots (cross-stream prevents the self-loop; no sub-masking --
        # the fixed distinct keys are what separate v0/v1 now).
        device = x_a.device
        mask_for_a = torch.zeros(2 * K, dtype=torch.bool, device=device)
        mask_for_a[K:] = True     # Zone A sees all of Zone B's slots
        mask_for_b = torch.zeros(2 * K, dtype=torch.bool, device=device)
        mask_for_b[:K] = True     # Zone B sees all of Zone A's slots
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
