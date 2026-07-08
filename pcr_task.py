"""
pcr_task.py -- PCR (Propose-Check-Repair) synthetic task.  Seeded, FREEZE-able.

State is an integer mod M.  A sample is:
  s0   : initial state in [0,M)
  ops  : a sequence of L operation indices in [0,N_OPS)
  T    : a target in [0,M)  (independent of s0/ops)
  x    : execute(s0, ops) mod M   -- the raw "proposed" output before repair
The required REPAIR is the single corrective op that maps x -> T; with add-type
repair that is "add (T-x) mod M", so the repair delta is
  label = r = (T - x) mod M
Applying it gives the FINAL corrected output (x + r) mod M == T.

Why the label is r, not T:  if the label were T and the model saw T (checker
side), it could COPY T -- the checker-only control would then pass, breaking the
bidirectional test.  r requires BOTH the proposer's x (via execution) AND the
checker's T, so neither half alone can produce it.  Accuracy on r is identical
to "corrected output equals T" (they are equivalent given x).

Information split (bidirectional):
  Proposer-side = (s0, ops)  -> determines x, NOT T   -> can't know r (T hidden)
  Checker-side  = T          -> NOT ops/x              -> can't know r (x hidden)
"""

import torch

M      = 16     # state modulus  (chance = 1/16 = 6.25%)
N_OPS  = 8      # number of distinct operations
L      = 2      # operations per sample
# NOTE on L: the full input space is M * N_OPS^L * M = 256 * 8^L.  At L=4 that is
# ~1M -- uncoverable in a CPU/time budget, so the model would have to GROK
# modular composition (it can't even fit train: verified train~=test~=0.13).
# L=2 -> 16,384 states, which the M0 coverage lesson covers, so the dense model
# learns it and the controls still fail.  (Deeper L is a grokking/GPU question.)

# 8 fixed operations, each a fixed BIJECTIVE lookup table (a random permutation
# of [0,M), seeded once and frozen).  We use LOOKUP ops rather than arithmetic
# (add/mul/sub) deliberately: arithmetic ops make x = execute(s0,ops) an additive
# modular SUM, and predicting r = (T - s0 - sum(deltas)) mod M then requires the
# dense model to learn multi-term modular addition -- which stalls at a partial-
# sum optimization plateau (~25%) on a CPU/time budget (verified L=2 additive
# train==test~=0.26).  Lookup ops make x a COMPOSITION OF PERMUTATIONS with no
# additive structure to stall on, so the covered mapping is learned by
# interpolation (M0 coverage lesson).  The repair r = (T - x) mod M is still the
# M0-proven 2-term modular step.  "execute -> check -> repair" and the
# bidirectional info split are unchanged.
_OP_SEED = 20260707


def _make_op_tables(n_ops: int, m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(m, generator=g) for _ in range(n_ops)])


# transition tables: _OP_TABLES[i, s] = state s after op i  (each row a bijection)
_OP_TABLES = _make_op_tables(N_OPS, M, _OP_SEED)

# ── token layout ─────────────────────────────────────────────────────────────
# values (s0, T) -> tokens 0..M-1 ; ops -> tokens M..M+N_OPS-1 ; MASK -> last
VAL_BASE   = 0
OP_BASE    = M
MASK_ID    = M + N_OPS          # 24
VOCAB_SIZE = M + N_OPS + 1      # 25
SEQ_LEN    = 1 + L + 1          # [s0, op0..op(L-1), T]


def execute(s0: torch.Tensor, ops: torch.Tensor) -> torch.Tensor:
    """s0:(B,) ops:(B,L) -> x:(B,).  Applies ops left-to-right, mod M."""
    x = s0.clone()
    for t in range(ops.size(1)):
        x = _OP_TABLES[ops[:, t], x]
    return x


def generate_batch(n: int, seed: int):
    """Returns (s0, ops, T, x, label=r).  Deterministic given seed."""
    g   = torch.Generator().manual_seed(seed)
    s0  = torch.randint(0, M, (n,), generator=g)
    ops = torch.randint(0, N_OPS, (n, L), generator=g)
    T   = torch.randint(0, M, (n,), generator=g)
    x   = execute(s0, ops)
    label = (T - x) % M
    return s0, ops, T, x, label


def encode(s0: torch.Tensor, ops: torch.Tensor, T: torch.Tensor, mode: str = "full"):
    """
    Build the (B, SEQ_LEN) token sequence.  Position 0 = s0, 1..L = ops, L+1 = T.
    mode: 'full'  -> everything visible
          'prop'  -> proposer-only: T is MASKed
          'check' -> checker-only : s0 and ops are MASKed
    """
    B = s0.size(0)
    toks = torch.empty(B, SEQ_LEN, dtype=torch.long)
    toks[:, 0]        = s0 + VAL_BASE
    toks[:, 1:1 + L]  = ops + OP_BASE
    toks[:, 1 + L]    = T + VAL_BASE
    if mode == "prop":
        toks[:, 1 + L] = MASK_ID
    elif mode == "check":
        toks[:, 0]       = MASK_ID
        toks[:, 1:1 + L] = MASK_ID
    elif mode != "full":
        raise ValueError(f"unknown mode {mode!r}")
    return toks
