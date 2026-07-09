"""
comp_task.py -- COMP (Compositional) relational type-compatibility task.
Deterministic, seeded, FREEZE-able.

A signature has two type fields: ret_type in [0,8) and param_type in [0,8).
Categories: types 0-3 = "numeric", 4-7 = "string"  ->  category(t) = t // 4.
Label is binary and RELATIONAL:
      label = (category(ret) == category(param))       (same category?)
Genuinely compositional: the answer needs BOTH fields and their relation.  It is
NOT two comparisons against a fixed probe -- it is a relation between the fields.

Why both single-field controls fail at chance (no balancing tricks needed):
ret and param are drawn uniformly and INDEPENDENTLY, so given ret, the hidden
param's category is still uniform -> P(label=1 | ret) = 0.5 for every ret (and
symmetrically for param).  A single field carries zero information about the
relation.  Same-/different-category are ~50/50 and all type values are varied.
"""

import torch

T_V      = 8      # type vocab (ret and param each in [0, T_V))
CAT_SIZE = 4      # first CAT_SIZE types = numeric, rest = string
N_CAT    = T_V // CAT_SIZE   # 2 categories

# ── token layout ─────────────────────────────────────────────────────────────
# value tokens (ret/param share) : 0 .. 7 ; MASK : 8
MASK_ID    = T_V          # 8
VOCAB_SIZE = T_V + 1      # 9
SEQ_LEN    = 2            # [ret, param]


def category(t):
    return t // CAT_SIZE


def generate_batch(n: int, seed: int):
    """Returns (ret, param, label).  Uniform-independent fields."""
    g = torch.Generator().manual_seed(seed)
    ret   = torch.randint(0, T_V, (n,), generator=g)
    param = torch.randint(0, T_V, (n,), generator=g)
    label = (category(ret) == category(param)).long()
    return ret, param, label


def encode(ret, param, mode: str = "full"):
    """
    (n, 2) tokens: [ret, param].
    mode: 'full'      -> both visible
          'retonly'   -> param MASKed (relation unknowable)
          'paramonly' -> ret MASKed
    """
    n = ret.size(0)
    toks = torch.empty(n, SEQ_LEN, dtype=torch.long)
    toks[:, 0] = ret
    toks[:, 1] = param
    if mode == "retonly":
        toks[:, 1] = MASK_ID
    elif mode == "paramonly":
        toks[:, 0] = MASK_ID
    elif mode != "full":
        raise ValueError(f"unknown mode {mode!r}")
    return toks
