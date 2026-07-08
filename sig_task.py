"""
sig_task.py -- SIG (Structured-value) selective field-extraction task.
Deterministic, seeded, FREEZE-able.

A "signature" has 3 fields, each from its own vocab:
  name       in [0, 16)   (field 0)
  ret_type   in [0, 8)    (field 1)
  param_type in [0, 8)    (field 2)
A sample also has a query in {0,1,2} (which field is asked about) and a
probe_value.  The binary label is whether the probe matches the signature at the
queried field:
      label = (probe_value == signature[query])
This forces SELECTIVE extraction: read the RIGHT field (chosen by query), not a
fixed one, not the whole thing.

Balancing so BOTH controls fail toward chance (50%):
  * probe (and any matched value) is confined to the COMMON sub-range [0, RET_V)
    that ALL fields share.  This is essential: name has values in [8,16) that no
    other field can take, so an unconfined probe >= 8 would reveal "matched field
    = name" and make label = (query == name) perfectly predictable from (query,
    probe) alone -- a No-signature leak.  Confining probe to [0,8) removes it.
    (The signature still carries name in [0,16) -- only the probe is confined, so
    selective extraction of the full-range name field is still exercised.)
  * the MATCHED FIELD m (the field probe equals) is sampled UNIFORMLY *first*,
    independent of positive/negative.  So P(m = name) = 1/3 in both positives and
    negatives -> "which field probe matches" carries no label signal (No-query
    fails).  (If m instead correlated with the label, No-query could exploit it --
    exactly what happens if the probe range differs by field.)
  * positive: query = m (label 1) ; negative: query = another field with a
    differing value (label 0).  query is then ~uniform in both -> P(label=1 |
    query) ~= 0.5 (No-signature fails, no query->label correlation).
"""

import torch

NAME_V  = 16
RET_V   = 8
PARAM_V = 8
VOCABS  = (NAME_V, RET_V, PARAM_V)   # per-field (per-query) vocab sizes
N_QUERY = 3

# ── token layout ─────────────────────────────────────────────────────────────
# value tokens (name/ret/param/probe share the ring)  : 0 .. 15
# query tokens                                          : 16, 17, 18
# MASK                                                  : 19
VAL_MAX    = NAME_V             # 16 -- widest value vocab
QUERY_BASE = VAL_MAX           # 16
MASK_ID    = VAL_MAX + N_QUERY  # 19
VOCAB_SIZE = VAL_MAX + N_QUERY + 1   # 20
SEQ_LEN    = 5                 # [name, ret, param, query, probe]


SHARED_V = RET_V   # common sub-range [0, SHARED_V) that all fields can take


def generate_batch(n: int, seed: int):
    """Returns (name, ret, param, query, probe, label). ~50/50, both controls fail."""
    g = torch.Generator().manual_seed(seed)
    name  = torch.randint(0, NAME_V,  (n,), generator=g)
    ret   = torch.randint(0, RET_V,   (n,), generator=g)
    param = torch.randint(0, PARAM_V, (n,), generator=g)

    ar = torch.arange(n)
    m  = torch.randint(0, 3, (n,), generator=g)            # matched field -- UNIFORM
    # the matched field's value must be in the shared range so probe < SHARED_V
    # (else a probe >= 8 reveals m = name).  Only name can exceed -> resample it.
    fix = (m == 0) & (name >= SHARED_V)
    name = torch.where(fix, torch.randint(0, SHARED_V, (n,), generator=g), name)
    sig = torch.stack([name, ret, param], dim=1)           # (n, 3)
    probe = sig[ar, m]                                     # < SHARED_V, matches field m

    positive = torch.rand(n, generator=g) < 0.5
    idx   = torch.arange(3).view(1, 3)
    other = idx != m.view(n, 1)                            # fields != m
    differ = other & (sig != probe.view(n, 1))            # != m AND value != probe
    has_diff = differ.any(1)
    q_diff = (torch.rand(n, 3, generator=g) * differ.float()).argmax(1)   # -> label 0
    q_any  = (torch.rand(n, 3, generator=g) * other.float()).argmax(1)    # fallback
    query_neg = torch.where(has_diff, q_diff, q_any)

    query = torch.where(positive, m, query_neg)
    label = (sig[ar, query] == probe).long()
    return name, ret, param, query, probe, label


def encode(name, ret, param, query, probe, mode: str = "full"):
    """
    (n, SEQ_LEN) tokens: [name, ret, param, query, probe].
    mode: 'full'    -> everything visible
          'noquery' -> query MASKed (can't know which field)
          'nosig'   -> name/ret/param MASKed (nothing to compare against)
    """
    n = name.size(0)
    toks = torch.empty(n, SEQ_LEN, dtype=torch.long)
    toks[:, 0] = name
    toks[:, 1] = ret
    toks[:, 2] = param
    toks[:, 3] = QUERY_BASE + query
    toks[:, 4] = probe
    if mode == "noquery":
        toks[:, 3] = MASK_ID
    elif mode == "nosig":
        toks[:, 0:3] = MASK_ID
    elif mode != "full":
        raise ValueError(f"unknown mode {mode!r}")
    return toks
