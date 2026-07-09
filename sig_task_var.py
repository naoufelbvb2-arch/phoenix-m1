"""
sig_task_var.py -- VARIABLE-ARITY signature task (SIG-Part3).  Seeded, FREEZE-able.
(sig_task.py, the fixed-3-field task, is left intact.)

Closes the original M1 open item: fixed write-assignment assumed a known, fixed
number of values.  Here the number of parameter fields VARIES per sample (real
functions have different parameter counts).  Placement stays ORDINAL and input-
INDEPENDENT ("param i -> slot i"); only the NUMBER of occupied slots varies, up
to a fixed cap.  Variable ARITY, not variable PLACEMENT.

Signature: name in [0,16), plus arity a in {1..4} parameter fields, each in
[0,PARAM_V=8).  Cap N_PARAM=4.  present[i] = (i < a).
Query = (param index q in {0..3}, probe in [0,8)).
Label (binary), 3 cases the model must distinguish:
  (i)  q < a and params[q] == probe          -> 1   (exists & matches)
  (ii) q < a and params[q] != probe          -> 0   (exists & differs)
  (iii) q >= a (param does not exist)         -> 0   (out of range)   <-- new
Case (iii) forces reading the VALIDITY signal, not just the value.
"""

import torch

NAME_V   = 16
PARAM_V  = 8
N_PARAM  = 4     # cap (deliberate known maximum)

# ── token layout ─────────────────────────────────────────────────────────────
# values (name/param/probe) : 0 .. 15 ; query index tokens : 16..19 ;
# PAD (absent param) : 20 ; MASK : 21
VAL_MAX    = NAME_V            # 16
QUERY_BASE = VAL_MAX          # 16  (query q -> 16 + q)
PAD_ID     = VAL_MAX + N_PARAM        # 20
MASK_ID    = VAL_MAX + N_PARAM + 1    # 21
VOCAB_SIZE = VAL_MAX + N_PARAM + 2    # 22

SEQ_A = 1 + N_PARAM   # Zone A view: [name, p0, p1, p2, p3]  = 5
SEQ_B = 2             # Zone B view: [query, probe]


def generate_batch(n: int, seed: int):
    """Returns name(n,), params(n,4), present(n,4 bool), query(n,), probe(n,), label(n,)."""
    g = torch.Generator().manual_seed(seed)
    ar = torch.arange(n)

    name   = torch.randint(0, NAME_V, (n,), generator=g)
    arity  = torch.randint(1, N_PARAM + 1, (n,), generator=g)          # 1..4
    params = torch.randint(0, PARAM_V, (n, N_PARAM), generator=g)      # values (absent ignored)
    present = torch.arange(N_PARAM).view(1, N_PARAM) < arity.view(n, 1)  # (n,4) present[i]=i<a

    query = torch.randint(0, N_PARAM, (n,), generator=g)              # 0..3
    in_range = query < arity                                          # present[query]
    pq = params[ar, query]                                            # queried value (if present)

    match      = torch.rand(n, generator=g) < 0.5
    probe_diff = (pq + 1 + torch.randint(0, PARAM_V - 1, (n,), generator=g)) % PARAM_V
    probe_oor  = torch.randint(0, PARAM_V, (n,), generator=g)
    probe = torch.where(in_range,
                        torch.where(match, pq, probe_diff),
                        probe_oor)
    label = (in_range & (params[ar, query] == probe)).long()
    return name, params, present, query, probe, label


def encode_a(name, params, present):
    """Zone A (signature-holder) view: [name, p0..p3] with absent params -> PAD."""
    n = name.size(0)
    toks = torch.empty(n, SEQ_A, dtype=torch.long)
    toks[:, 0] = name
    for i in range(N_PARAM):
        toks[:, 1 + i] = torch.where(present[:, i], params[:, i],
                                     torch.full_like(params[:, i], PAD_ID))
    return toks


def encode_b(query, probe):
    """Zone B (querier) view: [query_token, probe]."""
    n = query.size(0)
    toks = torch.empty(n, SEQ_B, dtype=torch.long)
    toks[:, 0] = QUERY_BASE + query
    toks[:, 1] = probe
    return toks
