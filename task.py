"""
M0-Part1 -- Modular Key-Value Matching, deterministic seeded data generator.

Vocab: tokens [0, 64)  + KEY_MARKER=64  VALUE_MARKER=65  -> TOTAL_VOCAB=66
Sequence length: 16 per stream.

stream_A: random tokens; one slot holds KEY_MARKER, the next holds k in [0, key_range).
stream_B: random tokens; one slot holds VALUE_MARKER, the next holds v in [0, 64).
label   : (k + v) % 64  --  requires BOTH streams; neither alone suffices.
"""

import numpy as np
import torch

VOCAB_SIZE   = 64
SEQ_LEN      = 16
KEY_MARKER   = 64   # injected into stream_A
VALUE_MARKER = 65   # injected into stream_B
TOTAL_VOCAB  = 66   # 0-63 regular + 2 specials


def generate_batch(batch_size: int, key_range: int = VOCAB_SIZE, seed=None):
    """
    Returns (stream_A, stream_B, labels) as CPU LongTensors of shapes
    (B, SEQ_LEN), (B, SEQ_LEN), (B,).

    key_range -- upper bound on k values (default 64).
                 Reduce to tighten the key channel in later ablations.
    seed      -- int or None for a reproducible draw.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(batch_size)

    # Fill both streams with uniformly random regular tokens.
    A = rng.integers(0, VOCAB_SIZE, size=(batch_size, SEQ_LEN), dtype=np.int64)
    B = rng.integers(0, VOCAB_SIZE, size=(batch_size, SEQ_LEN), dtype=np.int64)

    # Inject KEY_MARKER + k into stream_A (vectorised -- no Python loop).
    kpos = rng.integers(0, SEQ_LEN - 1, size=batch_size)   # [0, SEQ_LEN-2]
    k    = rng.integers(0, key_range,   size=batch_size, dtype=np.int64)
    A[idx, kpos]     = KEY_MARKER
    A[idx, kpos + 1] = k

    # Inject VALUE_MARKER + v into stream_B.
    vpos = rng.integers(0, SEQ_LEN - 1, size=batch_size)
    v    = rng.integers(0, VOCAB_SIZE,  size=batch_size, dtype=np.int64)
    B[idx, vpos]     = VALUE_MARKER
    B[idx, vpos + 1] = v

    labels = (k + v) % VOCAB_SIZE

    return (
        torch.from_numpy(A),
        torch.from_numpy(B),
        torch.from_numpy(labels),
    )


# Descriptive alias used by callers that want explicit per-stream tensors.
get_streams = generate_batch


def generate_all_pairs(repeats: int = 2, key_range: int = VOCAB_SIZE, seed: int = 0):
    """
    Systematic generator: covers every (k, v) pair exactly `repeats` times.
    Total samples = key_range * VOCAB_SIZE * repeats  (default 64*64*2 = 8192).
    Each repeat has independently drawn random positions and noise tokens.
    Returns a shuffled (stream_A, stream_B, labels) triple.

    Using this as a training set guarantees zero unseen (k,v) pairs,
    eliminating the ~8% coverage gap that caps accuracy at ~94% with a
    random 10 000-sample draw.
    """
    rng = np.random.default_rng(seed)
    n   = key_range * VOCAB_SIZE * repeats

    # Build the full (k, v) grid, tiled `repeats` times.
    ks_one = np.repeat(np.arange(key_range,  dtype=np.int64), VOCAB_SIZE)
    vs_one = np.tile(  np.arange(VOCAB_SIZE, dtype=np.int64), key_range )
    ks = np.tile(ks_one, repeats)
    vs = np.tile(vs_one, repeats)

    A = rng.integers(0, VOCAB_SIZE, size=(n, SEQ_LEN), dtype=np.int64)
    B = rng.integers(0, VOCAB_SIZE, size=(n, SEQ_LEN), dtype=np.int64)

    idx  = np.arange(n)
    kpos = rng.integers(0, SEQ_LEN - 1, size=n)
    vpos = rng.integers(0, SEQ_LEN - 1, size=n)

    A[idx, kpos]     = KEY_MARKER
    A[idx, kpos + 1] = ks
    B[idx, vpos]     = VALUE_MARKER
    B[idx, vpos + 1] = vs

    labels = (ks + vs) % VOCAB_SIZE

    perm = rng.permutation(n)
    return (
        torch.from_numpy(A[perm]),
        torch.from_numpy(B[perm]),
        torch.from_numpy(labels[perm]),
    )
