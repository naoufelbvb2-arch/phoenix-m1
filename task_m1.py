"""
M1-Part1 -- Typed Multi-Attribute Binding, deterministic seeded data generator.

Vocab: tokens [0, VOCAB_SIZE) regular + 2*N_TYPES marker tokens.
  KEY_MARKER(t)   = VOCAB_SIZE + 2*t
  VALUE_MARKER(t) = VOCAB_SIZE + 2*t + 1
TOTAL_VOCAB = VOCAB_SIZE + 2*N_TYPES.

Bidirectional split (n_types types, default 3):
  Group 1 = types [0, n_types-2] : stream_A holds KEY_t,   stream_B holds VALUE_t.
  Group 2 = type   n_types-1     : stream_B holds KEY_t,   stream_A holds VALUE_t.
  (Default n_types=3 -> group1={0,1}, group2={2}: A holds k0,k1; B holds v0,v1;
   B holds k2; A holds v2 -- matching the M1-Part1 spec exactly.)

Each stream is seq_len long and carries exactly n_types (marker, token) pairs,
one per type. To guarantee pairs never collide while still varying position
sample-to-sample, each type owns a fixed-size block of the sequence and the
pair is placed at a random offset within that block (type identity is always
recoverable from the marker token, independent of block or offset).

label_A = sum_{t in group1} (k_t + v_t)  mod vocab_size  -- needs A's keys + B's values.
label_B = sum_{t in group2} (k_t + v_t)  mod vocab_size  -- needs B's key  + A's value.
Neither stream alone determines either label.
"""

import numpy as np
import torch

VOCAB_SIZE = 64
SEQ_LEN    = 24
N_TYPES    = 3

TOTAL_VOCAB = VOCAB_SIZE + 2 * N_TYPES   # 70


def key_marker(t: int) -> int:
    return VOCAB_SIZE + 2 * t


def value_marker(t: int) -> int:
    return VOCAB_SIZE + 2 * t + 1


def generate_batch(batch_size: int, n_types: int = N_TYPES, seq_len: int = SEQ_LEN,
                    vocab_size: int = VOCAB_SIZE, seed=None):
    """
    Returns (stream_A, stream_B, label_A, label_B) as CPU LongTensors:
      stream_A, stream_B : (B, seq_len)
      label_A, label_B   : (B,)
    """
    assert n_types >= 2, "need >=2 types for a bidirectional split"
    block = seq_len // n_types
    assert block >= 2, "seq_len must fit n_types non-overlapping (marker, token) pairs"

    rng = np.random.default_rng(seed)
    idx = np.arange(batch_size)

    A = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    B = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)

    keys   = rng.integers(0, vocab_size, size=(batch_size, n_types), dtype=np.int64)
    values = rng.integers(0, vocab_size, size=(batch_size, n_types), dtype=np.int64)

    for t in range(n_types):
        block_start = t * block
        offset = rng.integers(0, block - 1, size=batch_size)   # [0, block-2]
        pos    = block_start + offset

        km = vocab_size + 2 * t
        vm = vocab_size + 2 * t + 1

        if t < n_types - 1:                        # group 1: A=key, B=value
            A[idx, pos]     = km
            A[idx, pos + 1] = keys[:, t]
            B[idx, pos]     = vm
            B[idx, pos + 1] = values[:, t]
        else:                                       # group 2 (last type): B=key, A=value
            B[idx, pos]     = km
            B[idx, pos + 1] = keys[:, t]
            A[idx, pos]     = vm
            A[idx, pos + 1] = values[:, t]

    label_A = (keys[:, :n_types - 1].sum(axis=1) + values[:, :n_types - 1].sum(axis=1)) % vocab_size
    label_B = (keys[:, n_types - 1] + values[:, n_types - 1]) % vocab_size

    return (
        torch.from_numpy(A),
        torch.from_numpy(B),
        torch.from_numpy(label_A),
        torch.from_numpy(label_B),
    )
