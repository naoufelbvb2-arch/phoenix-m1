"""
Passive Workspace Board -- M0-Part2.

No trainable parameters. Holds (key, value) slot pairs written by zones.
Each zone owns K=8 slots; the board is their concatenation.
"""

import torch

K      = 8    # slots per zone
D_ADDR = 64   # slot key (address) dimension
D_VAL  = 64   # slot value dimension


def make_board(keys_list, vals_list):
    """
    Concatenate per-zone slot tensors into a single board.

    keys_list : list of (B, K, D_ADDR) tensors, one per zone
    vals_list : list of (B, K, D_VAL)  tensors, one per zone

    Returns board_keys (B, total_slots, D_ADDR), board_vals (B, total_slots, D_VAL).
    """
    return torch.cat(keys_list, dim=1), torch.cat(vals_list, dim=1)
