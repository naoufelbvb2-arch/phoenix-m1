"""
Slot-key differentiation diagnostic -- M1-Part3.

After training, runs Zone B's write mechanism on a held-out batch and computes
pairwise cosine similarity among the K slot keys.

Low mean off-diagonal similarity  -> distinct slots carry distinct semantic content
                                     (content-based addressing is working)
Near 1.0                          -> keys collapsed to near-identical vectors
                                     (slot collapse; model can't differentiate writes)

Zone A's similarity is computed for comparison but is not the primary metric:
Zone A only needs to differentiate the ONE value it writes (v2), so moderate
collapse in Zone A's keys is expected and acceptable.
"""

import torch
import torch.nn.functional as F

from board import K


def _pairwise_cosine(keys: torch.Tensor):
    """
    keys : (B, K, D_ADDR)
    Returns:
      mean_off_diag : scalar -- mean of off-diagonal cosine similarities across batch
      mean_mat      : (K, K) tensor -- batch-averaged pairwise cosine similarity matrix
    """
    normed   = F.normalize(keys, dim=-1)                     # (B, K, D_ADDR)
    sim      = torch.bmm(normed, normed.transpose(1, 2))     # (B, K, K)
    mean_mat = sim.mean(0)                                   # (K, K)

    off_mask     = ~torch.eye(K, dtype=torch.bool, device=keys.device)
    mean_off_diag = mean_mat[off_mask].mean().item()
    return mean_off_diag, mean_mat


def analyze_slots(model, x_a: torch.Tensor, x_b: torch.Tensor):
    """
    Run Zone A and Zone B write mechanisms and return slot-key similarity stats.

    Returns dict with:
      sim_A, sim_B        -- mean off-diagonal cosine similarity (scalar)
      mat_A, mat_B        -- (K, K) batch-averaged similarity matrices
    """
    model.eval()
    with torch.no_grad():
        h_a   = model.zone_a.encode(x_a)
        h_b   = model.zone_b.encode(x_b)
        keys_a, _ = model.zone_a.write(h_a)   # (B, K, D_ADDR)
        keys_b, _ = model.zone_b.write(h_b)   # (B, K, D_ADDR)

    sim_a, mat_a = _pairwise_cosine(keys_a)
    sim_b, mat_b = _pairwise_cosine(keys_b)
    return dict(sim_A=sim_a, sim_B=sim_b, mat_A=mat_a, mat_B=mat_b)


def print_similarity_report(stats: dict):
    """Print the similarity matrices and summary statistics."""
    for zone in ("A", "B"):
        sim = stats[f"sim_{zone}"]
        mat = stats[f"mat_{zone}"]
        print(f"\n  Zone {zone} slot-key cosine similarity  "
              f"(mean off-diag = {sim:.4f})")
        hdr = "        " + "".join(f"  s{j:<2}" for j in range(K))
        print(hdr)
        for i in range(K):
            row = f"  s{i:<5}" + "".join(f"{mat[i, j].item():6.3f}" for j in range(K))
            print(row)
