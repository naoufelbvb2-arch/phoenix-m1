"""
M0-Part4 -- Channel-capacity sweeps.
Frozen: task.py, board.py, zones.py, model.py, lesion.py

Strategy: patch board.K / board.D_ADDR / board.D_VAL, then reload
zones / model / lesion so fresh Zone instances pick up the new constants.

IMPORTANT: importlib.reload() updates the module dict in-place.
Zone.read() looks up D_ADDR at call time from zones.__dict__.
Therefore each model MUST be fully trained and evaluated before
the next configure() call mutates the shared dict.

Sweep 1 (K-sweep):  K in {1, 4, 8},   d_addr = d_val = 64
Sweep 2 (aspect):   K * d_addr ~ 2048, d_val = d_addr
  WIDE-SHORT    K=4,  d_addr=512
  SQUARE        K=8,  d_addr=256
  NARROW-TALL   K=16, d_addr=128

Hyperparams: AUX_WEIGHT=1.0, N_TRAIN=20000, N_EPOCHS=8, SEED=42.
"""

import importlib
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

from task import generate_batch, KEY_MARKER

# frozen modules -- reloaded per-config
import board
import zones  as zones_mod
import model  as model_mod
import lesion as lesion_mod

# ---- fixed hyperparams -------------------------------------------------------
N_TRAIN    = 20_000
N_EPOCHS   = 8
BATCH_SIZE = 128
LR         = 1e-3
LR_ALPHA   = 1e-6   # near-freeze alpha; Part-3 showed this prevents gate collapse
AUX_WEIGHT = 1.0    # Part-3 best: AUX=1.0 -> 94.07%, AUX=3.0 -> 91.14%
LOG_EVERY  = 4
SEED       = 42

# ---- shared test set (built once, before any reload) ------------------------
A_te, B_te, y_te = generate_batch(4096, seed=9999)
test_loader = DataLoader(
    TensorDataset(A_te, B_te, y_te), batch_size=512, shuffle=False,
)


# ---- helpers -----------------------------------------------------------------

def extract_k(xa):
    """Token immediately after KEY_MARKER in stream_A."""
    km    = (xa == KEY_MARKER)
    kpos  = km.float().argmax(dim=1)
    k_idx = torch.clamp(kpos + 1, max=xa.size(1) - 1)
    return xa[torch.arange(xa.size(0)), k_idx]


def configure(K_val, D_ADDR_val, D_VAL_val):
    """
    Patch board constants, reload zones/model/lesion.
    After this: model_mod.PhoenixM0P2() builds a model with (K_val, D_ADDR_val, D_VAL_val).
    MUST train + eval each model before calling configure() again.
    """
    board.K      = K_val
    board.D_ADDR = D_ADDR_val
    board.D_VAL  = D_VAL_val
    importlib.reload(zones_mod)
    importlib.reload(model_mod)
    importlib.reload(lesion_mod)


def run_config(K_val, D_ADDR_val, D_VAL_val, desc):
    """Configure, build, train, eval one model. Returns (acc_c0, acc_c1)."""
    print(f"\n  [{desc}]  K={K_val}  d_addr={D_ADDR_val}  d_val={D_VAL_val}")
    configure(K_val, D_ADDR_val, D_VAL_val)

    torch.manual_seed(SEED)
    A_tr, B_tr, y_tr = generate_batch(N_TRAIN, seed=0)
    loader = DataLoader(
        TensorDataset(A_tr, B_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model   = model_mod.PhoenixM0P2()
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params={n_params:,}")

    alpha_params = [model.zone_a.alpha, model.zone_b.alpha]
    alpha_ids    = {id(p) for p in alpha_params}
    other_params = [p for p in model.parameters() if id(p) not in alpha_ids]
    opt = torch.optim.Adam([
        {'params': other_params, 'lr': LR},
        {'params': alpha_params, 'lr': LR_ALPHA},
    ])

    t0 = time.time()
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        ep_loss = 0.0; n = 0
        for xa, xb, y in loader:
            opt.zero_grad()
            logits, aux_logits = model(xa, xb)
            main_loss = loss_fn(logits, y)
            loss      = main_loss + AUX_WEIGHT * loss_fn(aux_logits, extract_k(xa))
            loss.backward()
            opt.step()
            ep_loss += main_loss.item(); n += 1

        if ep == 1 or ep % LOG_EVERY == 0:
            elapsed = time.time() - t0
            print(f"    ep {ep:2d}  loss={ep_loss/n:.4f}  t={elapsed:.0f}s")

    # eval BEFORE next configure() call
    acc_c0 = lesion_mod.eval_acc(model, test_loader, slot_mask=None)
    acc_c1 = lesion_mod.eval_acc(model, test_loader, slot_mask=lesion_mod.mask_c1())
    elapsed = time.time() - t0
    print(f"    -> ACC={acc_c0:.4f}  C1={acc_c1:.4f}  ({elapsed:.0f}s)")
    return acc_c0, acc_c1


# =============================================================================
# SWEEP 1: K-sweep  (d_addr = d_val = 64 fixed)
# =============================================================================
print("=" * 60)
print("SWEEP 1: K-sweep  (d_addr=d_val=64 fixed)")
print("=" * 60)
t_sw1 = time.time()

sw1 = {}
for K_val in [1, 4, 8]:
    sw1[K_val] = run_config(K_val, D_ADDR_val=64, D_VAL_val=64, desc=f"K={K_val}")

print(f"\nSWEEP 1 done in {time.time()-t_sw1:.0f}s")


# =============================================================================
# SWEEP 2: Aspect-ratio sweep  (K * d_addr ~ 2048, d_val = d_addr)
# =============================================================================
print()
print("=" * 60)
print("SWEEP 2: Aspect-ratio sweep  (K*d_addr~2048, d_val=d_addr)")
print("=" * 60)
t_sw2 = time.time()

aspect_configs = [
    (4,  512, 512, "WIDE-SHORT"),
    (8,  256, 256, "SQUARE"),
    (16, 128, 128, "NARROW-TALL"),
]
sw2 = {}
for K_val, D_ADDR_val, D_VAL_val, desc in aspect_configs:
    sw2[(K_val, D_ADDR_val)] = run_config(K_val, D_ADDR_val, D_VAL_val, desc)

print(f"\nSWEEP 2 done in {time.time()-t_sw2:.0f}s")


# =============================================================================
# RESULTS TABLES
# =============================================================================
print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print()
print("K-SWEEP  (d_addr=d_val=64):")
print(f"  {'K':>4}  {'ACC':>7}  {'C1':>7}  {'ACC-C1':>8}")
print(f"  {'':-<4}  {'':-<7}  {'':-<7}  {'':-<8}")
for K_val in [1, 4, 8]:
    acc, c1 = sw1[K_val]
    print(f"  {K_val:>4}  {acc:.4f}   {c1:.4f}   {acc-c1:+.4f}")

print()
print("ASPECT-RATIO SWEEP  (K*d_addr~2048):")
asp_order = [(4, 512, "WIDE-SHORT  K=4  d=512"),
             (8, 256, "SQUARE      K=8  d=256"),
             (16,128, "NARROW-TALL K=16 d=128")]
print(f"  {'Config':<25}  {'ACC':>7}  {'C1':>7}  {'ACC-C1':>8}")
print(f"  {'':-<25}  {'':-<7}  {'':-<7}  {'':-<8}")
for K_val, D_ADDR_val, label in asp_order:
    acc, c1 = sw2[(K_val, D_ADDR_val)]
    print(f"  {label:<25}  {acc:.4f}   {c1:.4f}   {acc-c1:+.4f}")


# =============================================================================
# INTERPRETATION (dynamic, based on actual results)
# =============================================================================
print()
print("Interpretation:")

k1_acc, _ = sw1[1]
k4_acc, _ = sw1[4]
k8_acc, _ = sw1[8]

if k1_acc >= 0.85:
    print(f"  K-sweep: even K=1 ({k1_acc:.1%}) is sufficient; a single 64-dim slot")
    print(f"  carries enough capacity to encode key k at this task scale.")
elif k1_acc < 0.15:
    print(f"  K-sweep: hard capacity knee at K=1 ({k1_acc:.1%}) -> K=4 ({k4_acc:.1%}).")
    print(f"  One slot is too narrow to reliably encode k; K>=4 is required.")
else:
    print(f"  K-sweep: soft knee; K=1 reaches {k1_acc:.1%} vs K=4 at {k4_acc:.1%}.")
    print(f"  One slot is partially sufficient; K>=4 achieves near-ceiling accuracy.")

wide_acc, _ = sw2[(4, 512)]
sq_acc,   _ = sw2[(8, 256)]
tall_acc, _ = sw2[(16, 128)]
accs  = [wide_acc, sq_acc, tall_acc]
names = ["WIDE-SHORT", "SQUARE", "NARROW-TALL"]
best  = names[accs.index(max(accs))]
spread = max(accs) - min(accs)

if spread < 0.04:
    print(f"  Aspect-ratio: all shapes converge within {spread:.1%}; total capacity")
    print(f"  (K*d_addr) is the binding constraint, not the slot-geometry tradeoff.")
elif best == "WIDE-SHORT":
    print(f"  Aspect-ratio: WIDE-SHORT ({wide_acc:.1%}) leads; wider address dims improve")
    print(f"  key-query discriminability more than extra slots at equal total capacity.")
elif best == "NARROW-TALL":
    print(f"  Aspect-ratio: NARROW-TALL ({tall_acc:.1%}) leads; more slots give finer-")
    print(f"  grained parallel channels than a few wide slots at equal total capacity.")
else:
    print(f"  Aspect-ratio: SQUARE ({sq_acc:.1%}) is optimal; balanced geometry")
    print(f"  outperforms extremes at the same K*d_addr budget.")

# one-liner summary
print()
k_line = "  ".join(f"K={k}:{sw1[k][0]:.4f}/{sw1[k][1]:.4f}" for k in [1, 4, 8])
asp_line_parts = [
    f"WIDE:{sw2[(4,512)][0]:.4f}/{sw2[(4,512)][1]:.4f}",
    f"SQ:{sw2[(8,256)][0]:.4f}/{sw2[(8,256)][1]:.4f}",
    f"TALL:{sw2[(16,128)][0]:.4f}/{sw2[(16,128)][1]:.4f}",
]
print(f"M0-Part4-K:      {k_line}")
print(f"M0-Part4-Aspect: {'  '.join(asp_line_parts)}")
