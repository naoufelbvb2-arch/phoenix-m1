"""
M0-Part2 training runner -- two-zone workspace-board model.

Zone A encodes stream_A (key k), Zone B encodes stream_B (value v).
Cross-zone information flows via the workspace board in Round 2.

Auxiliary k-prediction loss:
  Zone A's write mechanism has no path to the loss through the board read
  gate when that gate is initially small (alpha * W_out = 0 -> zero gradient).
  An auxiliary CE loss on Zone A's slot values predicts k directly, giving
  Zone A's write gradient from step 1.  This bootstraps the board channel:
  once Zone A writes k to its slots, Zone B reads k via the board, alpha_B
  grows, and the main task accuracy climbs.

Usage:
    python run_m0_part2.py
"""

import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

from task  import generate_batch, VOCAB_SIZE, KEY_MARKER
from model import PhoenixM0P2

# ---- hyperparams ------------------------------------------------------------
N_TRAIN    = 20_000
N_EPOCHS   = 6
BATCH_SIZE = 128
LR         = 1e-3
LR_ALPHA   = 1e-6   # near-freeze alpha gates; W_out trains at full LR without attenuation
AUX_WEIGHT = 3.0    # strong bootstrap: drive Zone A to write k by epoch 1
LOG_EVERY  = 3
SEED       = 42

# ---- setup ------------------------------------------------------------------
torch.manual_seed(SEED)

A_tr, B_tr, y_tr = generate_batch(N_TRAIN, seed=0)
train_loader = DataLoader(
    TensorDataset(A_tr, B_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)

A_te, B_te, y_te = generate_batch(4096, seed=9999)
test_loader = DataLoader(
    TensorDataset(A_te, B_te, y_te), batch_size=512, shuffle=False,
)

model   = PhoenixM0P2()
loss_fn = nn.CrossEntropyLoss()

# Separate LR for scalar gates: near-freeze so alpha stays ~1.0 and W_out
# trains at full LR without being attenuated by a rapidly-collapsing gate.
alpha_params = [model.zone_a.alpha, model.zone_b.alpha]
alpha_ids    = {id(p) for p in alpha_params}
other_params = [p for p in model.parameters() if id(p) not in alpha_ids]
opt = torch.optim.Adam([
    {'params': other_params, 'lr': LR},
    {'params': alpha_params, 'lr': LR_ALPHA},
])

n_params = sum(p.numel() for p in model.parameters())
print(f"PhoenixM0P2  params={n_params:,}  "
      f"train={N_TRAIN}  epochs={N_EPOCHS}  bs={BATCH_SIZE}  lr={LR}  aux={AUX_WEIGHT}")
print()


# ---- helpers ----------------------------------------------------------------

def extract_k(xa):
    """Read k token from stream_A: position after KEY_MARKER."""
    km    = (xa == KEY_MARKER)               # (B, T) bool
    kpos  = km.float().argmax(dim=1)         # (B,) KEY_MARKER positions
    k_idx = torch.clamp(kpos + 1, max=xa.size(1) - 1)
    return xa[torch.arange(xa.size(0)), k_idx]  # (B,) k values in [0,63]


def eval_acc():
    model.eval()
    correct = 0; k_correct = 0
    with torch.no_grad():
        for xa, xb, y in test_loader:
            logits, aux_logits = model(xa, xb)
            correct   += (logits.argmax(1) == y).sum().item()
            k_correct += (aux_logits.argmax(1) == extract_k(xa)).sum().item()
    model.train()
    return correct / len(y_te), k_correct / len(y_te)


# ---- training ---------------------------------------------------------------
t0 = time.time()

for ep in range(1, N_EPOCHS + 1):
    model.train()
    ep_loss = 0.0; n_batches = 0

    for xa, xb, y in train_loader:
        opt.zero_grad()
        logits, aux_logits = model(xa, xb)

        main_loss = loss_fn(logits, y)
        k_labels  = extract_k(xa)
        aux_loss  = loss_fn(aux_logits, k_labels)
        loss      = main_loss + AUX_WEIGHT * aux_loss

        loss.backward()
        opt.step()
        ep_loss += main_loss.item()
        n_batches += 1

    avg_loss = ep_loss / n_batches
    if avg_loss != avg_loss:
        print(f"ep {ep:3d}: NaN loss -- aborting")
        break

    if ep % LOG_EVERY == 0 or ep == 1:
        acc, k_acc = eval_acc()
        alpha_a = model.zone_a.alpha.item()
        alpha_b = model.zone_b.alpha.item()
        elapsed = time.time() - t0
        print(f"ep {ep:3d}: loss={avg_loss:.4f}  acc={acc:.4f}  k_acc={k_acc:.4f}  "
              f"gate_A={alpha_a:+.4f}  gate_B={alpha_b:+.4f}  t={elapsed:.1f}s")
        if acc <= 0.025 and alpha_b < 0.1:
            print("         [WARNING] channel silence: gate collapsing toward 0")

# ---- final evaluation -------------------------------------------------------
final_acc, final_k_acc = eval_acc()
alpha_a   = model.zone_a.alpha.item()
alpha_b   = model.zone_b.alpha.item()
elapsed   = time.time() - t0

if final_acc >= 0.90:
    verdict = "PASS"
elif final_acc <= 0.025:
    verdict = "FAIL"
else:
    verdict = "PARTIAL"

gate_note = ("gate grew above init=1" if alpha_b > 1.2 else
             ("stable near init=1" if alpha_b > 0.5 else
              "collapsed -- channel silent"))

print()
print("=" * 60)
print(f"  Final accuracy  : {final_acc:.4f}")
print(f"  Zone A gate     : {alpha_a:+.6f}")
print(f"  Zone B gate     : {alpha_b:+.6f}  ({gate_note})")
print(f"  Elapsed         : {elapsed:.1f}s / ~180s budget")
if verdict == "FAIL":
    print("  DIAGNOSTIC: gate near zero; cross-zone channel silent.")
print("=" * 60)
print(f"M0-Part2: ACC={final_acc:.4f} READ_GATE={alpha_b:.6f} VERDICT={verdict}")
