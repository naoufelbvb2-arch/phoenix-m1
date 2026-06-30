"""
M0-Part3 -- Lesion test + AUX-weight sensitivity sweep.

SECTION 1  (produces the verdict):
  Train ONE model (Part-2 config, AUX=3.0), then at INFERENCE measure:
    C0  full board visible       -> expect ~91%  (Part-2 baseline)
    C1  hide Zone A slots 0-7   -> expect <=2.5% (k info cut off)
    C2  hide Zone B slots 8-15  -> expect >=85%  (B doesn't need its own slots)

  PASS only if C0>=90% AND C1<=2.5% AND C2>=85%.

SECTION 2  (diagnostic, no hard pass/fail):
  Retrain at AUX in {3.0, 1.0, 0.0}, report (held-out acc, C1-lesion acc).
  Determines whether the cross-zone protocol is emergent (AUX=0 works)
  or requires the auxiliary scaffold to bootstrap Zone A's write.

Runtime: ~8-10 min on CPU.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from task   import generate_batch, KEY_MARKER
from model  import PhoenixM0P2
from lesion import mask_c1, mask_c2, eval_acc

# ---- Part-2 hyperparams (must match exactly) --------------------------------
N_TRAIN    = 20_000
N_EPOCHS   = 6
BATCH_SIZE = 128
LR         = 1e-3
LR_ALPHA   = 1e-6    # near-freeze alpha gates
AUX_WEIGHT = 3.0     # Section 1 / Section 2 AUX=3.0 baseline
SEED       = 42

# ---- shared test set -------------------------------------------------------
torch.manual_seed(SEED)
A_te, B_te, y_te = generate_batch(4096, seed=9999)
test_loader = DataLoader(
    TensorDataset(A_te, B_te, y_te), batch_size=512, shuffle=False,
)

# ---- training helper -------------------------------------------------------

def extract_k(xa):
    """Token immediately after KEY_MARKER in stream_A."""
    km    = (xa == KEY_MARKER)
    kpos  = km.float().argmax(dim=1)
    k_idx = torch.clamp(kpos + 1, max=xa.size(1) - 1)
    return xa[torch.arange(xa.size(0)), k_idx]


def train_model(aux_weight, label=""):
    """
    Train a fresh PhoenixM0P2 for N_EPOCHS using Part-2 config.
    aux_weight=0.0 disables the auxiliary k-prediction loss entirely.
    Returns trained model.
    """
    torch.manual_seed(SEED)
    t0 = time.time()

    A_tr, B_tr, y_tr = generate_batch(N_TRAIN, seed=0)
    loader = DataLoader(
        TensorDataset(A_tr, B_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model   = PhoenixM0P2()
    loss_fn = nn.CrossEntropyLoss()

    alpha_params = [model.zone_a.alpha, model.zone_b.alpha]
    alpha_ids    = {id(p) for p in alpha_params}
    other_params = [p for p in model.parameters() if id(p) not in alpha_ids]
    opt = torch.optim.Adam([
        {'params': other_params, 'lr': LR},
        {'params': alpha_params, 'lr': LR_ALPHA},
    ])

    for ep in range(1, N_EPOCHS + 1):
        model.train()
        ep_loss = 0.0; n = 0
        for xa, xb, y in loader:
            opt.zero_grad()
            logits, aux_logits = model(xa, xb)
            main_loss = loss_fn(logits, y)
            if aux_weight > 0.0:
                loss = main_loss + aux_weight * loss_fn(aux_logits, extract_k(xa))
            else:
                loss = main_loss
            loss.backward()
            opt.step()
            ep_loss += main_loss.item(); n += 1

        if ep == 1 or ep % 2 == 0:
            print(f"    ep {ep:2d}  main_loss={ep_loss/n:.4f}  t={time.time()-t0:.0f}s")

    print(f"  [{label}] training done in {time.time()-t0:.0f}s")
    return model


# ============================================================
# SECTION 1: Cross-zone lesion test
# ============================================================
print("=" * 60)
print("SECTION 1: Cross-zone lesion test")
print("=" * 60)
print(f"Training Section-1 model (AUX={AUX_WEIGHT})...")
t_s1 = time.time()
model_s1 = train_model(aux_weight=AUX_WEIGHT, label="S1/AUX3.0")

print("Running C0 / C1 / C2 inference...")
c0 = eval_acc(model_s1, test_loader, slot_mask=None)
c1 = eval_acc(model_s1, test_loader, slot_mask=mask_c1())
c2 = eval_acc(model_s1, test_loader, slot_mask=mask_c2())

ok_c0 = c0 >= 0.90
ok_c1 = c1 <= 0.025
ok_c2 = c2 >= 0.85
verdict_s1 = "PASS" if (ok_c0 and ok_c1 and ok_c2) else "FAIL"

print()
print(f"  C0  full board          : {c0:.4f}  {'>=90% OK' if ok_c0 else 'FAIL need >=90%'}")
print(f"  C1  hide Zone-A slots   : {c1:.4f}  {'<=2.5% OK' if ok_c1 else 'FAIL need <=2.5%'}")
print(f"  C2  hide Zone-B slots   : {c2:.4f}  {'>=85% OK' if ok_c2 else 'FAIL need >=85%'}")
print()
if ok_c0 and ok_c1 and ok_c2:
    print("  Interpretation: C1 collapses to chance; C2 stays near baseline.")
    print("  Zone B depends specifically on Zone A's slots to obtain key k.")
    print("  The board channel is genuinely cross-zone, not a board-statistics artifact.")
elif not ok_c1:
    print("  WARNING: C1 did not collapse -- Zone B may be reading k from Zone B's own slots.")
elif not ok_c2:
    print("  WARNING: C2 too low -- Zone B may have learned to depend on its own slots.")
elif not ok_c0:
    print("  WARNING: Baseline (C0) too low -- model may be undertrained.")

print()
print(f"  Section 1 elapsed: {time.time()-t_s1:.0f}s")
print()
print(f"M0-Part3-Lesion: C0={c0:.4f} C1={c1:.4f} C2={c2:.4f} VERDICT={verdict_s1}")


# ============================================================
# SECTION 2: AUX-weight sweep
# ============================================================
print()
print("=" * 60)
print("SECTION 2: AUX-weight sensitivity sweep")
print("=" * 60)
t_s2 = time.time()

sweep = {}   # aux_weight -> (acc, c1_acc)

# AUX=3.0: reuse Section-1 model (same config, saves ~168s)
sweep[3.0] = (c0, c1)
print(f"  AUX=3.0 -> reusing Section-1 model  acc={c0:.4f}  C1={c1:.4f}")

for aux_w in [1.0, 0.0]:
    print()
    print(f"  Training AUX={aux_w}...")
    m = train_model(aux_weight=aux_w, label=f"AUX={aux_w}")
    acc   = eval_acc(m, test_loader, slot_mask=None)
    c1acc = eval_acc(m, test_loader, slot_mask=mask_c1())
    sweep[aux_w] = (acc, c1acc)
    print(f"  AUX={aux_w} -> acc={acc:.4f}  C1={c1acc:.4f}")

print()
print("  AUX sweep results:")
print("  AUX   | acc    | C1-lesion | channel")
print("  ------|--------|-----------|--------")
for aux_w in [3.0, 1.0, 0.0]:
    acc_w, c1_w = sweep[aux_w]
    ch = "PASS" if (acc_w >= 0.90 and c1_w <= 0.025) else (
         "PART" if acc_w >= 0.10 else "FAIL")
    print(f"  {aux_w:5.1f} | {acc_w:.4f} | {c1_w:.6f}  | {ch}")

print()
acc0, c10 = sweep[0.0]
if acc0 >= 0.90 and c10 <= 0.025:
    print("  Interpretation: AUX=0.0 achieves high accuracy with C1 collapse.")
    print("  Cross-zone protocol is EMERGENT; the auxiliary loss is optional.")
elif acc0 >= 0.10:
    print(f"  Interpretation: AUX=0.0 reaches partial accuracy ({acc0:.1%}).")
    print("  The auxiliary loss significantly accelerates convergence (scaffold).")
    print("  Emergent learning may succeed with more epochs or higher LR for writes.")
else:
    print(f"  Interpretation: AUX=0.0 fails ({acc0:.1%} accuracy in {N_EPOCHS} epochs).")
    print("  Zone A's write mechanism cannot bootstrap from task loss alone in this budget.")
    print("  The auxiliary k-prediction loss is a REQUIRED SCAFFOLD at this training scale.")

print()
print(f"  Section 2 elapsed: {time.time()-t_s2:.0f}s")

r = sweep
print()
print(f"M0-Part3-AUX: "
      f"AUX3.0={r[3.0][0]:.4f}/{r[3.0][1]:.4f} "
      f"AUX1.0={r[1.0][0]:.4f}/{r[1.0][1]:.4f} "
      f"AUX0.0={r[0.0][0]:.4f}/{r[0.0][1]:.4f}")
