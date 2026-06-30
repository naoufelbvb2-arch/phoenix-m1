"""
M0-Part1 entry point.

Runs both experiments and prints a single verdict line:
    M0-Part1: CEILING=<acc> CONTROL=<acc> VERDICT=<PASS/FAIL>

Usage:
    python run_m0p1.py
"""

import time

from task              import VOCAB_SIZE
from dense_ceiling     import run as run_ceiling
from half_info_control import run as run_control

RANDOM_CHANCE = 1.0 / VOCAB_SIZE   # 1/64 ~= 0.0156
CTRL_MAX      = RANDOM_CHANCE + 0.05   # ~= 0.0656
CEIL_MIN      = 0.95
TIME_BUDGET   = 120.0               # seconds

# ---- run both experiments ---------------------------------------------------

t_start = time.time()

print(">> Training ceiling  (stream_A + stream_B) ...")
ceil_acc, ceil_time = run_ceiling()

print(">> Training control  (stream_A only)       ...")
ctrl_acc, ctrl_time = run_control()

total = time.time() - t_start

# ---- verdict ----------------------------------------------------------------

ceil_ok = ceil_acc >= CEIL_MIN
ctrl_ok = ctrl_acc <= CTRL_MAX
time_ok = total    <  TIME_BUDGET
verdict = "PASS" if (ceil_ok and ctrl_ok and time_ok) else "FAIL"

print()
print("=" * 62)
print(f"  [CEILING] acc={ceil_acc:.4f}   target >= {CEIL_MIN:.4f}       -> {'PASS' if ceil_ok else 'FAIL'}")
print(f"  [CONTROL] acc={ctrl_acc:.4f}   target <= {CTRL_MAX:.4f}       -> {'PASS' if ctrl_ok else 'FAIL'}")
print(f"  [TIMING ] {total:.1f}s / {TIME_BUDGET:.0f}s                -> {'PASS' if time_ok else 'FAIL'}")
print("=" * 62)
print(f"M0-Part1: CEILING={ceil_acc:.4f} CONTROL={ctrl_acc:.4f} VERDICT={verdict}")
