"""
M1-Part1 entry point.

Runs the dense ceiling and both decomposability controls, prints:
    M1-Part1: CEILING_A=<acc> CEILING_B=<acc> CTRL_A=<acc> CTRL_B=<acc>
              METHOD=<extraction description> VERDICT=<PASS/FAIL>
"""

import time

from task_m1                     import VOCAB_SIZE
from dense_ceiling_m1             import run as run_ceiling
from decomposability_control_m1   import run as run_control

RANDOM_CHANCE = 1.0 / VOCAB_SIZE        # 1/64 ~= 0.0156
CTRL_MAX      = RANDOM_CHANCE + 0.05    # ~= 0.0656
CEIL_MIN      = 0.90
TIME_BUDGET   = 240.0                   # seconds (~4 min)

METHOD = ("shared 2L enc + branch_A 1L (label_A only); "
          "_extract(h,x) reads transformer h states only; "
          "aux cascade for label_A")

# ---- run all three experiments -----------------------------------------------

t_start = time.time()

print(">> Training ceiling   (stream_A + stream_B, both labels) ...")
ceil_acc_A, ceil_acc_B, ceil_time = run_ceiling()

print(">> Training control A (stream_A only, label_A)            ...")
ctrl_acc_A, ctrl_time_A = run_control("A")

print(">> Training control B (stream_B only, label_B)            ...")
ctrl_acc_B, ctrl_time_B = run_control("B")

total = time.time() - t_start

# ---- verdict ------------------------------------------------------------------

ceil_ok_A = ceil_acc_A >= CEIL_MIN
ceil_ok_B = ceil_acc_B >= CEIL_MIN
ctrl_ok_A = ctrl_acc_A <= CTRL_MAX
ctrl_ok_B = ctrl_acc_B <= CTRL_MAX
time_ok   = total < TIME_BUDGET
verdict   = "PASS" if (ceil_ok_A and ceil_ok_B and ctrl_ok_A and ctrl_ok_B and time_ok) else "FAIL"

print()
print("=" * 70)
print(f"  [CEILING_A] acc={ceil_acc_A:.4f}   target >= {CEIL_MIN:.4f}   -> {'PASS' if ceil_ok_A else 'FAIL'}")
print(f"  [CEILING_B] acc={ceil_acc_B:.4f}   target >= {CEIL_MIN:.4f}   -> {'PASS' if ceil_ok_B else 'FAIL'}")
print(f"  [CTRL_A   ] acc={ctrl_acc_A:.4f}   target <= {CTRL_MAX:.4f}   -> {'PASS' if ctrl_ok_A else 'FAIL'}")
print(f"  [CTRL_B   ] acc={ctrl_acc_B:.4f}   target <= {CTRL_MAX:.4f}   -> {'PASS' if ctrl_ok_B else 'FAIL'}")
print(f"  [TIMING   ] {total:.1f}s / {TIME_BUDGET:.0f}s              -> {'PASS' if time_ok else 'FAIL'}")
print("=" * 70)
print(f"M1-Part1: CEILING_A={ceil_acc_A:.4f} CEILING_B={ceil_acc_B:.4f} "
      f"CTRL_A={ctrl_acc_A:.4f} CTRL_B={ctrl_acc_B:.4f} "
      f"METHOD={METHOD} VERDICT={verdict}")
