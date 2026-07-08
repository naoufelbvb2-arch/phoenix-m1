"""
pcr_controls.py -- two half-info controls + the PCR-Part1 acceptance verdict.

Proposer-only : sees (s0, ops) but NOT T   -> must FAIL (T hidden -> r unknowable)
Checker-only  : sees T but NOT (s0, ops)   -> must FAIL (x hidden -> r unknowable)

Both must sit near chance (1/M = 6.25%) and WELL below the dense ceiling.  Runs
the same PCRModel on all three input modes and prints the final verdict line.
"""

import time

from pcr_dense_ceiling import train_eval
from pcr_task import M

CEIL_TH  = 0.95
CTRL_TH  = 0.12               # controls must stay <= ~12%
MARGIN   = 6.0                # controls must be >= 6x below the ceiling
TIME_TH  = 180.0             # CPU budget: < 3 min for all three runs
CHANCE   = 1.0 / M            # 6.25%


def main():
    t0 = time.time()
    ceiling    = train_eval("full")
    ctrl_prop  = train_eval("prop")
    ctrl_check = train_eval("check")
    elapsed = time.time() - t0

    passed = (
        ceiling >= CEIL_TH
        and ctrl_prop  <= CTRL_TH
        and ctrl_check <= CTRL_TH
        and ceiling >= MARGIN * ctrl_prop
        and ceiling >= MARGIN * ctrl_check
        and elapsed < TIME_TH
    )
    verdict = "PASS" if passed else "FAIL"

    print(f"chance=1/M={CHANCE:.4f}   ceiling_th={CEIL_TH}   ctrl_th={CTRL_TH}   "
          f"margin={MARGIN}x   time_th={TIME_TH}s")
    print(f"PCR-Part1: CEILING={ceiling:.4f} CTRL_PROP={ctrl_prop:.4f} "
          f"CTRL_CHECK={ctrl_check:.4f} VERDICT={verdict}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
