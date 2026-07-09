"""
comp_controls.py -- two single-field controls + the COMP-Part1 verdict.

ret-only   : sees ret_type but NOT param_type -> must fail toward chance (the
             relation needs both fields).
param-only : sees param_type but NOT ret_type -> must fail toward chance.

If either single-field control succeeds, the task is not genuinely relational
(a single field leaks the category relation).  Both must sit near 50% and
clearly below the dense ceiling.
"""

import time

from comp_dense_ceiling import train_eval

CEIL_TH  = 0.95
CTRL_TH  = 0.60          # controls must stay <= ~60% (chance 50%)
GAP      = 1.5           # ceiling must be >= GAP x each control
TIME_TH  = 180.0
CHANCE   = 0.50


def main():
    t0 = time.time()
    ceiling   = train_eval("full")
    ctrl_ret  = train_eval("retonly")
    ctrl_par  = train_eval("paramonly")
    elapsed = time.time() - t0

    passed = (
        ceiling >= CEIL_TH
        and ctrl_ret <= CTRL_TH
        and ctrl_par <= CTRL_TH
        and ceiling >= GAP * ctrl_ret
        and ceiling >= GAP * ctrl_par
        and elapsed < TIME_TH
    )
    verdict = "PASS" if passed else "FAIL"

    print(f"chance={CHANCE}   ceiling_th={CEIL_TH}   ctrl_th={CTRL_TH}   gap={GAP}x   time_th={TIME_TH}s")
    print(f"COMP-Part1: CEILING={ceiling:.4f} CTRL_RET={ctrl_ret:.4f} "
          f"CTRL_PARAM={ctrl_par:.4f} VERDICT={verdict}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
