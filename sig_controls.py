"""
sig_controls.py -- two half-info controls + the SIG-Part1 acceptance verdict.

No-query   : signature + probe, but NOT which field (query MASKed) -> can't know
             which field to compare -> must fail toward chance (50%).
No-signature: query + probe, but NOT the signature -> nothing to compare against
             -> must fail toward chance.

Both must sit near 50% (<= ~60%) and clearly below the dense ceiling.  Binary
label -> chance = 50%.
"""

import time

from sig_dense_ceiling import train_eval

CEIL_TH  = 0.95
CTRL_TH  = 0.60          # controls must stay <= ~60% (chance 50%)
TIME_TH  = 180.0         # CPU < 3 min for all three runs
CHANCE   = 0.50


def main():
    t0 = time.time()
    ceiling  = train_eval("full")
    ctrl_nq  = train_eval("noquery")
    ctrl_ns  = train_eval("nosig")
    elapsed = time.time() - t0

    passed = (
        ceiling >= CEIL_TH
        and ctrl_nq <= CTRL_TH
        and ctrl_ns <= CTRL_TH
        and elapsed < TIME_TH
    )
    verdict = "PASS" if passed else "FAIL"

    print(f"chance={CHANCE}   ceiling_th={CEIL_TH}   ctrl_th={CTRL_TH}   time_th={TIME_TH}s")
    print(f"SIG-Part1: CEILING={ceiling:.4f} CTRL_NOQUERY={ctrl_nq:.4f} "
          f"CTRL_NOSIG={ctrl_ns:.4f} VERDICT={verdict}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
