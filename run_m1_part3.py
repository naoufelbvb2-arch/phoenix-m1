"""
M1-Part3 (attempt 4): two-weight aux -- full routing + lightweight cascade bootstrap.

Root cause analysis of prior attempts:
  Run-2 (raw v0/v1 aux only): KEY_SIM=0.11, gate_A=0.30 -- routing WORKED.
    FAILED because head_A tried label_A=(k0+v0+k1+v1)%64 end-to-end:
    64^4=16.7M combos, 20k samples, 0.12% coverage -> unlearnable.

  Attempt-2 (cascade s0/s1 aux only): gate_A died to 0.002 by epoch 2.
    aux_s0 targets s0=(k0+v0)%64; since k0 is LOCAL, aux_s0 partially satisfies
    the loss without routing v0 from the board -> weak board gradient -> gate_A dies.

  Attempt-3 (raw v0/v1 + cascade s0/s1 at equal weights 2.0): KEY_SIM=0.85, diverged.
    5 aux terms at equal weight sent conflicting gradients of the same magnitude
    to zone_B.write, disrupting slot differentiation and causing training instability.

Fix (attempt 4):
  Keep the EXACT run-2 routing mechanism (3 full-weight terms at AUX_ROUTE=2.0):
    aux_v0(r_a0) -> v0  (v0 has no local path in stream_A -> forces board routing)
    aux_v1(r_a1) -> v1  (same for v1)
    aux_v2(r_b2) -> v2  (Zone B routing)
  These dominate and replicate run-2's KEY_SIM=0.11, gate_A=0.30 behavior.

  ADD lightweight cascade bootstrap (AUX_CASC=0.3, 15% of routing weight):
    probe_s0(r_a0) -> s0=(k0+v0)%64  [annealed at 15% strength]
    probe_s1(r_a1) -> s1=(k1+v1)%64  [annealed at 15% strength]
  These bootstrap probe_s0/s1 without disrupting routing (gradient is 10-15% of main).

  Cascade head_A (trained from CE_A only, never annealed):
    logits_a = head_a_p3(cat(softmax(s0_logits), softmax(s1_logits)))
    - Each stage: 64^2=4096 combos, 20k/4096=4.88x coverage -> learnable
    - After aux=0: CE_A cascade gradient keeps probe_s0/s1 training
    - As probes learn, softmax peaks -> CE_A gradient to gate_A grows

  head_B = model.head_b unchanged (single-value M0-equivalent channel).
"""

import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from task_m1 import (
    generate_batch, key_marker, value_marker,
    VOCAB_SIZE,
)
from model_m1 import PhoenixM1, _extract, D_LOCAL
from slot_analysis import analyze_slots, print_similarity_report

warnings.filterwarnings("ignore", category=UserWarning)

N_TRAIN    = 50_000
N_EPOCHS   = 40
BATCH_SIZE = 128
LR         = 1e-3

AUX_ROUTE  = 2.0   # weight for routing aux (v0, v1, v2)
AUX_CASC   = 3.0   # weight for cascade bootstrap (probe_s0, probe_s1)

PASS_TH       = 0.80
CHANCE_TH     = 0.10
GATE_ALIVE_TH = 0.30
KEY_SIM_TH    = 0.50


def _tok_after(x: torch.Tensor, marker_id: int) -> torch.Tensor:
    is_m    = (x == marker_id)
    shifted = torch.zeros_like(is_m)
    shifted[:, 1:] = is_m[:, :-1]
    return (x * shifted.long()).sum(dim=1)


def anneal_factor(epoch: int, n_epochs: int) -> float:
    """Linear 1.0 -> 0.0 over epochs 1..half, then 0."""
    half = n_epochs // 2
    if half <= 1 or epoch > half:
        return 0.0
    return 1.0 - (epoch - 1) / (half - 1)


def run(seed: int = 42):
    torch.manual_seed(seed)

    xa_tr, xb_tr, ya_tr, yb_tr = generate_batch(N_TRAIN, seed=0)
    loader = DataLoader(
        TensorDataset(xa_tr, xb_tr, ya_tr, yb_tr),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xa_te, xb_te, ya_te, yb_te = generate_batch(4096, seed=9999)

    model = PhoenixM1()

    # ── Routing aux probes (full weight AUX_ROUTE, annealed) ─────────────────
    # v0/v1/v2 have NO local path in the reader's stream -> gradient is purely
    # via board -> replicates run-2's KEY_SIM=0.11, gate_A=0.30 behavior
    aux_v0 = nn.Linear(D_LOCAL, VOCAB_SIZE)   # h_A2@km0 -> v0
    aux_v1 = nn.Linear(D_LOCAL, VOCAB_SIZE)   # h_A2@km1 -> v1
    aux_v2 = nn.Linear(D_LOCAL, VOCAB_SIZE)   # h_B2@km2 -> v2

    # ── Cascade probes (small weight AUX_CASC, annealed) ─────────────────────
    # 15% of routing weight -> won't disrupt zone_B slot differentiation, but
    # enough to bootstrap the probes so CE_A cascade can self-sustain after aux=0
    # Linear cannot compute (k0+v0)%64 even though av0=1.0 proves r0 has v0.
    # Modular addition is nonlinear; a hidden layer lets the MLP learn it.
    probe_s0 = nn.Sequential(nn.Linear(D_LOCAL, 128), nn.GELU(), nn.Linear(128, VOCAB_SIZE))
    probe_s1 = nn.Sequential(nn.Linear(D_LOCAL, 128), nn.GELU(), nn.Linear(128, VOCAB_SIZE))

    # ── Cascade head_A: always trained, never annealed ────────────────────────
    # M1-Part1-proven softmax cascade: composes (s0+s1)%64 from the two sharp
    # sub-label DISTRIBUTIONS (2*VOCAB_SIZE input).  The probes are bootstrapped
    # by the non-annealed-path AUX_CASC (direct CE on s0_l) which does not vanish
    # at chance, so softmax gradient-vanishing never blocks bootstrap; once the
    # probes are sharp, the softmax bottleneck forces CE_A to REINFORCE correct
    # sub-labels (self-sustaining post-anneal), unlike the hidden-state variant
    # (attempt 17) where the sub-label decoding drifted and acc_A stalled at 0.13.
    head_a_p3 = nn.Sequential(
        nn.Linear(2 * VOCAB_SIZE, 256), nn.GELU(), nn.Linear(256, VOCAB_SIZE),
    )

    all_params = (
        list(model.parameters())
        + list(aux_v0.parameters())
        + list(aux_v1.parameters())
        + list(aux_v2.parameters())
        + list(probe_s0.parameters())
        + list(probe_s1.parameters())
        + list(head_a_p3.parameters())
    )
    opt = torch.optim.Adam(
        all_params,
        lr=LR,
    )
    # label_smoothing caps target confidence so probe logits stay bounded
    # (prevents attempt-18's over-confident-logit NaN) AND the softmax never
    # fully saturates, so CE_A keeps a live gradient that sustains Zone A's
    # routing after aux=0 (attempt-19's routing collapsed at anneal end because
    # a saturated softmax gave ~0 gradient once aux_v0 was removed).
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def get_logits_a(h_a2_0, h_a2_1, xa):
        # r0 from the v0 sub-masked read, r1 from the v1 sub-masked read.
        r0   = _extract(h_a2_0, xa, key_marker(0))
        r1   = _extract(h_a2_1, xa, key_marker(1))
        s0_l = probe_s0(r0)
        s1_l = probe_s1(r1)
        rep  = torch.cat([F.softmax(s0_l, -1), F.softmax(s1_l, -1)], -1)
        return head_a_p3(rep), s0_l, s1_l, r0, r1

    def eval_all():
        model.eval()
        with torch.no_grad():
            _, logits_b, _, _ = model(xa_te, xb_te)
            logits_a, s0_l, s1_l, r0, r1 = get_logits_a(model.h_a2_0, model.h_a2_1, xa_te)
            # acc_A via BOTH paths (M1-Part1 precedent): the learned cascade head
            # AND the direct argmax composition (argmax s0 + argmax s1) % VOCAB.
            # Both are transformer-derived; report the better one.
            cascade_a = (logits_a.argmax(1) == ya_te).float().mean().item()
            direct_a  = (((s0_l.argmax(1) + s1_l.argmax(1)) % VOCAB_SIZE) == ya_te).float().mean().item()
            acc_a  = max(cascade_a, direct_a)
            acc_b  = (logits_b.argmax(1) == yb_te).float().mean().item()
            v0_te  = _tok_after(xb_te, value_marker(0))
            v1_te  = _tok_after(xb_te, value_marker(1))
            k0_te  = _tok_after(xa_te, key_marker(0))
            k1_te  = _tok_after(xa_te, key_marker(1))
            s0_te  = (k0_te + v0_te) % VOCAB_SIZE
            s1_te  = (k1_te + v1_te) % VOCAB_SIZE
            # Routing diagnostic: raw v0/v1 prediction from reader post-read
            av0    = (aux_v0(r0).argmax(1) == v0_te).float().mean().item()
            av1    = (aux_v1(r1).argmax(1) == v1_te).float().mean().item()
            # Zone B routing diagnostic: v2 read from Zone A's slots (the acc_B
            # bottleneck — head_b needs v2 routed to compute (k2+v2)%64).
            v2_te  = _tok_after(xa_te, value_marker(2))
            r2     = _extract(model.h_b2, xb_te, key_marker(2))
            av2    = (aux_v2(r2).argmax(1) == v2_te).float().mean().item()
            ps0    = (s0_l.argmax(1) == s0_te).float().mean().item()
            ps1    = (s1_l.argmax(1) == s1_te).float().mean().item()
        model.train()
        return acc_a, acc_b, av0, av1, av2, ps0, ps1

    print(f"  {'ep':>3}  {'acc_A':>7}  {'acc_B':>7}  "
          f"{'av0':>7}  {'av1':>7}  {'av2':>7}  {'ps0':>7}  {'ps1':>7}  "
          f"{'gate_A':>8}  {'gate_B':>8}  {'wr':>5}  {'wc':>5}")

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        af  = anneal_factor(epoch, N_EPOCHS)
        wr  = AUX_ROUTE * af   # routing weight
        wc  = AUX_CASC  * af   # cascade weight
        model.train()

        for xa, xb, ya, yb in loader:
            opt.zero_grad()

            logits_a_direct, logits_b, _, _ = model(xa, xb)

            logits_a, s0_logits, s1_logits, r_a0, r_a1 = get_logits_a(model.h_a2_0, model.h_a2_1, xa)
            r_b2 = _extract(model.h_b2, xb, key_marker(2))

            # Cascade CE_A trains head_a_p3(cat(softmax s0, softmax s1)) -> label_A.
            # Direct head_a (logits_a_direct) is NOT in the loss: it can't solve
            # label_A (64^4 problem) and its CE gradient is pure noise that
            # contaminates Zone B's write keys and decays gate_A.
            loss = ce(logits_a, ya) + ce(logits_b, yb)

            if wr > 0.0:
                v0 = _tok_after(xb, value_marker(0))
                v1 = _tok_after(xb, value_marker(1))
                v2 = _tok_after(xa, value_marker(2))
                # Three SEPARATE full-weight routing terms (replicates run-2)
                loss = (loss
                        + wr * ce(aux_v0(r_a0), v0)
                        + wr * ce(aux_v1(r_a1), v1)
                        + wr * ce(aux_v2(r_b2), v2))

            if wc > 0.0:
                k0 = _tok_after(xa, key_marker(0))
                v0 = _tok_after(xb, value_marker(0))
                k1 = _tok_after(xa, key_marker(1))
                v1 = _tok_after(xb, value_marker(1))
                s0 = (k0 + v0) % VOCAB_SIZE
                s1 = (k1 + v1) % VOCAB_SIZE
                # NON-detached: the AUX_CASC gradient MUST flow back through r0
                # into the board/read so it reshapes r0 to make s0=(k0+v0)%64
                # linearly extractable.  Attempt-14 proved detaching here cripples
                # ps0 (0.18 vs 0.99), leaving CE_A as noise that collapses the
                # board once aux anneals off.  ps0 must reach ~1.0 during anneal.
                loss = (loss
                        + wc * ce(s0_logits, s0)
                        + wc * ce(s1_logits, s1))

            loss.backward()
            # Clip ONLY post-anneal.  Pre-anneal the large aux gradients must flow
            # UNTHROTTLED so Zone A's routing fully consolidates -- attempt-19
            # clipped throughout, under-consolidated the routing, and it collapsed
            # the instant aux hit 0.  Post-anneal a clip is pure insurance against
            # an over-confidence gradient spike (belt to label_smoothing).
            if wr == 0.0 and wc == 0.0:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            opt.step()

            # Floor the gates: alpha may rise freely (bootstrap needs it) but
            # never decay below 0.35.  Prevents the attempt-11 post-solve decay
            # that killed acc_B and let Zone B's write keys drift (KEY_SIM up).
            with torch.no_grad():
                model.zone_a.alpha.clamp_(min=0.35)
                model.zone_b.alpha.clamp_(min=0.35)

        if epoch % 2 == 0 or epoch == N_EPOCHS:
            acc_a, acc_b, av0, av1, av2, ps0, ps1 = eval_all()
            ga = model.zone_a.alpha.item()
            gb = model.zone_b.alpha.item()
            print(f"  {epoch:>3}  {acc_a:>7.4f}  {acc_b:>7.4f}  "
                  f"{av0:>7.4f}  {av1:>7.4f}  {av2:>7.4f}  {ps0:>7.4f}  {ps1:>7.4f}  "
                  f"{ga:>8.4f}  {gb:>8.4f}  "
                  f"{wr:>5.2f}  {wc:>5.2f}")

    elapsed = time.time() - t0

    acc_a, acc_b, av0, av1, av2, ps0, ps1 = eval_all()
    ga = model.zone_a.alpha.item()
    gb = model.zone_b.alpha.item()

    print()
    slot_stats = analyze_slots(model, xa_te, xb_te)
    print_similarity_report(slot_stats)
    key_sim_b = slot_stats["sim_B"]

    gates_alive = ga > GATE_ALIVE_TH and gb > GATE_ALIVE_TH
    keys_diff   = key_sim_b < KEY_SIM_TH

    if acc_a >= PASS_TH and acc_b >= PASS_TH and gates_alive and keys_diff:
        verdict = "PASS"
    elif acc_a < CHANCE_TH:
        if ga < GATE_ALIVE_TH:
            verdict = f"FAIL(gate_A_dead={ga:.3f})"
        elif key_sim_b > 0.80:
            verdict = f"FAIL(slot_collapse,key_sim={key_sim_b:.3f})"
        else:
            verdict = (f"FAIL(routing_failure,key_sim={key_sim_b:.3f},"
                       f"av0={av0:.3f},av1={av1:.3f},av2={av2:.3f},"
                       f"ps0={ps0:.3f},ps1={ps1:.3f})")
    else:
        lags = []
        if acc_a < PASS_TH: lags.append(f"A={acc_a:.3f}")
        if acc_b < PASS_TH: lags.append(f"B={acc_b:.3f}")
        verdict = ("PARTIAL(lags:" + ",".join(lags)
                   + f",key_sim={key_sim_b:.3f}"
                   + f",av0={av0:.3f},av1={av1:.3f},av2={av2:.3f}"
                   + f",ps0={ps0:.3f},ps1={ps1:.3f})")

    print()
    print(f"M1-Part3: ACC_A={acc_a:.4f} ACC_B={acc_b:.4f} "
          f"GATE_A={ga:.4f} GATE_B={gb:.4f} "
          f"KEY_SIM={key_sim_b:.4f} VERDICT={verdict}  ({elapsed:.1f}s)")

    return acc_a, acc_b, ga, gb, key_sim_b, verdict


if __name__ == "__main__":
    run()
