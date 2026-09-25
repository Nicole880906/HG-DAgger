#!/usr/bin/env python3
"""Audit a deploy session's .npz log: was the handover safe?

Answers the questions you actually have after a run, in order of how much they
matter:

1. Was anything commanded while this node was supposed to be silent?
2. How big was the first command after each handback, in position and angle?
3. Did any command under policy control exceed the clamps?
4. How far did the surgeon move the arm -- i.e. how hard was each handback?

A note on measuring step sizes, because getting it wrong produces a frightening
number that means nothing.  The log has one row per control cycle across every
phase, and the arm keeps moving while this node is silent (teleop is driving).
Differencing a phase-filtered array therefore straddles those gaps and reports
the surgeon's whole excursion as if it were one commanded step -- 60 mm against
a 5 mm clamp, in a run where the clamp held perfectly.  Only pairs that are
**adjacent in cycle number and both under policy control** are commanded steps.

Usage::

    python tools/check_handover.py deploy/logs/intervention_run.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

POS = slice(0, 3)
ROT6D = slice(3, 9)
ARMS = {"cutter (PSM2)": 0, "retract (PSM1)": 10}


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = d[:3], d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=0)


def angles_between(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    out = np.empty(len(rot6d_a))
    for i, (a, b) in enumerate(zip(rot6d_a, rot6d_b)):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            out[i] = np.nan
            continue
        delta = rot6d_to_matrix(a) @ rot6d_to_matrix(b).T
        out[i] = np.rad2deg(np.linalg.norm(R.from_matrix(delta).as_rotvec()))
    return out


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges where ``mask`` is True."""
    padded = np.concatenate([[0], mask.view(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded) != 0)
    return list(zip(edges[::2], edges[1::2]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="the .npz written by deploy_with_intervention.py")
    args = ap.parse_args(argv)

    d = np.load(args.log, allow_pickle=False)
    phases = [str(p) for p in d["phase_names"]]
    ph, cyc, t = d["phase"], d["cycles"], d["t"]
    current, action = d["current"], d["action"]
    policy_i = phases.index("policy")
    max_pos = float(d["max_pos_step"])
    max_ang = float(d["max_angle_step_deg"])

    print("=" * 72)
    print(f"HANDOVER AUDIT  {args.log}")
    print("=" * 72)
    print(f"  action mode   : {str(d['action_mode'])}")
    print(f"  cycles        : {len(ph)} over {t[-1] - t[0]:.1f}s")
    print(f"  takeovers     : {int(d['n_takeovers'])}")
    print(f"  align window  : {float(d['align_seconds']):.1f}s each way")
    print(f"  clamps        : {max_pos * 1000:.1f} mm, {max_ang:.1f} deg per command")
    print("  cycles/phase  : " + ", ".join(
        f"{p}={int((ph == i).sum())}" for i, p in enumerate(phases)))

    failures: list[str] = []

    # 1. Silence. The node logs an action only on cycles it commanded, so a
    #    finite action outside POLICY is a command that should not exist.
    held = ph != policy_i
    commanded_while_held = held & np.isfinite(action).any(axis=1)
    print("\n-- 1. was anything commanded while the node should be silent?")
    if commanded_while_held.any():
        bad = np.flatnonzero(commanded_while_held)
        failures.append(f"{len(bad)} command(s) logged outside POLICY")
        print(f"   FAIL: {len(bad)} cycle(s), first at t={t[bad[0]]:.1f}s "
              f"in phase {phases[ph[bad[0]]]}")
    else:
        print(f"   PASS: all {int(held.sum())} held cycles logged no action")

    # 2/3. Commanded steps: adjacent cycles, both under policy control.
    adjacent = np.diff(cyc) == 1
    both_policy = (ph[1:] == policy_i) & (ph[:-1] == policy_i)
    valid = adjacent & both_policy
    print("\n-- 2. commanded motion under policy control")
    if not valid.any():
        print("   (no consecutive policy cycles to measure)")
    for name, base in ARMS.items():
        pos = current[:, base + POS.start:base + POS.stop]
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)[valid]
        rot = current[:, base + ROT6D.start:base + ROT6D.stop]
        swing = angles_between(rot[1:], rot[:-1])[valid]
        swing = swing[np.isfinite(swing)]
        over_p = int((steps > max_pos + 1e-6).sum())
        over_a = int((swing > max_ang + 1e-3).sum()) if len(swing) else 0
        print(f"   {name:<16} pos: max {steps.max() * 1000:6.2f} mm  "
              f"median {np.median(steps) * 1000:5.2f} mm  over-clamp {over_p}")
        if len(swing):
            print(f"   {'':<16} ang: max {swing.max():6.2f} deg  "
                  f"median {np.median(swing):5.2f} deg  over-clamp {over_a}")
        if over_p or over_a:
            failures.append(f"{name}: {over_p} position and {over_a} angle steps over clamp")

    # 4. Each handback, and what it had to absorb.
    print("\n-- 3. each handback")
    entries = np.flatnonzero((ph[1:] == policy_i) & (ph[:-1] != policy_i)) + 1
    if not len(entries):
        print("   (no handback in this run)")
    for e in entries:
        silent = [r for r in runs_of(held) if r[1] == e]
        moved = ""
        if silent:
            a, b = silent[0]
            drift = np.linalg.norm(current[b - 1, POS] - current[a, POS])
            moved = (f"  surgeon moved the cutter {drift * 1000:.1f} mm over "
                     f"{t[b - 1] - t[a]:.1f}s")
        after = slice(e, min(e + 5, len(ph)))
        steps = np.linalg.norm(np.diff(current[after, POS], axis=0), axis=1)
        first = f"{steps.max() * 1000:.2f} mm" if len(steps) else "n/a"
        print(f"   t={t[e]:6.1f}s  first commanded steps peak at {first}"
              f"   (clamp {max_pos * 1000:.1f} mm){moved}")

    print("\n" + "=" * 72)
    if failures:
        print("RESULT: FAIL")
        for line in failures:
            print(f"  - {line}")
    else:
        print("RESULT: PASS -- nothing commanded while held, no clamp exceeded")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
