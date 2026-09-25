#!/usr/bin/env python3
"""Who is driving the arms, and when nobody is.

Control does not switch straight between the policy and the surgeon.  Each
direction goes through an **alignment window** in which *nothing commands the
PSMs at all* -- they hold the pose they were left in:

    POLICY ──pedal down──► ALIGN_TO_EXPERT ──(5 s)──► EXPERT
    EXPERT ──pedal up────► ALIGN_TO_POLICY ──(5 s)──► POLICY

Going to the surgeon, the window is for the MTM wrist to come into alignment
with the PSM tool before the master starts driving.  Coming back, it is for the
arm to settle and for the policy to re-condition on what the surgeon actually
did, so its first command is computed from reality rather than from the scene it
last saw.

Only ``PhaseState.commands_allowed`` -- true in ``POLICY`` alone -- decides
whether the deploy node publishes.  The alignment windows and ``EXPERT`` are
identical in that respect, which is deliberate: there is exactly one predicate
in the whole system that permits motion, so "is the robot allowed to move right
now" has one answer and one place to check it.

What the alignment window does NOT do
-------------------------------------
It does not hold off dVRK's teleoperation.  dVRK engages teleop on its own
schedule when COAG goes down; this node has no say in that and does not pretend
to.  What the window guarantees is that *this* node stays silent across the
transition, so the two never command the same arm in the same instant.  Going
the other way -- pedal up -- teleop has disengaged and the window genuinely does
own the arm.

The pedal is polled, and its current state always wins.  If it goes down again
during ``ALIGN_TO_POLICY``, the machine returns to ``ALIGN_TO_EXPERT`` and
restarts the timer rather than pressing on to ``POLICY``; a surgeon reaching for
the pedal a second time is not asking to wait out a countdown.  Ping-ponging
between the two alignment phases is safe by construction, because neither one
commands anything.

The clock is injected so tests can drive it directly instead of sleeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

POLICY = "policy"
ALIGN_TO_EXPERT = "align_to_expert"
EXPERT = "expert"
ALIGN_TO_POLICY = "align_to_policy"

PHASES = (POLICY, ALIGN_TO_EXPERT, EXPERT, ALIGN_TO_POLICY)
ALIGN_PHASES = (ALIGN_TO_EXPERT, ALIGN_TO_POLICY)

# Human-readable, for log lines and the end-of-run summary.
PHASE_LABEL = {
    POLICY: "POLICY",
    ALIGN_TO_EXPERT: "ALIGNING -> surgeon",
    EXPERT: "SURGEON",
    ALIGN_TO_POLICY: "ALIGNING -> policy",
}


@dataclass(frozen=True)
class PhaseState:
    """The machine's state after one ``update()``."""

    phase: str
    changed: bool           # did this update move to a different phase
    previous: str
    align_remaining: float  # seconds left in an alignment window, else 0.0
    pedal_pressed: bool

    @property
    def commands_allowed(self) -> bool:
        """The single predicate that permits this node to publish to the arms."""
        return self.phase == POLICY

    @property
    def aligning(self) -> bool:
        return self.phase in ALIGN_PHASES


class HandoverMachine:
    """Pedal state in, phase out.

    ``update()`` must be called exactly once per control cycle: it is what
    advances the alignment timer and reports transitions.
    """

    def __init__(
        self,
        align_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        start_phase: str = POLICY,
    ) -> None:
        if align_seconds < 0:
            raise ValueError(f"align_seconds must be >= 0, got {align_seconds}")
        if start_phase not in PHASES:
            raise ValueError(f"unknown start phase {start_phase!r}")
        self.align_seconds = float(align_seconds)
        self._clock = clock
        self._phase = start_phase
        self._align_until = 0.0
        self.n_takeovers = 0

    # ------------------------------------------------------------- readers
    @property
    def phase(self) -> str:
        return self._phase

    @property
    def commands_allowed(self) -> bool:
        return self._phase == POLICY

    def align_remaining(self) -> float:
        if self._phase not in ALIGN_PHASES:
            return 0.0
        return max(0.0, self._align_until - self._clock())

    # -------------------------------------------------------------- update
    def _enter(self, phase: str) -> None:
        self._phase = phase
        if phase in ALIGN_PHASES:
            self._align_until = self._clock() + self.align_seconds

    def update(self, pedal_pressed: bool) -> PhaseState:
        previous = self._phase
        pressed = bool(pedal_pressed)

        if pressed:
            # The pedal is the surgeon asking for the arms. Any phase that is
            # not already on its way to them restarts the alignment window --
            # including ALIGN_TO_POLICY, so a second press cancels a handback
            # in progress instead of waiting out its countdown.
            if self._phase in (POLICY, ALIGN_TO_POLICY):
                self._enter(ALIGN_TO_EXPERT)
                self.n_takeovers += 1
            elif self._phase == ALIGN_TO_EXPERT and self.align_remaining() <= 0.0:
                self._phase = EXPERT
        else:
            if self._phase in (EXPERT, ALIGN_TO_EXPERT):
                # Releasing during the pre-handover window is a cancelled
                # takeover; it still goes through a settle before the policy
                # resumes, because the arm may have been moved by teleop
                # already.
                self._enter(ALIGN_TO_POLICY)
            elif self._phase == ALIGN_TO_POLICY and self.align_remaining() <= 0.0:
                self._phase = POLICY

        return PhaseState(
            phase=self._phase,
            changed=self._phase != previous,
            previous=previous,
            align_remaining=self.align_remaining(),
            pedal_pressed=pressed,
        )
