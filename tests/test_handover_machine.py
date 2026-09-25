"""The four-phase handover machine, driven by a fake clock.

Pure logic -- no ROS, no robot, no sleeping -- so the alignment windows can be
stepped through instant by instant and every edge case a foot can produce is
cheap to cover.

The property that matters most is the negative one: ``commands_allowed`` is true
in exactly one phase. Everything else here is about making sure the machine
reaches that phase only when it should.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from handover import (  # noqa: E402
    ALIGN_PHASES,
    ALIGN_TO_EXPERT,
    ALIGN_TO_POLICY,
    EXPERT,
    PHASES,
    POLICY,
    HandoverMachine,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def rig():
    clock = FakeClock()
    return HandoverMachine(align_seconds=5.0, clock=clock), clock


def settle(machine, clock, pressed: bool, seconds: float = 5.001):
    """Hold the pedal steady long enough for an alignment window to expire."""
    machine.update(pressed)
    clock.advance(seconds)
    return machine.update(pressed)


class TestHappyPath:
    def test_starts_under_policy_control(self, rig):
        machine, _ = rig
        assert machine.phase == POLICY
        assert machine.commands_allowed

    def test_press_goes_through_alignment_before_the_surgeon_drives(self, rig):
        machine, clock = rig
        state = machine.update(True)
        assert state.phase == ALIGN_TO_EXPERT
        assert state.changed
        assert not state.commands_allowed

        clock.advance(4.9)
        assert machine.update(True).phase == ALIGN_TO_EXPERT, "left early"

        clock.advance(0.2)
        assert machine.update(True).phase == EXPERT

    def test_release_goes_through_alignment_before_the_policy_drives(self, rig):
        machine, clock = rig
        settle(machine, clock, True)
        assert machine.phase == EXPERT

        state = machine.update(False)
        assert state.phase == ALIGN_TO_POLICY
        assert not state.commands_allowed

        clock.advance(4.9)
        assert machine.update(False).phase == ALIGN_TO_POLICY

        clock.advance(0.2)
        state = machine.update(False)
        assert state.phase == POLICY
        assert state.commands_allowed

    def test_a_full_cycle_counts_one_takeover(self, rig):
        machine, clock = rig
        settle(machine, clock, True)
        settle(machine, clock, False)
        assert machine.phase == POLICY
        assert machine.n_takeovers == 1


class TestCommandsAllowed:
    def test_only_policy_phase_permits_motion(self):
        """The whole safety argument in one assertion."""
        for phase in PHASES:
            machine = HandoverMachine(align_seconds=5.0, clock=FakeClock(),
                                      start_phase=phase)
            assert machine.commands_allowed == (phase == POLICY), phase

    def test_both_alignment_windows_are_silent(self, rig):
        machine, clock = rig
        assert not machine.update(True).commands_allowed        # align to expert
        clock.advance(6)
        machine.update(True)
        assert not machine.update(False).commands_allowed       # align to policy

    def test_expert_phase_is_silent(self, rig):
        machine, clock = rig
        state = settle(machine, clock, True)
        assert state.phase == EXPERT and not state.commands_allowed


class TestPedalWins:
    def test_press_during_handback_cancels_it(self, rig):
        """A surgeon reaching for the pedal again is not asking to wait."""
        machine, clock = rig
        settle(machine, clock, True)
        machine.update(False)
        clock.advance(2.0)
        assert machine.phase == ALIGN_TO_POLICY

        state = machine.update(True)
        assert state.phase == ALIGN_TO_EXPERT
        assert state.align_remaining == pytest.approx(5.0)

    def test_release_during_pre_handover_still_settles(self, rig):
        """A cancelled takeover settles too: teleop may already have moved the arm."""
        machine, clock = rig
        machine.update(True)
        clock.advance(2.0)
        state = machine.update(False)
        assert state.phase == ALIGN_TO_POLICY
        assert not state.commands_allowed

    def test_ping_pong_never_permits_motion(self, rig):
        """Rapid pedal flutter must not find a gap where commands slip through."""
        machine, clock = rig
        machine.update(True)
        for i in range(40):
            clock.advance(0.1)
            state = machine.update(i % 2 == 0)
            assert not state.commands_allowed, f"motion permitted at step {i}"

    def test_pedal_already_down_at_startup_is_honoured(self, rig):
        machine, _ = rig
        state = machine.update(True)
        assert state.phase == ALIGN_TO_EXPERT
        assert not state.commands_allowed

    def test_holding_the_pedal_stays_expert_indefinitely(self, rig):
        machine, clock = rig
        settle(machine, clock, True)
        for _ in range(50):
            clock.advance(1.0)
            assert machine.update(True).phase == EXPERT


class TestTimer:
    def test_align_remaining_counts_down(self, rig):
        machine, clock = rig
        machine.update(True)
        assert machine.align_remaining() == pytest.approx(5.0)
        clock.advance(3.0)
        assert machine.align_remaining() == pytest.approx(2.0)
        clock.advance(9.0)
        assert machine.align_remaining() == 0.0

    def test_align_remaining_is_zero_outside_the_windows(self, rig):
        machine, clock = rig
        assert machine.align_remaining() == 0.0
        settle(machine, clock, True)
        assert machine.align_remaining() == 0.0

    def test_zero_align_seconds_switches_on_the_next_cycle(self):
        """Opt out of the windows without changing any other behaviour."""
        clock = FakeClock()
        machine = HandoverMachine(align_seconds=0.0, clock=clock)
        assert machine.update(True).phase == ALIGN_TO_EXPERT
        assert machine.update(True).phase == EXPERT
        assert machine.update(False).phase == ALIGN_TO_POLICY
        assert machine.update(False).phase == POLICY

    def test_transition_is_reported_once(self, rig):
        machine, clock = rig
        assert machine.update(True).changed
        assert not machine.update(True).changed
        clock.advance(6.0)
        assert machine.update(True).changed
        assert not machine.update(True).changed

    def test_previous_phase_is_reported(self, rig):
        machine, clock = rig
        clock.advance(0.0)
        machine.update(True)
        clock.advance(6.0)
        state = machine.update(True)
        assert state.previous == ALIGN_TO_EXPERT and state.phase == EXPERT


class TestValidation:
    def test_negative_align_seconds_is_rejected(self):
        with pytest.raises(ValueError):
            HandoverMachine(align_seconds=-1.0)

    def test_unknown_start_phase_is_rejected(self):
        with pytest.raises(ValueError):
            HandoverMachine(start_phase="driving")

    def test_align_phases_are_the_two_windows(self):
        assert set(ALIGN_PHASES) == {ALIGN_TO_EXPERT, ALIGN_TO_POLICY}
