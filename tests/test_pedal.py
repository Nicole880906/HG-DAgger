"""The takeover pedal, against a real ROS 2 publisher.

This is the one input whose misreading is dangerous in both directions -- a
missed press leaves the policy driving while the operator is trying to stop it,
a stuck press leaves the arms dead -- so it is tested against rclpy rather than
a mock, including the latched-QoS path that decides whether the pedal's state is
known at startup at all.

Skipped where rclpy is not importable -- see tests/conftest.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Joy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

from pedal import PedalMonitor, wait_for_pedal  # noqa: E402

# dVRK latches its button topics: RELIABLE + TRANSIENT_LOCAL.
DVRK_BUTTON_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
UNLATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class FakePedal(Node):
    """Stands in for the dVRK console's footpedal publisher."""

    def __init__(self, topic: str, qos: QoSProfile = DVRK_BUTTON_QOS):
        super().__init__("fake_pedal_" + topic.strip("/").replace("/", "_"))
        self.pub = self.create_publisher(Joy, topic, qos)

    def send(self, value: int) -> None:
        msg = Joy()
        msg.buttons = [int(value)]
        self.pub.publish(msg)


def pump(nodes, seconds: float = 0.3) -> None:
    """Spin every node briefly so published messages are delivered.

    Publishers here use depth-1 KEEP_LAST, like dVRK's, so two messages sent
    without a pump between them are not two events on the wire -- the second
    overwrites the first. Any test that needs a *sequence* observed must pump
    between the sends.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.005)


@pytest.fixture
def rig(request):
    """A monitor and a publisher on a topic unique to the calling test."""
    topic = f"/test_footpedals/{request.node.name}"
    listener = Node("pedal_listener_" + request.node.name.replace("[", "_").replace("]", "_"))
    monitor = PedalMonitor(listener, topic=topic, name="coag")
    publisher = FakePedal(topic)
    pump([listener, publisher])
    yield monitor, publisher, listener
    listener.destroy_node()
    publisher.destroy_node()


class TestState:
    def test_starts_with_nothing_seen(self, rig):
        monitor, _, _ = rig
        assert not monitor.seen
        assert not monitor.pressed
        assert monitor.age is None

    def test_press_and_release_are_tracked(self, rig):
        monitor, publisher, listener = rig
        publisher.send(1)
        pump([listener, publisher])
        assert monitor.seen and monitor.pressed

        publisher.send(0)
        pump([listener, publisher])
        assert not monitor.pressed

    def test_quick_tap_does_not_leave_the_pedal_held(self, rig):
        """cisst encodes a quick tap as 2. Treating it as pressed would strand
        control with the expert until the pedal was pressed and released again."""
        monitor, publisher, listener = rig
        publisher.send(2)
        pump([listener, publisher])
        assert monitor.seen
        assert not monitor.pressed
        assert monitor.poll().taps == 1

    def test_empty_button_array_reads_as_released(self, rig):
        monitor, publisher, listener = rig
        publisher.pub.publish(Joy())
        pump([listener, publisher])
        assert monitor.seen and not monitor.pressed


class TestEdges:
    def test_rising_edge_fires_once(self, rig):
        monitor, publisher, listener = rig
        assert not monitor.poll().rising

        publisher.send(1)
        pump([listener, publisher])
        edge = monitor.poll()
        assert edge.rising and edge.pressed and not edge.falling
        # The next cycle sees the pedal still held, but no new edge.
        again = monitor.poll()
        assert again.pressed and not again.rising

    def test_falling_edge_fires_once(self, rig):
        monitor, publisher, listener = rig
        publisher.send(1)
        pump([listener, publisher])
        monitor.poll()

        publisher.send(0)
        pump([listener, publisher])
        edge = monitor.poll()
        assert edge.falling and not edge.pressed and not edge.rising
        assert not monitor.poll().falling

    def test_press_and_release_inside_one_cycle_reports_the_current_state(self, rig):
        """Both messages land between polls: the second one is the truth.

        A stale 'still pressed' here would suspend the policy indefinitely with
        the operator's foot already off the pedal.
        """
        monitor, publisher, listener = rig
        monitor.poll()
        publisher.send(1)
        pump([listener, publisher])
        publisher.send(0)
        pump([listener, publisher])
        edge = monitor.poll()
        assert not edge.pressed
        assert not edge.rising

    def test_taps_are_cleared_by_polling(self, rig):
        monitor, publisher, listener = rig
        # A real second tap is separated from the first by a released state on
        # the wire; back-to-back identical values are the double delivery from
        # the two subscriptions, and are deduplicated.
        publisher.send(2)
        pump([listener, publisher])
        publisher.send(0)
        pump([listener, publisher])
        publisher.send(2)
        pump([listener, publisher])
        assert monitor.poll().taps == 2
        assert monitor.poll().taps == 0

    def test_repeated_identical_messages_are_counted_once(self, rig):
        """The documented cost of subscribing twice.

        Both the volatile and the transient-local reader get every message, so
        the counters key on a change of value. Two taps with nothing between
        them collapse into one -- acceptable, because taps are informational
        and the held state, which is what gates the arms, is unaffected.
        """
        monitor, publisher, listener = rig
        publisher.send(2)
        pump([listener, publisher])
        publisher.send(2)
        pump([listener, publisher])
        assert monitor.poll().taps == 1
        assert not monitor.pressed


class TestLatchedStartup:
    def test_retained_state_arrives_without_touching_the_pedal(self, request):
        """The decisive case for the double subscription.

        dVRK publishes the pedal latched and event-driven: if the pedal was
        pressed before this node started, the only evidence is the retained
        sample, and a volatile-only subscriber is never handed it. The node
        would then come up believing the pedal had never been seen and refuse
        to run -- or worse, with --allow-no-pedal, run anyway.
        """
        topic = f"/test_footpedals/{request.node.name}_latched"
        publisher = FakePedal(topic)
        publisher.send(1)
        pump([publisher], 0.2)

        listener = Node("late_listener_" + request.node.name)
        monitor = PedalMonitor(listener, topic=topic)
        pump([listener, publisher], 0.5)
        try:
            assert monitor.seen, "retained pedal state never arrived"
            assert monitor.pressed
        finally:
            listener.destroy_node()
            publisher.destroy_node()

    def test_an_unlatched_publisher_still_matches(self, request):
        """The volatile half of the double subscription.

        Transient-local alone would silently fail to match a VOLATILE publisher
        -- incompatible QoS in DDS is not an error, just no data ever arriving.
        """
        topic = f"/test_footpedals/{request.node.name}_volatile"
        listener = Node("volatile_listener_" + request.node.name)
        monitor = PedalMonitor(listener, topic=topic)
        publisher = FakePedal(topic, qos=UNLATCHED_QOS)
        pump([listener, publisher], 0.3)

        publisher.send(1)
        pump([listener, publisher], 0.3)
        try:
            assert monitor.seen and monitor.pressed
        finally:
            listener.destroy_node()
            publisher.destroy_node()


class TestWaitForPedal:
    def test_returns_false_when_the_topic_is_silent(self, request):
        listener = Node("silent_listener_" + request.node.name)
        monitor = PedalMonitor(listener, topic=f"/test_footpedals/{request.node.name}_silent")
        try:
            started = time.monotonic()
            assert wait_for_pedal(monitor, timeout=0.3) is False
            assert time.monotonic() - started >= 0.3
        finally:
            listener.destroy_node()

    def test_returns_true_once_a_state_is_known(self, rig):
        monitor, publisher, listener = rig
        publisher.send(0)
        pump([listener, publisher])
        assert wait_for_pedal(monitor, timeout=1.0) is True

    def test_it_spins_the_node_itself(self, request):
        """Regression: it must deliver the message, not just wait for it.

        This runs before rclpy.spin() owns the node, so nothing else is
        executing callbacks. A version that slept instead of spinning waited on
        a message already sitting in the middleware, always timed out, and made
        every --execute run refuse to start with a healthy pedal attached.

        Deliberately NO pump() here -- pumping is what hid the bug.
        """
        topic = f"/test_footpedals/{request.node.name}_selfspin"
        publisher = FakePedal(topic)
        listener = Node("selfspin_listener_" + request.node.name)
        monitor = PedalMonitor(listener, topic=topic)
        publisher.send(1)
        try:
            assert wait_for_pedal(monitor, timeout=5.0) is True
            assert monitor.pressed
        finally:
            listener.destroy_node()
            publisher.destroy_node()
