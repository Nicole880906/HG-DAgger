#!/usr/bin/env python3
"""dVRK footpedal state, read safely enough to gate robot motion on it.

The deploy node hands control of the arms to the human the instant this says
the pedal is down, so every awkward detail below exists because getting it
wrong means either the policy and the operator fight over the arms, or the
takeover the operator asked for never happens.

QoS
---
dVRK publishes every button topic **latched** -- ``cisst_ral.h``'s
``create_publisher`` defaults ``latched`` to true, which is RELIABLE +
TRANSIENT_LOCAL.  A pedal is event-driven: between presses there is no traffic
at all, and the current state exists only as that one retained sample.  A
VOLATILE subscriber is never given a retained sample, so a node that subscribed
only that way would sit at startup believing the pedal had never been touched
until somebody physically stamped on it.

So subscribe **twice**, with both profiles.  The transient-local reader picks up
the retained state on connect; the volatile reader still matches a publisher
that does not latch, which transient-local alone would silently fail to match
(incompatible QoS in DDS is not an error, it is just no data).

The price is that both readers run the same callback on the same message.
Assigning the pedal's state twice is harmless -- it is a boolean, and the second
assignment writes the same value.  *Counting* twice is not, so the counters in
the callback only advance when the button value actually changes.

Button encoding
---------------
``sensor_msgs/Joy`` with a single button: ``0`` released, ``1`` pressed.  Some
cisst button interfaces also emit ``2`` for a quick tap, which is a press and
release collapsed into one message.  Only ``1`` is treated as held; anything
else is released.  A quick tap must never leave this object believing the pedal
is still down, because that state is what keeps the policy switched off.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy

COAG_TOPIC = "/footpedals/coag"
CLUTCH_TOPIC = "/footpedals/clutch"

_VOLATILE = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
_LATCHED = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclass(frozen=True)
class PedalEdge:
    """What the pedal did between two consecutive ``poll()`` calls."""

    pressed: bool
    rising: bool   # released -> pressed, i.e. the expert just took over
    falling: bool  # pressed -> released, i.e. the expert just handed back
    taps: int      # quick-tap events seen since the last poll


class PedalMonitor:
    """Latest state of one footpedal, with edge detection on ``poll()``.

    ``poll()`` is the only method the control loop calls, and it must be called
    exactly once per cycle: it consumes the edges, so calling it twice in one
    cycle would hide a transition from the second caller.
    """

    def __init__(self, node: Node, topic: str = COAG_TOPIC, name: str = "coag") -> None:
        self.node = node
        self.topic = topic
        self.name = name
        self._lock = threading.Lock()
        self._pressed = False
        self._received_at: float | None = None
        self._press_count = 0
        self._tap_count = 0
        self._last_raw: int | None = None
        self._last_polled_pressed = False

        for profile in (_VOLATILE, _LATCHED):
            node.create_subscription(Joy, topic, self._on_joy, profile)

    # ------------------------------------------------------------- callback
    def _on_joy(self, msg: Joy) -> None:
        raw = int(msg.buttons[0]) if msg.buttons else 0
        pressed = raw == 1
        with self._lock:
            # The volatile and transient-local subscriptions both deliver every
            # message, so anything that accumulates has to be guarded on a real
            # change of value. The state assignments below are idempotent and
            # are not guarded: if a message is ever delivered only once, the
            # pedal's state -- the part that decides who is driving -- is still
            # correct.
            if raw != self._last_raw:
                if pressed and not self._pressed:
                    self._press_count += 1
                if raw == 2:
                    self._tap_count += 1
            self._last_raw = raw
            self._pressed = pressed
            self._received_at = time.monotonic()

    # -------------------------------------------------------------- readers
    @property
    def seen(self) -> bool:
        """True once any message has arrived on the topic.

        False does not mean the pedal is broken -- it is event-driven, and a
        publisher that does not latch says nothing until the pedal is used.  It
        does mean this node cannot yet prove a takeover path exists, which is
        why ``wait_for_pedal`` blocks on it before any arm is commanded.
        """
        with self._lock:
            return self._received_at is not None

    @property
    def pressed(self) -> bool:
        with self._lock:
            return self._pressed

    @property
    def age(self) -> float | None:
        """Seconds since the last message, or None if there has been none.

        Not a freshness signal: silence is the normal state of a pedal that is
        not being used.  Useful only for logging.
        """
        with self._lock:
            if self._received_at is None:
                return None
            return time.monotonic() - self._received_at

    def poll(self) -> PedalEdge:
        """Current state plus the edges since the previous call."""
        with self._lock:
            pressed = self._pressed
            taps = self._tap_count
            self._tap_count = 0
        rising = pressed and not self._last_polled_pressed
        falling = (not pressed) and self._last_polled_pressed
        self._last_polled_pressed = pressed
        return PedalEdge(pressed=pressed, rising=rising, falling=falling, taps=taps)


def wait_for_pedal(monitor: PedalMonitor, timeout: float) -> bool:
    """Block until the pedal topic has produced a state, or time out.

    Returns True if a state arrived.  The caller decides what a False means --
    ``deploy_with_intervention.py`` refuses to move the arms, because a policy
    running with no working takeover pedal is the one configuration this whole
    program exists to avoid.
    """
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    deadline = time.monotonic() + float(timeout)
    monitor.node.get_logger().info(
        f"Waiting up to {timeout:.0f}s for {monitor.topic} "
        f"(tap the {monitor.name} pedal once if nothing arrives) ..."
    )
    # Spin, do not sleep. This runs before rclpy.spin() owns the node, so
    # nothing else is executing callbacks: sleeping here waits for a message
    # that is sitting in the middleware and will never be delivered, the wait
    # always times out, and a node started with --execute refuses to run
    # because it cannot prove the pedal exists. The pedal was fine; the program
    # just never looked.
    #
    # One executor for the whole wait, rather than rclpy.spin_once() per
    # iteration -- that helper builds and tears down an executor on every call,
    # which at 20 calls a second is most of the cost of waiting.
    executor = SingleThreadedExecutor()
    executor.add_node(monitor.node)
    try:
        while rclpy.ok() and not monitor.seen:
            if time.monotonic() > deadline:
                return False
            executor.spin_once(timeout_sec=0.05)
        return monitor.seen
    finally:
        executor.remove_node(monitor.node)
        executor.shutdown()
