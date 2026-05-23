from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from sts2_rl.live.types import LiveSnapshot


class Clock(Protocol):
    def monotonic(self) -> float:
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        raise NotImplementedError


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def wait_for_decision_point(
    poll: Callable[[], LiveSnapshot],
    timeout_s: float = 10.0,
    interval_s: float = 0.1,
    clock: Clock | None = None,
) -> LiveSnapshot:
    active_clock = clock or SystemClock()
    deadline = active_clock.monotonic() + timeout_s
    last = poll()
    while not last.is_decision_point():
        if active_clock.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for decision point; last state={last.state_type}"
            )
        active_clock.sleep(interval_s)
        last = poll()
    return last
