from sts2_rl.live.types import LiveSnapshot
from sts2_rl.live.wait import wait_for_decision_point


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_wait_returns_first_decision_point():
    snapshots = [
        LiveSnapshot(state_type="unknown"),
        LiveSnapshot(state_type="map"),
    ]

    def poll():
        return snapshots.pop(0)

    result = wait_for_decision_point(
        poll, timeout_s=1.0, interval_s=0.1, clock=FakeClock()
    )

    assert result.state_type == "map"
