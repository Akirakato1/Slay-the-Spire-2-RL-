from __future__ import annotations

from sts2_rl.live.action_mapper import map_live_action_to_sts2mcp_body
from sts2_rl.live.normalizer import normalize_sts2mcp_state
from sts2_rl.live.sts2mcp_client import Sts2McpClient
from sts2_rl.live.types import LiveAction, LiveActionResult, LiveSnapshot
from sts2_rl.live.wait import wait_for_decision_point


class LiveGameClient:
    def __init__(self, raw_client: Sts2McpClient | None = None) -> None:
        self.raw_client = raw_client or Sts2McpClient()

    def health(self) -> dict:
        return self.raw_client.health()

    def snapshot(self) -> LiveSnapshot:
        return normalize_sts2mcp_state(self.raw_client.get_singleplayer_state())

    def execute(self, action: LiveAction) -> LiveActionResult:
        body = map_live_action_to_sts2mcp_body(action)
        raw = self.raw_client.post_singleplayer_action(body)
        return LiveActionResult(
            status=str(raw.get("status", "error")),
            message=raw.get("message"),
            raw=raw,
        )

    def step(self, action: LiveAction, wait: bool = True) -> LiveSnapshot:
        result = self.execute(action)
        if not result.ok:
            raise RuntimeError(result.message or "live action failed")
        if wait:
            return self.wait_for_decision_point()
        return self.snapshot()

    def wait_for_decision_point(self, timeout_s: float = 10.0) -> LiveSnapshot:
        return wait_for_decision_point(self.snapshot, timeout_s=timeout_s)
