from __future__ import annotations

import argparse

from sts2_rl.live.game_client import LiveGameClient
from sts2_rl.live.sts2mcp_client import Sts2McpClient
from sts2_rl.live.types import LiveAction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test STS2MCP live interaction.")
    parser.add_argument("--base-url", default="http://localhost:15526")
    parser.add_argument("--execute-end-turn", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = LiveGameClient(raw_client=Sts2McpClient(base_url=args.base_url))
    health = client.health()
    print(f"health={health}")
    snapshot = client.snapshot()
    print(f"state_type={snapshot.state_type}")
    print(f"decision_point={snapshot.is_decision_point()}")
    if args.execute_end_turn:
        result = client.execute(LiveAction("end_turn"))
        print(f"end_turn={result.raw}")


if __name__ == "__main__":
    main()
