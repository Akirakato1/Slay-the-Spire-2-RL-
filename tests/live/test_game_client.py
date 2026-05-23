from sts2_rl.live.game_client import LiveGameClient
from sts2_rl.live.types import LiveAction


class FakeRawClient:
    def __init__(self):
        self.actions = []
        self.state = {
            "state_type": "map",
            "run": {"act": 1, "floor": 1, "ascension": 0},
            "player": {"character": "Ironclad", "hp": 80, "max_hp": 80, "gold": 99},
            "map": {"next_options": [{"index": 0, "type": "monster"}]},
        }

    def health(self):
        return {"status": "ok", "message": "Hello from STS2 MCP v0.4.0"}

    def get_singleplayer_state(self):
        return self.state

    def post_singleplayer_action(self, body):
        self.actions.append(body)
        return {"status": "ok", "message": "done"}


def test_snapshot_normalizes_raw_state():
    client = LiveGameClient(raw_client=FakeRawClient())

    snapshot = client.snapshot()

    assert snapshot.state_type == "map"
    assert snapshot.player.character == "Ironclad"


def test_execute_maps_and_posts_action():
    raw = FakeRawClient()
    client = LiveGameClient(raw_client=raw)

    result = client.execute(LiveAction("choose_map_node", {"index": 0}))

    assert result.ok
    assert raw.actions == [{"action": "choose_map_node", "index": 0}]
