from sts2_rl.live.action_mapper import map_live_action_to_sts2mcp_body
from sts2_rl.live.types import LiveAction


def test_maps_play_card_action():
    action = LiveAction("play_card", {"card_index": 0, "target": "JAW_WORM_0"})

    assert map_live_action_to_sts2mcp_body(action) == {
        "action": "play_card",
        "card_index": 0,
        "target": "JAW_WORM_0",
    }


def test_maps_end_turn_action():
    assert map_live_action_to_sts2mcp_body(LiveAction("end_turn")) == {
        "action": "end_turn",
    }


def test_rejects_unknown_action_kind():
    try:
        map_live_action_to_sts2mcp_body(LiveAction("unknown"))
    except ValueError as error:
        assert "unknown live action kind" in str(error)
    else:
        raise AssertionError("expected ValueError")
