from sts2_rl.live.legal_actions import live_legal_actions
from sts2_rl.live.types import LivePlayer, LiveSnapshot


def test_combat_legal_actions_include_playable_cards_and_end_turn():
    snapshot = LiveSnapshot(
        state_type="monster",
        player=LivePlayer(
            hand=[
                {
                    "index": 0,
                    "id": "StrikeIronclad",
                    "can_play": True,
                    "target_type": "Enemy",
                },
                {
                    "index": 1,
                    "id": "DefendIronclad",
                    "can_play": True,
                    "target_type": "Self",
                },
            ]
        ),
        battle={"enemies": [{"entity_id": "JAW_WORM_0", "is_alive": True}]},
    )

    actions = live_legal_actions(snapshot)

    assert any(
        action.kind == "play_card" and action.params["card_index"] == 0
        for action in actions
    )
    assert any(action.kind == "end_turn" for action in actions)


def test_map_legal_actions_include_next_options():
    snapshot = LiveSnapshot(
        state_type="map",
        choices={"map_nodes": [{"index": 0}, {"index": 1}]},
    )

    actions = live_legal_actions(snapshot)

    assert [action.params["index"] for action in actions] == [0, 1]
