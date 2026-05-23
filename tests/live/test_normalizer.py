from sts2_rl.live.normalizer import normalize_sts2mcp_state


def test_normalizes_combat_state():
    raw = {
        "state_type": "monster",
        "run": {"act": 1, "floor": 2, "ascension": 0},
        "player": {
            "character": "Ironclad",
            "hp": 70,
            "max_hp": 80,
            "gold": 99,
            "energy": 3,
            "max_energy": 3,
            "hand": [{"id": "StrikeIronclad", "index": 0, "can_play": True}],
            "draw_pile_count": 5,
            "discard_pile_count": 0,
            "exhaust_pile_count": 0,
            "relics": [{"id": "BurningBlood"}],
            "potions": [],
        },
        "battle": {"enemies": [{"id": "JAW_WORM", "entity_id": "JAW_WORM_0"}]},
    }

    snapshot = normalize_sts2mcp_state(raw)

    assert snapshot.state_type == "monster"
    assert snapshot.player.character == "Ironclad"
    assert snapshot.player.hand[0]["id"] == "StrikeIronclad"
    assert snapshot.battle["enemies"][0]["entity_id"] == "JAW_WORM_0"


def test_normalizes_map_choices():
    raw = {
        "state_type": "map",
        "run": {"act": 1, "floor": 1, "ascension": 0},
        "player": {"character": "Ironclad", "hp": 80, "max_hp": 80, "gold": 99},
        "map": {"next_options": [{"index": 0, "type": "monster"}]},
    }

    snapshot = normalize_sts2mcp_state(raw)

    assert snapshot.state_type == "map"
    assert snapshot.choices["map_nodes"][0]["index"] == 0
