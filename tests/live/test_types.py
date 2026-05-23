from sts2_rl.live.types import LiveAction, LiveActionResult, LivePlayer, LiveSnapshot


def test_live_snapshot_identifies_decision_point():
    snapshot = LiveSnapshot(
        state_type="monster",
        run={"act": 1, "floor": 3, "ascension": 0},
        player=LivePlayer(character="Ironclad", hp=70, max_hp=80, gold=99),
        raw={"state_type": "monster"},
    )

    assert snapshot.is_decision_point()


def test_live_action_result_requires_status_ok():
    result = LiveActionResult(status="ok", message="played card", raw={"status": "ok"})

    assert result.ok is True


def test_live_action_builds_policy_independent_action():
    action = LiveAction(
        kind="play_card", params={"card_index": 0, "target": "JAW_WORM_0"}
    )

    assert action.kind == "play_card"
    assert action.params["target"] == "JAW_WORM_0"
