from __future__ import annotations

from sts2_rl.live.types import LiveAction, LiveSnapshot


def live_legal_actions(snapshot: LiveSnapshot) -> list[LiveAction]:
    if snapshot.state_type in {"monster", "elite", "boss"}:
        return _combat_actions(snapshot)
    if snapshot.state_type == "map":
        return [
            LiveAction("choose_map_node", {"index": node["index"]})
            for node in snapshot.choices.get("map_nodes", [])
        ]
    if snapshot.state_type == "event":
        return [
            LiveAction("choose_event_option", {"index": option["index"]})
            for option in snapshot.choices.get("event_options", [])
        ]
    if snapshot.state_type == "card_reward":
        actions = [
            LiveAction("select_card_reward", {"card_index": card["index"]})
            for card in snapshot.choices.get("cards", [])
        ]
        actions.append(LiveAction("skip_card_reward"))
        return actions
    if snapshot.state_type == "rewards":
        actions = [
            LiveAction("claim_reward", {"index": reward["index"]})
            for reward in snapshot.choices.get("rewards", [])
        ]
        actions.append(LiveAction("proceed"))
        return actions
    if snapshot.state_type in {"shop", "fake_merchant"}:
        actions = [
            LiveAction("shop_purchase", {"index": item["index"]})
            for item in snapshot.choices.get("shop_items", [])
        ]
        actions.append(LiveAction("proceed"))
        return actions
    if snapshot.state_type == "rest_site":
        return [
            LiveAction("choose_rest_option", {"index": option["index"]})
            for option in snapshot.choices.get("rest_options", [])
        ]
    if snapshot.state_type == "treasure":
        actions = [
            LiveAction("claim_treasure_relic", {"index": relic["index"]})
            for relic in snapshot.choices.get("treasure_relics", [])
        ]
        actions.append(LiveAction("proceed"))
        return actions
    return []


def _combat_actions(snapshot: LiveSnapshot) -> list[LiveAction]:
    actions: list[LiveAction] = []
    player = snapshot.player
    enemies = []
    if snapshot.battle:
        enemies = [
            enemy
            for enemy in snapshot.battle.get("enemies", [])
            if enemy.get("is_alive", True)
        ]
    if player:
        for card in player.hand:
            if not card.get("can_play", False):
                continue
            target_type = str(card.get("target_type", "")).lower()
            if "enemy" in target_type:
                for enemy in enemies:
                    actions.append(
                        LiveAction(
                            "play_card",
                            {
                                "card_index": card["index"],
                                "target": enemy["entity_id"],
                            },
                        )
                    )
            else:
                actions.append(LiveAction("play_card", {"card_index": card["index"]}))
    actions.append(LiveAction("end_turn"))
    return actions
