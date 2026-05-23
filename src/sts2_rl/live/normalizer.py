from __future__ import annotations

from typing import Any

from sts2_rl.live.types import LivePlayer, LiveSnapshot


def normalize_sts2mcp_state(raw: dict[str, Any]) -> LiveSnapshot:
    state_type = str(raw.get("state_type", "unknown"))
    player_raw = raw.get("player") or {}
    player = LivePlayer(
        character=player_raw.get("character"),
        hp=player_raw.get("hp"),
        max_hp=player_raw.get("max_hp"),
        gold=player_raw.get("gold"),
        energy=player_raw.get("energy"),
        max_energy=player_raw.get("max_energy"),
        relics=list(player_raw.get("relics") or []),
        potions=list(player_raw.get("potions") or []),
        hand=list(player_raw.get("hand") or []),
        draw_pile_count=player_raw.get("draw_pile_count"),
        discard_pile_count=player_raw.get("discard_pile_count"),
        exhaust_pile_count=player_raw.get("exhaust_pile_count"),
    )
    choices = _extract_choices(state_type, raw)
    return LiveSnapshot(
        state_type=state_type,
        run=dict(raw.get("run") or {}),
        player=player,
        battle=raw.get("battle"),
        map=raw.get("map"),
        choices=choices,
        raw=raw,
    )


def _extract_choices(state_type: str, raw: dict[str, Any]) -> dict[str, Any]:
    if state_type == "map":
        map_state = raw.get("map") or {}
        return {"map_nodes": list(map_state.get("next_options") or [])}
    if state_type == "event":
        event_state = raw.get("event") or {}
        return {"event_options": list(event_state.get("options") or [])}
    if state_type == "rewards":
        reward_state = raw.get("rewards") or {}
        return {"rewards": list(reward_state.get("items") or [])}
    if state_type == "card_reward":
        card_reward = raw.get("card_reward") or {}
        return {"cards": list(card_reward.get("cards") or [])}
    if state_type in {"shop", "fake_merchant"}:
        shop_state = raw.get("shop") or raw.get("fake_merchant", {}).get("shop") or {}
        return {"shop_items": list(shop_state.get("items") or [])}
    if state_type == "rest_site":
        rest_state = raw.get("rest_site") or {}
        return {"rest_options": list(rest_state.get("options") or [])}
    if state_type == "treasure":
        treasure_state = raw.get("treasure") or {}
        return {"treasure_relics": list(treasure_state.get("relics") or [])}
    return {}
