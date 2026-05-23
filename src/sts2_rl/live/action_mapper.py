from __future__ import annotations

from typing import Any

from sts2_rl.live.types import LiveAction


SUPPORTED_ACTION_KINDS = {
    "play_card",
    "use_potion",
    "discard_potion",
    "end_turn",
    "combat_select_card",
    "combat_confirm_selection",
    "claim_reward",
    "proceed",
    "select_card_reward",
    "skip_card_reward",
    "choose_map_node",
    "choose_event_option",
    "advance_dialogue",
    "choose_rest_option",
    "shop_purchase",
    "claim_treasure_relic",
    "select_card",
    "confirm_selection",
    "cancel_selection",
    "select_bundle",
    "confirm_bundle_selection",
    "cancel_bundle_selection",
    "select_relic",
    "skip_relic_selection",
    "crystal_sphere_set_tool",
    "crystal_sphere_click_cell",
    "crystal_sphere_proceed",
}


def map_live_action_to_sts2mcp_body(action: LiveAction) -> dict[str, Any]:
    if action.kind not in SUPPORTED_ACTION_KINDS:
        raise ValueError(f"unknown live action kind: {action.kind}")
    body: dict[str, Any] = {"action": action.kind}
    body.update(action.params)
    return body
