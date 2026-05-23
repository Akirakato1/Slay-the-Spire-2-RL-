from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DECISION_STATE_TYPES = {
    "monster",
    "elite",
    "boss",
    "hand_select",
    "rewards",
    "card_reward",
    "map",
    "event",
    "rest_site",
    "shop",
    "fake_merchant",
    "treasure",
    "card_select",
    "bundle_select",
    "relic_select",
    "crystal_sphere",
    "game_over",
}


@dataclass(frozen=True)
class LivePlayer:
    character: str | None = None
    hp: int | None = None
    max_hp: int | None = None
    gold: int | None = None
    energy: int | None = None
    max_energy: int | None = None
    relics: list[dict[str, Any]] = field(default_factory=list)
    potions: list[dict[str, Any]] = field(default_factory=list)
    hand: list[dict[str, Any]] = field(default_factory=list)
    draw_pile_count: int | None = None
    discard_pile_count: int | None = None
    exhaust_pile_count: int | None = None


@dataclass(frozen=True)
class LiveSnapshot:
    state_type: str
    run: dict[str, Any] = field(default_factory=dict)
    player: LivePlayer | None = None
    battle: dict[str, Any] | None = None
    map: dict[str, Any] | None = None
    choices: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def is_decision_point(self) -> bool:
        return self.state_type in DECISION_STATE_TYPES


@dataclass(frozen=True)
class LiveAction:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveActionResult:
    status: str
    message: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"
