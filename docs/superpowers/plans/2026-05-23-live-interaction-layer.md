# Live Interaction Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a policy-independent live game interaction layer that can read Slay the Spire 2 singleplayer state through the STS2MCP mod, normalize it into our semantic contract, and execute semantic actions back into the game.

**Architecture:** The interaction layer is a backend adapter, not a policy. It wraps the STS2MCP singleplayer REST API behind `LiveGameClient`, converts raw mod JSON into stable semantic dataclasses, maps semantic actions into STS2MCP POST bodies, and exposes a `snapshot -> legal_actions -> step -> snapshot` loop that later plugs into the observation layer and trained policy.

**Tech Stack:** Python 3.10+, `requests`, `pytest`, dataclasses, STS2MCP singleplayer API on `localhost:15526`.

---

## Scope

This plan belongs in the RL repo. It does not change simulator gameplay and does
not train a model. The interaction layer should be useful before a trained
policy exists.

Build only the singleplayer path first:

```text
Actual StS2 game
  -> STS2MCP GET/POST /api/v1/singleplayer
  -> LiveGameClient
  -> LiveSnapshot + LiveAction
  -> future observation layer/policy
```

Out of scope for this plan:

- multiplayer,
- model inference,
- PPO/transformer/training loops,
- direct game mod development,
- full observation encoder integration.

## File Structure

- `pyproject.toml`: package metadata and dependencies.
- `src/sts2_rl/__init__.py`: package marker.
- `src/sts2_rl/live/__init__.py`: live interaction package exports.
- `src/sts2_rl/live/types.py`: semantic state/action/result dataclasses.
- `src/sts2_rl/live/sts2mcp_client.py`: raw HTTP wrapper for STS2MCP.
- `src/sts2_rl/live/normalizer.py`: raw STS2MCP JSON -> `LiveSnapshot`.
- `src/sts2_rl/live/action_mapper.py`: `LiveAction` -> STS2MCP POST body.
- `src/sts2_rl/live/game_client.py`: high-level interaction loop.
- `src/sts2_rl/live/wait.py`: decision-point polling helper.
- `scripts/live_smoke.py`: optional read-only/controlled live bridge smoke check.
- `tests/live/`: tests with fake clients and raw JSON fixtures.
- `docs/architecture/live-game-interaction-layer.md`: short architecture doc.

---

### Task 1: Python Package Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/sts2_rl/__init__.py`
- Create: `src/sts2_rl/live/__init__.py`
- Create: `tests/live/test_package_import.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/live/test_package_import.py`:

```python
def test_live_package_imports():
    import sts2_rl.live as live

    assert isinstance(live.__all__, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/live/test_package_import.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sts2_rl'`.

- [ ] **Step 3: Create package scaffold**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "sts2-rl"
version = "0.1.0"
description = "RL architecture and live interaction tooling for Slay the Spire 2"
requires-python = ">=3.10"
dependencies = [
  "requests>=2.31.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `src/sts2_rl/__init__.py`:

```python
"""Slay the Spire 2 RL tooling."""
```

Create `src/sts2_rl/live/__init__.py`:

```python
"""Live game interaction layer."""

__all__: list[str] = []
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
pytest tests/live/test_package_import.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add pyproject.toml src/sts2_rl/__init__.py src/sts2_rl/live/__init__.py tests/live/test_package_import.py
git commit -m "feat: scaffold rl python package"
```

---

### Task 2: Semantic Live Types

**Files:**
- Create: `src/sts2_rl/live/types.py`
- Modify: `src/sts2_rl/live/__init__.py`
- Create: `tests/live/test_types.py`

- [ ] **Step 1: Write failing tests**

Create `tests/live/test_types.py`:

```python
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
    action = LiveAction(kind="play_card", params={"card_index": 0, "target": "JAW_WORM_0"})

    assert action.kind == "play_card"
    assert action.params["target"] == "JAW_WORM_0"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/live/test_types.py -q
```

Expected: FAIL with import error for `sts2_rl.live.types`.

- [ ] **Step 3: Implement semantic types**

Create `src/sts2_rl/live/types.py`:

```python
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
```

Modify `src/sts2_rl/live/__init__.py`:

```python
"""Live game interaction layer."""

from sts2_rl.live.types import LiveAction, LiveActionResult, LivePlayer, LiveSnapshot

__all__ = [
    "LiveAction",
    "LiveActionResult",
    "LivePlayer",
    "LiveSnapshot",
]
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/live/test_types.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/live/types.py src/sts2_rl/live/__init__.py tests/live/test_types.py
git commit -m "feat: define live interaction types"
```

---

### Task 3: Raw STS2MCP HTTP Client

**Files:**
- Create: `src/sts2_rl/live/sts2mcp_client.py`
- Create: `tests/live/test_sts2mcp_client.py`

- [ ] **Step 1: Write failing tests with a fake HTTP session**

Create `tests/live/test_sts2mcp_client.py`:

```python
from sts2_rl.live.sts2mcp_client import Sts2McpClient


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout):
        self.calls.append(("GET", url, timeout))
        if url.endswith("/"):
            return FakeResponse({"status": "ok", "message": "Hello from STS2 MCP v0.4.0"})
        return FakeResponse({"state_type": "map"})

    def post(self, url, json, timeout):
        self.calls.append(("POST", url, json, timeout))
        return FakeResponse({"status": "ok", "message": "done"})


def test_health_calls_root_endpoint():
    fake = FakeSession()
    client = Sts2McpClient(base_url="http://localhost:15526", session=fake)

    assert client.health()["status"] == "ok"
    assert fake.calls[0][1] == "http://localhost:15526/"


def test_get_singleplayer_state_uses_json_format():
    fake = FakeSession()
    client = Sts2McpClient(base_url="http://localhost:15526", session=fake)

    assert client.get_singleplayer_state()["state_type"] == "map"
    assert fake.calls[0][1] == "http://localhost:15526/api/v1/singleplayer?format=json"


def test_post_singleplayer_action_posts_body():
    fake = FakeSession()
    client = Sts2McpClient(base_url="http://localhost:15526", session=fake)

    result = client.post_singleplayer_action({"action": "end_turn"})

    assert result["status"] == "ok"
    assert fake.calls[0][2] == {"action": "end_turn"}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/live/test_sts2mcp_client.py -q
```

Expected: FAIL with import error for `sts2_rl.live.sts2mcp_client`.

- [ ] **Step 3: Implement client**

Create `src/sts2_rl/live/sts2mcp_client.py`:

```python
from __future__ import annotations

from typing import Any, Protocol

import requests


class HttpSession(Protocol):
    def get(self, url: str, timeout: float):
        raise NotImplementedError

    def post(self, url: str, json: dict[str, Any], timeout: float):
        raise NotImplementedError


class Sts2McpClient:
    def __init__(
        self,
        base_url: str = "http://localhost:15526",
        timeout_s: float = 2.0,
        session: HttpSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def get_singleplayer_state(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/v1/singleplayer?format=json",
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def post_singleplayer_action(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/v1/singleplayer",
            json=body,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/live/test_sts2mcp_client.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/live/sts2mcp_client.py tests/live/test_sts2mcp_client.py
git commit -m "feat: add sts2mcp singleplayer client"
```

---

### Task 4: Raw State Normalizer

**Files:**
- Create: `src/sts2_rl/live/normalizer.py`
- Create: `tests/live/test_normalizer.py`

- [ ] **Step 1: Write failing tests for combat and map states**

Create `tests/live/test_normalizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/live/test_normalizer.py -q
```

Expected: FAIL with import error for `sts2_rl.live.normalizer`.

- [ ] **Step 3: Implement normalizer**

Create `src/sts2_rl/live/normalizer.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/live/test_normalizer.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/live/normalizer.py tests/live/test_normalizer.py
git commit -m "feat: normalize sts2mcp state"
```

---

### Task 5: Live Action Mapper

**Files:**
- Create: `src/sts2_rl/live/action_mapper.py`
- Create: `tests/live/test_action_mapper.py`

- [ ] **Step 1: Write failing tests**

Create `tests/live/test_action_mapper.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/live/test_action_mapper.py -q
```

Expected: FAIL with import error for `sts2_rl.live.action_mapper`.

- [ ] **Step 3: Implement mapper**

Create `src/sts2_rl/live/action_mapper.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/live/test_action_mapper.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/live/action_mapper.py tests/live/test_action_mapper.py
git commit -m "feat: map live actions to sts2mcp"
```

---

### Task 6: Decision-Point Waiter

**Files:**
- Create: `src/sts2_rl/live/wait.py`
- Create: `tests/live/test_wait.py`

- [ ] **Step 1: Write failing tests**

Create `tests/live/test_wait.py`:

```python
from sts2_rl.live.types import LiveSnapshot
from sts2_rl.live.wait import wait_for_decision_point


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_wait_returns_first_decision_point():
    snapshots = [
        LiveSnapshot(state_type="unknown"),
        LiveSnapshot(state_type="map"),
    ]

    def poll():
        return snapshots.pop(0)

    result = wait_for_decision_point(poll, timeout_s=1.0, interval_s=0.1, clock=FakeClock())

    assert result.state_type == "map"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/live/test_wait.py -q
```

Expected: FAIL with import error for `sts2_rl.live.wait`.

- [ ] **Step 3: Implement waiter**

Create `src/sts2_rl/live/wait.py`:

```python
from __future__ import annotations

import time
from typing import Callable, Protocol

from sts2_rl.live.types import LiveSnapshot


class Clock(Protocol):
    def monotonic(self) -> float:
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        raise NotImplementedError


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def wait_for_decision_point(
    poll: Callable[[], LiveSnapshot],
    timeout_s: float = 10.0,
    interval_s: float = 0.1,
    clock: Clock | None = None,
) -> LiveSnapshot:
    active_clock = clock or SystemClock()
    deadline = active_clock.monotonic() + timeout_s
    last = poll()
    while not last.is_decision_point():
        if active_clock.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for decision point; last state={last.state_type}")
        active_clock.sleep(interval_s)
        last = poll()
    return last
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/live/test_wait.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/live/wait.py tests/live/test_wait.py
git commit -m "feat: wait for live decision point"
```

---

### Task 7: High-Level Live Game Client

**Files:**
- Create: `src/sts2_rl/live/game_client.py`
- Modify: `src/sts2_rl/live/__init__.py`
- Create: `tests/live/test_game_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/live/test_game_client.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/live/test_game_client.py -q
```

Expected: FAIL with import error for `sts2_rl.live.game_client`.

- [ ] **Step 3: Implement high-level client**

Create `src/sts2_rl/live/game_client.py`:

```python
from __future__ import annotations

from sts2_rl.live.action_mapper import map_live_action_to_sts2mcp_body
from sts2_rl.live.normalizer import normalize_sts2mcp_state
from sts2_rl.live.sts2mcp_client import Sts2McpClient
from sts2_rl.live.types import LiveAction, LiveActionResult, LiveSnapshot
from sts2_rl.live.wait import wait_for_decision_point


class LiveGameClient:
    def __init__(self, raw_client: Sts2McpClient | None = None) -> None:
        self.raw_client = raw_client or Sts2McpClient()

    def health(self) -> dict:
        return self.raw_client.health()

    def snapshot(self) -> LiveSnapshot:
        return normalize_sts2mcp_state(self.raw_client.get_singleplayer_state())

    def execute(self, action: LiveAction) -> LiveActionResult:
        body = map_live_action_to_sts2mcp_body(action)
        raw = self.raw_client.post_singleplayer_action(body)
        return LiveActionResult(
            status=str(raw.get("status", "error")),
            message=raw.get("message"),
            raw=raw,
        )

    def step(self, action: LiveAction, wait: bool = True) -> LiveSnapshot:
        result = self.execute(action)
        if not result.ok:
            raise RuntimeError(result.message or "live action failed")
        if wait:
            return self.wait_for_decision_point()
        return self.snapshot()

    def wait_for_decision_point(self, timeout_s: float = 10.0) -> LiveSnapshot:
        return wait_for_decision_point(self.snapshot, timeout_s=timeout_s)
```

Modify `src/sts2_rl/live/__init__.py`:

```python
"""Live game interaction layer."""

from sts2_rl.live.game_client import LiveGameClient
from sts2_rl.live.types import LiveAction, LiveActionResult, LivePlayer, LiveSnapshot

__all__ = [
    "LiveAction",
    "LiveActionResult",
    "LiveGameClient",
    "LivePlayer",
    "LiveSnapshot",
]
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/live/test_game_client.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/live/game_client.py src/sts2_rl/live/__init__.py tests/live/test_game_client.py
git commit -m "feat: add live game client"
```

---

### Task 8: Legal Action Derivation From Live Snapshot

**Files:**
- Create: `src/sts2_rl/live/legal_actions.py`
- Create: `tests/live/test_legal_actions.py`

- [ ] **Step 1: Write failing tests**

Create `tests/live/test_legal_actions.py`:

```python
from sts2_rl.live.legal_actions import live_legal_actions
from sts2_rl.live.types import LivePlayer, LiveSnapshot


def test_combat_legal_actions_include_playable_cards_and_end_turn():
    snapshot = LiveSnapshot(
        state_type="monster",
        player=LivePlayer(
            hand=[
                {"index": 0, "id": "StrikeIronclad", "can_play": True, "target_type": "Enemy"},
                {"index": 1, "id": "DefendIronclad", "can_play": True, "target_type": "Self"},
            ]
        ),
        battle={"enemies": [{"entity_id": "JAW_WORM_0", "is_alive": True}]},
    )

    actions = live_legal_actions(snapshot)

    assert any(action.kind == "play_card" and action.params["card_index"] == 0 for action in actions)
    assert any(action.kind == "end_turn" for action in actions)


def test_map_legal_actions_include_next_options():
    snapshot = LiveSnapshot(
        state_type="map",
        choices={"map_nodes": [{"index": 0}, {"index": 1}]},
    )

    actions = live_legal_actions(snapshot)

    assert [action.params["index"] for action in actions] == [0, 1]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/live/test_legal_actions.py -q
```

Expected: FAIL with import error for `sts2_rl.live.legal_actions`.

- [ ] **Step 3: Implement legal action derivation**

Create `src/sts2_rl/live/legal_actions.py`:

```python
from __future__ import annotations

from sts2_rl.live.types import LiveAction, LiveSnapshot


def live_legal_actions(snapshot: LiveSnapshot) -> list[LiveAction]:
    if snapshot.state_type in {"monster", "elite", "boss"}:
        return _combat_actions(snapshot)
    if snapshot.state_type == "map":
        return [LiveAction("choose_map_node", {"index": node["index"]}) for node in snapshot.choices.get("map_nodes", [])]
    if snapshot.state_type == "event":
        return [LiveAction("choose_event_option", {"index": option["index"]}) for option in snapshot.choices.get("event_options", [])]
    if snapshot.state_type == "card_reward":
        actions = [LiveAction("select_card_reward", {"card_index": card["index"]}) for card in snapshot.choices.get("cards", [])]
        actions.append(LiveAction("skip_card_reward"))
        return actions
    if snapshot.state_type == "rewards":
        actions = [LiveAction("claim_reward", {"index": reward["index"]}) for reward in snapshot.choices.get("rewards", [])]
        actions.append(LiveAction("proceed"))
        return actions
    if snapshot.state_type in {"shop", "fake_merchant"}:
        actions = [LiveAction("shop_purchase", {"index": item["index"]}) for item in snapshot.choices.get("shop_items", [])]
        actions.append(LiveAction("proceed"))
        return actions
    if snapshot.state_type == "rest_site":
        return [LiveAction("choose_rest_option", {"index": option["index"]}) for option in snapshot.choices.get("rest_options", [])]
    if snapshot.state_type == "treasure":
        actions = [LiveAction("claim_treasure_relic", {"index": relic["index"]}) for relic in snapshot.choices.get("treasure_relics", [])]
        actions.append(LiveAction("proceed"))
        return actions
    return []


def _combat_actions(snapshot: LiveSnapshot) -> list[LiveAction]:
    actions: list[LiveAction] = []
    player = snapshot.player
    enemies = []
    if snapshot.battle:
        enemies = [enemy for enemy in snapshot.battle.get("enemies", []) if enemy.get("is_alive", True)]
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
                            {"card_index": card["index"], "target": enemy["entity_id"]},
                        )
                    )
            else:
                actions.append(LiveAction("play_card", {"card_index": card["index"]}))
    actions.append(LiveAction("end_turn"))
    return actions
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/live/test_legal_actions.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/live/legal_actions.py tests/live/test_legal_actions.py
git commit -m "feat: derive live legal actions"
```

---

### Task 9: Optional Live Smoke Script

**Files:**
- Create: `scripts/live_smoke.py`
- Create: `tests/live/test_live_smoke.py`

- [ ] **Step 1: Write failing test for argument parsing**

Create `tests/live/test_live_smoke.py`:

```python
from scripts.live_smoke import build_parser


def test_live_smoke_defaults_to_read_only():
    args = build_parser().parse_args([])

    assert args.base_url == "http://localhost:15526"
    assert args.execute_end_turn is False
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
pytest tests/live/test_live_smoke.py -q
```

Expected: FAIL with import error for `scripts.live_smoke`.

- [ ] **Step 3: Implement script**

Create `scripts/live_smoke.py`:

```python
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
```

- [ ] **Step 4: Run test to verify pass**

Run:

```powershell
pytest tests/live/test_live_smoke.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts/live_smoke.py tests/live/test_live_smoke.py
git commit -m "feat: add live bridge smoke script"
```

---

### Task 10: Documentation And Final Verification

**Files:**
- Create: `docs/architecture/live-game-interaction-layer.md`
- Modify: `README.md`

- [ ] **Step 1: Create architecture doc**

Create `docs/architecture/live-game-interaction-layer.md`:

````markdown
# Live Game Interaction Layer

The live interaction layer is policy-independent. It wraps the STS2MCP
singleplayer API and exposes a stable backend contract:

```text
Actual StS2 game
  -> STS2MCP singleplayer API
  -> LiveGameClient
  -> LiveSnapshot / LiveAction
  -> future observation layer and policy
```

The policy should never consume raw STS2MCP JSON. Raw state is normalized before
encoding, and decoded semantic actions are mapped back into STS2MCP POST bodies.

## First Backend

The first backend is `STS2MCP` singleplayer:

- `GET /api/v1/singleplayer?format=json`
- `POST /api/v1/singleplayer`

The interaction layer intentionally ignores multiplayer.

## Safety

Use read-only `scripts/live_smoke.py` first. Only pass `--execute-end-turn` on a
test profile/run because it sends an actual game action.
````

- [ ] **Step 2: Update README**

Add this section to `README.md`:

```markdown
## Live Game Interaction Layer

The live interaction layer wraps the STS2MCP singleplayer mod API and is
independent of the policy/training method. It provides a future bridge from the
real game to the same semantic state/action contract used by the simulator.

See [docs/architecture/live-game-interaction-layer.md](docs/architecture/live-game-interaction-layer.md).
```

- [ ] **Step 3: Run full tests**

Run:

```powershell
pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit docs**

Run:

```powershell
git add README.md docs/architecture/live-game-interaction-layer.md
git commit -m "docs: document live interaction layer"
```

- [ ] **Step 5: Push branch**

Run:

```powershell
git push origin main
```

Expected: `main -> main`.

---

## Final Acceptance Gate

The interaction layer is ready for the next phase when:

- `pytest -q` passes.
- `LiveGameClient.snapshot()` reads mocked STS2MCP state and returns
  `LiveSnapshot`.
- `LiveGameClient.execute()` maps `LiveAction` into the correct singleplayer
  POST body.
- `live_legal_actions()` derives snapshot-local actions for combat, map,
  rewards, card reward, event, rest site, shop, and treasure.
- `scripts/live_smoke.py` can perform a read-only health/state check against a
  running STS2MCP mod.
- No policy code imports raw STS2MCP JSON.
- Multiplayer APIs are not used.

## Build Order Recommendation

Implement in this order:

1. Tasks 1-3: package + raw client.
2. Tasks 4-5: normalizer + action mapper.
3. Tasks 6-8: wait loop + high-level client + legal actions.
4. Tasks 9-10: smoke script + docs.

This creates a useful live backend before any policy exists and keeps the
interaction layer independent from model architecture.
