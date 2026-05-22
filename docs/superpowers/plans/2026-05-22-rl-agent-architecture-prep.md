# RL Agent Architecture Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the RL-side scaffolding and experiment contracts needed to design, train, and compare a Slay the Spire 2 strategy/combat agent while paper review is still in progress.

**Architecture:** Keep the simulator as the source of truth and build a Python RL harness around the native bridge. Start with a shared observation/action interface, deterministic evaluation suites, combat curricula, and baseline policies; leave the final learner choice open until paper notes have been reviewed.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pytest, the simulator Python ctypes binding, JSONL trajectory files, optional TensorBoard or Weights & Biases logging.

---

## Scope

This plan belongs in the future RL repository, `Akirakato1/Slay-the-Spire-2-RL-`. It does not change simulator gameplay. The simulator repo only needs to provide the built native library and Python binding.

The first week should not try to produce a strong agent. It should produce:

- a reliable environment wrapper,
- deterministic train/eval seeds,
- combat archetype fixtures,
- policy/model interface contracts,
- random and heuristic baselines,
- trajectory logging,
- metrics dashboards,
- an architecture decision log that can absorb the paper-reading conclusions.

## Proposed RL Repo Files

- `README.md`: project purpose, setup, simulator dependency, first-run commands.
- `pyproject.toml`: Python package and dev dependencies.
- `src/sts2_rl/env/sim_env.py`: wraps the simulator `Session` and normalizes reset/step outputs.
- `src/sts2_rl/env/action_space.py`: legal-action table helpers and action masking utilities.
- `src/sts2_rl/env/observation.py`: tensor conversion and schema-hash validation.
- `src/sts2_rl/curricula/combat_archetypes.json`: seed/deck/relic/potion archetypes for combat-first training.
- `src/sts2_rl/curricula/combat_curriculum.py`: loads archetypes and builds reset configs.
- `src/sts2_rl/policies/random_policy.py`: legal random baseline.
- `src/sts2_rl/policies/heuristic_policy.py`: simple rule-based baseline for sanity checks.
- `src/sts2_rl/models/interfaces.py`: model input/output dataclasses independent of final architecture.
- `src/sts2_rl/models/candidate_scorer.py`: action-candidate scoring model stub.
- `src/sts2_rl/train/rollout.py`: parallel rollout collection interface.
- `src/sts2_rl/train/trajectory.py`: JSONL trajectory schema and writer.
- `src/sts2_rl/eval/evaluator.py`: deterministic evaluation runner.
- `src/sts2_rl/eval/metrics.py`: win rate, act progress, HP, damage, reward quality, and action entropy metrics.
- `docs/research/paper-reading-matrix.md`: notes and decisions from PPO, GTrXL, IMPALA, Option-Critic, Decision Transformer, AlphaStar, Dominion, Two-Step Card Game RL, and RLCard.
- `docs/architecture/rl-agent-decision-log.md`: selected architecture, rejected alternatives, and reasons.
- `tests/`: pytest tests for all wrappers and contracts.

---

### Task 1: Create Research Decision Matrix

**Files:**
- Create: `docs/research/paper-reading-matrix.md`
- Create: `docs/architecture/rl-agent-decision-log.md`

- [ ] **Step 1: Create the paper matrix**

Create `docs/research/paper-reading-matrix.md` with:

```markdown
# Paper Reading Matrix

| Paper | Read Status | What It Decides For This Project | Keep | Reject | Follow-up Experiment |
|---|---:|---|---|---|---|
| PPO | unread | First online RL optimizer baseline | Candidate first optimizer | Large-scale distributed assumptions | Compare masked PPO against random/heuristic baselines |
| GTrXL | unread | Whether transformer memory should replace LSTM/MLP state encoder | Candidate memory encoder | Full-size architecture before baseline works | Add memory only after environment wrapper is stable |
| IMPALA | unread | Distributed actor/learner scaling plan | Future scaling reference | First implementation dependency | Use only after single-machine PPO bottlenecks |
| Option-Critic | unread | Whether strategy/combat should be options or phase heads | Conceptual guide | Autonomous option discovery as first version | Compare fixed phase router vs learned high-level option |
| Decision Transformer | unread | Offline pretraining from trajectories | Possible imitation/offline phase | Primary first learner | Use if heuristic/self-play trajectory corpus exists |
| AlphaStar | unread | League/self-play, action heads, large-scale architecture | Long-term reference | Large-scale replica | Borrow action-candidate/league ideas selectively |
| Dominion RL | unread | Deck-building representation and variable action sets | Direct deck-building reference | Game-specific assumptions | Use multiset/deck metrics in evaluator |
| Two-Step Card Game RL | unread | Staged combat/strategy training | Direct curriculum reference | Exact multi-agent split until tested | Implement combat-first curriculum |
| RLCard | unread | Environment API and card-game baselines | Interface reference | Multi-agent assumptions not needed now | Compare reset/step conventions |
```

- [ ] **Step 2: Create the decision log**

Create `docs/architecture/rl-agent-decision-log.md` with:

```markdown
# RL Agent Decision Log

## Current Default

Use one shared observation encoder with deterministic phase routing:

- combat phase routes to combat policy head,
- map/reward/shop/event/fireplace/ancient phases route to strategy policy head,
- both heads score the current legal action candidates,
- value prediction starts shared, then may split into combat and strategy values after baselines.

## Decisions Pending Paper Review

1. PPO versus another first optimizer.
2. MLP/set encoder versus transformer encoder for v0.
3. Transformer memory now versus after baseline.
4. Fixed strategy/combat phase router versus learned option selector.
5. Online-only training versus offline trajectory pretraining.

## Guardrails

- The simulator observation schema is the source of truth.
- Policies may only select actions present in `legal_actions`.
- Every experiment records simulator version, schema hash, seed, character, ascension, and git commit.
- No architecture is accepted without beating random and heuristic baselines on fixed seeds.
```

- [ ] **Step 3: Commit**

Run:

```powershell
git add docs/research/paper-reading-matrix.md docs/architecture/rl-agent-decision-log.md
git commit -m "docs: add rl research decision matrix"
```

Expected: a docs-only commit.

---

### Task 2: Build Environment Wrapper Contract

**Files:**
- Create: `src/sts2_rl/env/sim_env.py`
- Create: `tests/env/test_sim_env.py`

- [ ] **Step 1: Write failing tests**

Create `tests/env/test_sim_env.py`:

```python
from sts2_rl.env.sim_env import Sts2Env


def test_reset_with_fixed_seed_is_reproducible():
    first = Sts2Env(character="Ironclad", ascension=0)
    second = Sts2Env(character="Ironclad", ascension=0)

    obs_a = first.reset(seed="SAME_SEED")
    obs_b = second.reset(seed="SAME_SEED")

    assert obs_a.schema_hash == obs_b.schema_hash
    assert obs_a.features == obs_b.features
    assert obs_a.legal_actions


def test_reset_without_seed_exposes_generated_seed():
    env = Sts2Env(character="Ironclad", ascension=0)

    obs = env.reset()

    assert len(env.seed) == 8
    assert all(ch.isupper() or ch.isdigit() for ch in env.seed)
    assert obs.legal_actions
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/env/test_sim_env.py -q
```

Expected: failure because `sts2_rl.env.sim_env` does not exist.

- [ ] **Step 3: Implement minimal wrapper**

Create `src/sts2_rl/env/sim_env.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sts2_sim_rl import Session


@dataclass(frozen=True)
class EnvObservation:
    schema_hash: str
    features: list[float]
    objects: list[dict[str, Any]]
    action_mask: list[float]
    legal_actions: list[dict[str, Any]]


class Sts2Env:
    def __init__(self, character: str = "Ironclad", ascension: int = 0) -> None:
        self.character = character
        self.ascension = ascension
        self.seed: str | None = None
        self._session: Session | None = None

    def reset(self, seed: str | None = None) -> EnvObservation:
        self.close()
        self._session = Session(
            seed=seed,
            character=self.character,
            ascension=self.ascension,
        )
        self.seed = self._session.seed
        return self._convert(self._session.observation())

    def step(self, action_index: int) -> EnvObservation:
        if self._session is None:
            raise RuntimeError("reset must be called before step")
        return self._convert(self._session.step(action_index))

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _convert(self, raw: dict[str, Any]) -> EnvObservation:
        return EnvObservation(
            schema_hash=raw["schema_hash"],
            features=raw["features"],
            objects=raw["objects"],
            action_mask=raw["action_mask"],
            legal_actions=raw["legal_actions"],
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/env/test_sim_env.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/env/sim_env.py tests/env/test_sim_env.py
git commit -m "feat: add simulator environment wrapper"
```

---

### Task 3: Add Action Candidate Utilities

**Files:**
- Create: `src/sts2_rl/env/action_space.py`
- Create: `tests/env/test_action_space.py`

- [ ] **Step 1: Write failing tests**

Create `tests/env/test_action_space.py`:

```python
from sts2_rl.env.action_space import legal_action_indices, choose_random_legal_action


def test_legal_action_indices_reads_current_table():
    actions = [
        {"action_index": 4, "head": "map", "label": "a"},
        {"action_index": 9, "head": "map", "label": "b"},
    ]

    assert legal_action_indices(actions) == [4, 9]


def test_random_legal_action_returns_table_member():
    actions = [
        {"action_index": 4, "head": "map", "label": "a"},
        {"action_index": 9, "head": "map", "label": "b"},
    ]

    assert choose_random_legal_action(actions, seed=123) in [4, 9]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/env/test_action_space.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement utilities**

Create `src/sts2_rl/env/action_space.py`:

```python
from __future__ import annotations

import random
from typing import Any


def legal_action_indices(legal_actions: list[dict[str, Any]]) -> list[int]:
    return [int(action["action_index"]) for action in legal_actions]


def choose_random_legal_action(
    legal_actions: list[dict[str, Any]],
    seed: int | None = None,
) -> int:
    indices = legal_action_indices(legal_actions)
    if not indices:
        raise ValueError("cannot choose from an empty legal action table")
    rng = random.Random(seed)
    return rng.choice(indices)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/env/test_action_space.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/env/action_space.py tests/env/test_action_space.py
git commit -m "feat: add legal action utilities"
```

---

### Task 4: Add Baseline Policies

**Files:**
- Create: `src/sts2_rl/policies/random_policy.py`
- Create: `src/sts2_rl/policies/heuristic_policy.py`
- Create: `tests/policies/test_baselines.py`

- [ ] **Step 1: Write failing tests**

Create `tests/policies/test_baselines.py`:

```python
from sts2_rl.env.sim_env import EnvObservation
from sts2_rl.policies.heuristic_policy import HeuristicPolicy
from sts2_rl.policies.random_policy import RandomPolicy


def fake_obs(labels):
    return EnvObservation(
        schema_hash="test",
        features=[],
        objects=[],
        action_mask=[1.0 for _ in labels],
        legal_actions=[
            {"action_index": idx, "head": "test", "label": label}
            for idx, label in enumerate(labels)
        ],
    )


def test_random_policy_selects_legal_action():
    obs = fake_obs(["End turn", "Strike"])

    action = RandomPolicy(seed=7).act(obs)

    assert action in [0, 1]


def test_heuristic_policy_prefers_non_end_turn_action():
    obs = fake_obs(["End turn", "Strike"])

    assert HeuristicPolicy().act(obs) == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/policies/test_baselines.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement random policy**

Create `src/sts2_rl/policies/random_policy.py`:

```python
from __future__ import annotations

import random

from sts2_rl.env.sim_env import EnvObservation


class RandomPolicy:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: EnvObservation) -> int:
        if not observation.legal_actions:
            raise ValueError("no legal actions available")
        return int(self._rng.choice(observation.legal_actions)["action_index"])
```

- [ ] **Step 4: Implement heuristic policy**

Create `src/sts2_rl/policies/heuristic_policy.py`:

```python
from __future__ import annotations

from sts2_rl.env.sim_env import EnvObservation


class HeuristicPolicy:
    def act(self, observation: EnvObservation) -> int:
        if not observation.legal_actions:
            raise ValueError("no legal actions available")
        for action in observation.legal_actions:
            label = str(action.get("label", "")).lower()
            if "end turn" not in label and "skip" not in label:
                return int(action["action_index"])
        return int(observation.legal_actions[0]["action_index"])
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```powershell
pytest tests/policies/test_baselines.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/sts2_rl/policies/random_policy.py src/sts2_rl/policies/heuristic_policy.py tests/policies/test_baselines.py
git commit -m "feat: add baseline policies"
```

---

### Task 5: Add Combat Archetype Curriculum Manifest

**Files:**
- Create: `src/sts2_rl/curricula/combat_archetypes.json`
- Create: `src/sts2_rl/curricula/combat_curriculum.py`
- Create: `tests/curricula/test_combat_curriculum.py`

- [ ] **Step 1: Write failing tests**

Create `tests/curricula/test_combat_curriculum.py`:

```python
from sts2_rl.curricula.combat_curriculum import load_combat_archetypes


def test_combat_archetypes_have_unique_names():
    archetypes = load_combat_archetypes()
    names = [item.name for item in archetypes]

    assert len(names) == len(set(names))
    assert "ironclad_starter" in names


def test_each_archetype_has_replay_seed_and_deck():
    for archetype in load_combat_archetypes():
        assert archetype.seed
        assert archetype.deck
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/curricula/test_combat_curriculum.py -q
```

Expected: import failure.

- [ ] **Step 3: Create starter archetype manifest**

Create `src/sts2_rl/curricula/combat_archetypes.json`:

```json
[
  {
    "name": "ironclad_starter",
    "character": "Ironclad",
    "ascension": 0,
    "seed": "IRONCLAD",
    "deck": [
      "StrikeIronclad",
      "StrikeIronclad",
      "StrikeIronclad",
      "StrikeIronclad",
      "StrikeIronclad",
      "DefendIronclad",
      "DefendIronclad",
      "DefendIronclad",
      "DefendIronclad",
      "Bash"
    ],
    "relics": ["BurningBlood"],
    "potions": []
  },
  {
    "name": "attack_damage",
    "character": "Ironclad",
    "ascension": 0,
    "seed": "ATTACK01",
    "deck": ["StrikeIronclad", "StrikeIronclad", "Bash", "PommelStrike", "Anger"],
    "relics": ["BurningBlood"],
    "potions": ["FirePotion"]
  },
  {
    "name": "defense_stall",
    "character": "Ironclad",
    "ascension": 0,
    "seed": "BLOCK001",
    "deck": ["DefendIronclad", "DefendIronclad", "ShrugItOff", "Entrench", "StrikeIronclad"],
    "relics": ["BurningBlood"],
    "potions": ["BlockPotion"]
  }
]
```

- [ ] **Step 4: Implement loader**

Create `src/sts2_rl/curricula/combat_curriculum.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CombatArchetype:
    name: str
    character: str
    ascension: int
    seed: str
    deck: list[str]
    relics: list[str]
    potions: list[str]


def load_combat_archetypes() -> list[CombatArchetype]:
    path = Path(__file__).with_name("combat_archetypes.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [CombatArchetype(**item) for item in raw]
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```powershell
pytest tests/curricula/test_combat_curriculum.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/sts2_rl/curricula/combat_archetypes.json src/sts2_rl/curricula/combat_curriculum.py tests/curricula/test_combat_curriculum.py
git commit -m "feat: add combat archetype curriculum manifest"
```

---

### Task 6: Add Trajectory Logging Contract

**Files:**
- Create: `src/sts2_rl/train/trajectory.py`
- Create: `tests/train/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/train/test_trajectory.py`:

```python
import json

from sts2_rl.train.trajectory import TrajectoryWriter


def test_trajectory_writer_records_seed_schema_action_and_reward(tmp_path):
    path = tmp_path / "rollout.jsonl"
    writer = TrajectoryWriter(path)

    writer.write_step(
        run_id="run-1",
        seed="FIXED001",
        schema_hash="schema",
        timestep=0,
        action_index=3,
        reward=0.25,
        done=False,
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["seed"] == "FIXED001"
    assert row["schema_hash"] == "schema"
    assert row["action_index"] == 3
    assert row["reward"] == 0.25
    assert row["done"] is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/train/test_trajectory.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement writer**

Create `src/sts2_rl/train/trajectory.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


class TrajectoryWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_step(
        self,
        run_id: str,
        seed: str,
        schema_hash: str,
        timestep: int,
        action_index: int,
        reward: float,
        done: bool,
    ) -> None:
        row = {
            "run_id": run_id,
            "seed": seed,
            "schema_hash": schema_hash,
            "timestep": timestep,
            "action_index": action_index,
            "reward": reward,
            "done": done,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/train/test_trajectory.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/train/trajectory.py tests/train/test_trajectory.py
git commit -m "feat: add trajectory logging contract"
```

---

### Task 7: Add Deterministic Evaluation Runner

**Files:**
- Create: `src/sts2_rl/eval/evaluator.py`
- Create: `src/sts2_rl/eval/metrics.py`
- Create: `tests/eval/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/eval/test_evaluator.py`:

```python
from sts2_rl.eval.evaluator import EvaluationResult, summarize_results


def test_summarize_results_counts_terminal_outcomes():
    results = [
        EvaluationResult(seed="A", steps=10, terminal="Victory"),
        EvaluationResult(seed="B", steps=12, terminal="Defeat"),
        EvaluationResult(seed="C", steps=9, terminal="Running"),
    ]

    summary = summarize_results(results)

    assert summary["runs"] == 3
    assert summary["victories"] == 1
    assert summary["defeats"] == 1
    assert summary["unfinished"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/eval/test_evaluator.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement evaluator summary**

Create `src/sts2_rl/eval/evaluator.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    seed: str
    steps: int
    terminal: str


def summarize_results(results: list[EvaluationResult]) -> dict[str, int]:
    return {
        "runs": len(results),
        "victories": sum(1 for item in results if item.terminal == "Victory"),
        "defeats": sum(1 for item in results if item.terminal == "Defeat"),
        "unfinished": sum(1 for item in results if item.terminal == "Running"),
    }
```

Create `src/sts2_rl/eval/metrics.py`:

```python
from __future__ import annotations


def win_rate(victories: int, runs: int) -> float:
    if runs == 0:
        return 0.0
    return victories / runs
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/eval/test_evaluator.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/eval/evaluator.py src/sts2_rl/eval/metrics.py tests/eval/test_evaluator.py
git commit -m "feat: add deterministic evaluation summary"
```

---

### Task 8: Define Model Interface Before Model Architecture

**Files:**
- Create: `src/sts2_rl/models/interfaces.py`
- Create: `tests/models/test_interfaces.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_interfaces.py`:

```python
from sts2_rl.models.interfaces import PolicyBatch, PolicyOutput


def test_policy_output_selects_argmax_legal_candidate():
    output = PolicyOutput(action_indices=[4, 9], logits=[0.1, 0.9], value=0.5)

    assert output.greedy_action_index() == 9


def test_policy_batch_preserves_schema_hash():
    batch = PolicyBatch(schema_hash="abc", features=[[0.0]], action_features=[[[1.0]]])

    assert batch.schema_hash == "abc"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/models/test_interfaces.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement interfaces**

Create `src/sts2_rl/models/interfaces.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyBatch:
    schema_hash: str
    features: list[list[float]]
    action_features: list[list[list[float]]]


@dataclass(frozen=True)
class PolicyOutput:
    action_indices: list[int]
    logits: list[float]
    value: float

    def greedy_action_index(self) -> int:
        if not self.action_indices:
            raise ValueError("no candidate actions")
        best = max(range(len(self.logits)), key=lambda idx: self.logits[idx])
        return self.action_indices[best]
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/models/test_interfaces.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sts2_rl/models/interfaces.py tests/models/test_interfaces.py
git commit -m "feat: define policy model interfaces"
```

---

## Paper Review Checkpoints

During the week of reading, update `docs/research/paper-reading-matrix.md` after each paper with one concrete project decision.

Recommended order:

1. Two-Step RL for Multistage Strategy Card Game.
2. PPO.
3. GTrXL.
4. Dominion RL.
5. Option-Critic.
6. IMPALA.
7. Decision Transformer.
8. AlphaStar.
9. RLCard.

At the end of the week, update `docs/architecture/rl-agent-decision-log.md` with:

- first optimizer,
- first model encoder,
- whether memory is in v0,
- whether combat/strategy are fixed phase heads or learned options,
- first evaluation gate that must be beaten.

## First Architecture Recommendation Before Paper Review

Use this as the default until evidence changes it:

```text
EncodedObservation
  -> object/global feature encoder
  -> shared state embedding
  -> candidate action encoder
  -> phase router
      -> combat candidate-scoring head
      -> strategy candidate-scoring head
  -> value head
```

Training sequence:

1. Train/evaluate random and heuristic policies.
2. Train combat-only suites from archetype fixtures.
3. Train strategy decisions with combat policy fixed or slow-updated.
4. Joint train full runs.
5. Add memory/transformer only after baseline metrics are stable.

## Verification Gates

Before real training begins:

```powershell
pytest -q
python -m sts2_rl.eval.run_baselines --seeds seeds/eval_100.txt
```

Expected:

- all tests pass,
- random baseline produces valid full-run trajectories,
- heuristic baseline produces valid full-run trajectories,
- every trajectory records seed and schema hash,
- rerunning the same fixed seed and policy gives the same action sequence.

