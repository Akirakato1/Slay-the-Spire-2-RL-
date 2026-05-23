# Slay the Spire 2 RL

RL architecture, training, evaluation, and experiment tooling for **Slay the
Spire 2 patch 103.2**.

This repo depends on the simulator repo as the environment source of truth:

- Simulator: [Akirakato1/Slay-the-Spire-2-Simulator](https://github.com/Akirakato1/Slay-the-Spire-2-Simulator)
- Local simulator checkout used during setup: `../sim`

The simulator owns gameplay correctness, extracted game data, the Rust play API,
the observation encoder/decoder, the native C ABI, and the Python ctypes sample.
This RL repo should own agent architecture, rollouts, training, evaluation,
curricula, experiment logs, and paper-driven design decisions.

## Intended Runtime Path

```text
Python RL trainer
  -> simulator Python binding
  -> simulator C ABI
  -> Rust observation bridge
  -> Rust play API
  -> simulator
```

The RL policy should consume encoded observations and select only action indices
from the current legal action table.

## Simulator Dependency

Build the simulator native library from the simulator repo:

```powershell
cd ..\sim
cargo build -p sts2-sim
$env:STS2_SIM_LIB = (Resolve-Path target\debug\sts2_sim.dll).Path
python -m unittest discover -s bindings\python\tests -v
```

Then use that `STS2_SIM_LIB` path from this repo while developing the RL wrapper
and training harness.

Fixed seeds give deterministic controlled runs. Omitting a seed through the
simulator bridge generates a valid replay seed, which must be recorded with each
trajectory.

## Live Game Interaction Layer

The live interaction layer is policy-independent. It wraps the STS2MCP
singleplayer mod API and provides a future bridge from the actual game to the
same semantic state/action contract used by the simulator:

```text
Actual StS2 game
  -> STS2MCP singleplayer API
  -> LiveGameClient
  -> LiveSnapshot / LiveAction
  -> observation layer and policy
```

Implemented live-layer modules live under `src/sts2_rl/live/`. The first backend
uses `GET /api/v1/singleplayer?format=json` for snapshots and
`POST /api/v1/singleplayer` for actions. Multiplayer is intentionally out of
scope.

See [docs/architecture/live-game-interaction-layer.md](docs/architecture/live-game-interaction-layer.md).

Attribution: this layer is designed around the singleplayer API exposed by
[Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP). StS2 modding
references used during design include
[Alchyr/ModTemplate-StS2](https://github.com/Alchyr/ModTemplate-StS2) and
[Alchyr/BaseLib-StS2](https://github.com/Alchyr/BaseLib-StS2).

## Current Planning Docs

- [RL architecture prep plan](docs/superpowers/plans/2026-05-22-rl-agent-architecture-prep.md)
- [RL agent reading list](docs/research/rl-agent-reading-list.md)

## First Architecture Assumption

Start simple and keep the model contract flexible:

```text
EncodedObservation
  -> shared observation encoder
  -> candidate action encoder
  -> phase router
      -> combat policy head
      -> strategy policy head
  -> value head
```

The near-term goal is not a strong agent. It is a trustworthy RL harness:

- deterministic simulator reset/step wrapper,
- random and heuristic baselines,
- combat archetype curriculum fixtures,
- trajectory logging with schema hash and seed,
- deterministic evaluation suites,
- paper-reading decision log.

## Reading Before Final Architecture

The initial paper list is in
[docs/research/rl-agent-reading-list.md](docs/research/rl-agent-reading-list.md).
Architecture choices should be recorded in a decision log before the first
serious training run.
