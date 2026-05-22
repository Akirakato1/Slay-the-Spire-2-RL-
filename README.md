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
