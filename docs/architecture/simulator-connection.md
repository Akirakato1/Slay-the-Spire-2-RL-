# Simulator Connection

The RL repo uses the simulator repo as its environment backend.

- GitHub: https://github.com/Akirakato1/Slay-the-Spire-2-Simulator
- Local development path used by this workspace: `../sim`
- Patch target: Slay the Spire 2 patch `103.2`

## Boundary

The simulator repo owns:

- extracted patch data,
- gameplay implementation,
- Rust play API,
- observation encoding and decoding,
- native C ABI,
- sample Python ctypes binding.

The RL repo owns:

- environment wrapper around the simulator binding,
- action selection policies,
- model architecture,
- rollout collection,
- training loops,
- evaluation suites,
- experiment tracking and research notes.

## Bridge Contract

The RL side should call the simulator through the native/Python bridge:

```text
RL policy -> Python wrapper -> C ABI -> observation bridge -> play API -> simulator
```

Each rollout must record:

- simulator repo commit,
- observation `schema_hash`,
- seed,
- character,
- ascension,
- policy/model version.

Fixed seeds are used for reproducible evaluation. Missing seeds may be used for
fresh training rollouts, but the resolved seed must be stored so the rollout can
be replayed.

