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

## Implemented Components

- `Sts2McpClient`: raw HTTP wrapper for the STS2MCP singleplayer API.
- `normalize_sts2mcp_state`: converts raw STS2MCP JSON into `LiveSnapshot`.
- `map_live_action_to_sts2mcp_body`: maps `LiveAction` into POST bodies.
- `wait_for_decision_point`: polls across game transitions.
- `LiveGameClient`: high-level `health`, `snapshot`, `execute`, and `step` API.
- `live_legal_actions`: derives snapshot-local `LiveAction` values.

## Safety

Use read-only `scripts/live_smoke.py` first:

```powershell
python scripts/live_smoke.py
```

Only pass `--execute-end-turn` on a test profile/run because it sends an actual
game action:

```powershell
python scripts/live_smoke.py --execute-end-turn
```

## Attribution

The live interaction backend is designed around the singleplayer API exposed by
`STS2MCP`:

- `Gennadiyev/STS2MCP`: https://github.com/Gennadiyev/STS2MCP

StS2 modding references used while designing the layer:

- `Alchyr/ModTemplate-StS2`: https://github.com/Alchyr/ModTemplate-StS2
- `Alchyr/BaseLib-StS2`: https://github.com/Alchyr/BaseLib-StS2

