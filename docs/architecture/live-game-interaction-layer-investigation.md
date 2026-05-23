# Live Game Interaction Layer Investigation

Date: 2026-05-23

## Summary

Slay the Spire 2 live-game integration is feasible through a game mod. The most
promising path is a small local bridge mod that exposes current game state and
accepts action commands through localhost HTTP or another local IPC channel.

There is already a working reference implementation:

- `Gennadiyev/STS2MCP`: https://github.com/Gennadiyev/STS2MCP

`STS2MCP` exposes state and actions over `localhost:15526`, supports
singleplayer and multiplayer, and was tested against StS2 `v0.103.2`.

## Relevant Modding Facts

Current StS2 modding is C#/.NET/Godot based.

- Mods can be built with `.NET SDK`.
- Community templates exist through `Alchyr.Sts2.Templates`.
- BaseLib is the common modding support library.
- Mods are loaded from the game's `mods` directory.
- A mod usually ships as a manifest `.json`, optional code `.dll`, and optional
  asset `.pck`.

Useful references:

- Mod template setup: https://github-wiki-see.page/m/Alchyr/ModTemplate-StS2/wiki/Setup
- Modding basics: https://github-wiki-see.page/m/Alchyr/ModTemplate-StS2/wiki/Modding-Basics
- BaseLib: https://github.com/Alchyr/BaseLib-StS2

## Existing Bridge Reference: STS2MCP

`STS2MCP` is directly relevant because it already does the core job of the
future interaction layer.

Documented endpoints:

- `GET /api/v1/singleplayer`: read current singleplayer state.
- `POST /api/v1/singleplayer`: perform a singleplayer action.
- `GET /api/v1/multiplayer`: read multiplayer state.
- `POST /api/v1/multiplayer`: perform multiplayer action.
- `GET /api/v1/profile`: read profile progress.
- `GET /api/v1/compendium`: read profile progress grouped like the Compendium.
- `GET /api/v1/wiki`: fuzzy-search discovered card/relic entries.
- `GET /api/v1/profiles`: list profile slots.
- `POST /api/v1/profiles`: switch or delete profile slots.

The singleplayer state endpoint returns:

- `state_type`,
- `run` fields such as act, floor, and ascension,
- `player` fields such as character, HP, gold, relics, potions, and, during
  combat, energy, hand, piles, and orbs.

Supported state/action families include:

- combat: `play_card`, `use_potion`, `discard_potion`, `end_turn`,
- combat card-selection prompts: `combat_select_card`,
  `combat_confirm_selection`,
- rewards: `claim_reward`, `proceed`,
- card reward: `select_card_reward`, `skip_card_reward`,
- map: `choose_map_node`,
- event: `choose_event_option`, `advance_dialogue`,
- rest site: `choose_rest_option`, `proceed`,
- shop: `shop_purchase`, `proceed`,
- treasure: `claim_treasure_relic`, `proceed`,
- card selection overlays: `select_card`, `confirm_selection`,
  `cancel_selection`,
- bundle/relic/crystal-sphere special screens,
- menu/profile/game-over controls.

## Recommended Project Direction

Start by treating `STS2MCP` as the reference backend and design our RL-side
interaction layer as an adapter interface. Do not train the policy directly
against `STS2MCP` JSON. Instead:

```text
STS2MCP JSON
  -> LiveStateNormalizer
  -> simulator-like semantic observation
  -> observation layer encoder
  -> trained RL policy
  -> action index decoder
  -> LiveActionMapper
  -> STS2MCP POST action
```

This keeps the trained policy attached to our observation/action contract, not
to one community mod's JSON shape.

## Proposed Interface

The RL repo should eventually expose:

```python
class LiveGameClient:
    def health(self) -> LiveBridgeInfo: ...
    def snapshot(self) -> LiveSnapshot: ...
    def execute(self, action: LiveAction) -> LiveActionResult: ...
    def wait_for_decision_point(self, timeout_s: float = 10.0) -> LiveSnapshot: ...
```

Important supporting components:

- `LiveVersionManager`: checks game/mod version and whether extracted data and
  observation schema exist for that version.
- `LiveStateNormalizer`: converts raw mod JSON into simulator-compatible
  semantic state.
- `LiveActionMapper`: converts decoded policy actions into mod API POST bodies.
- `LiveTransitionWatcher`: polls until the game reaches a stable decision point
  after an action.
- `LiveResumeBootstrapper`: reconstructs full run context when attaching to an
  already-started run.

## Startup Flow

```text
connect to localhost bridge
  -> read bridge health/version
  -> read game/profile/current-run state
  -> detect or request game version
  -> ensure extracted data exists for version
  -> load/build observation schema
  -> verify model schema hash compatibility
  -> snapshot current game state
  -> normalize to observation input
  -> start policy loop
```

## Resume Flow

When attaching to a run already in progress, the agent cannot trust any prior
memory. It must bootstrap from live state:

- character,
- ascension,
- act/floor,
- map and current position,
- deck, upgrades, and enchantments if available,
- relics and counters,
- potions and potion slots,
- HP/max HP, gold,
- current screen/state type,
- event/reward/shop/rest/treasure choices if active,
- combat hand/draw/discard/exhaust piles if in combat,
- monster IDs, HP/block/statuses/intents,
- player statuses, energy, orbs, pets/summons if present.

Any simulator observation feature that cannot be recovered from this snapshot is
not live-compatible and should be hidden from the policy before final training.

## Action Mapping Sketch

| Simulator/Observation Action | Live Bridge Action |
|---|---|
| play combat card with target | `play_card` with `card_index`, optional `target` |
| use potion | `use_potion` with `slot`, optional `target` |
| end turn | `end_turn` |
| choose map point | `choose_map_node` with index from `next_options` |
| choose event option | `choose_event_option` with option index |
| choose rest option | `choose_rest_option` with option index |
| buy shop item | `shop_purchase` with item index |
| claim reward | `claim_reward` with reward index |
| choose card reward | `select_card_reward` with card index |
| skip card reward | `skip_card_reward` |
| proceed from resolved screen | `proceed` |
| deck/card selection choice | `select_card`, then maybe `confirm_selection` |

The adapter must resolve simulator-style object IDs/actions into the live
bridge's current visible indices. This is a key risk area: after every snapshot,
the adapter must preserve a mapping from policy legal action index to the exact
live bridge command that is valid for that snapshot.

## Risks

1. **Version drift:** StS2 early access updates often break mods and BaseLib.
   Keep live-game support version-gated.
2. **JSON shape drift:** Wrap `STS2MCP` behind our own adapter instead of binding
   the RL policy directly to its raw state.
3. **Hidden simulator state:** Audit final training observations for
   live-observable parity.
4. **Timing/state transitions:** Actions need `wait_for_decision_point`, not
   immediate re-snapshot assumptions.
5. **Indexes versus IDs:** The live bridge often acts by visible indices; our
   observation layer must retain a snapshot-local action mapping.
6. **Safety:** Localhost control APIs should stay loopback-only, no remote bind,
   no auth secrets in logs, and only run on test profiles/runs.

## Recommendation

Do not build the final live interaction layer yet. During RL training, keep
training against the simulator.

After a trained policy exists:

1. Build a thin `STS2MCPClient` in the RL repo.
2. Implement `LiveStateNormalizer` for a small set of states: combat, map,
   rewards, card reward, event, rest site, shop.
3. Run read-only live snapshot parity tests against simulator semantic
   observations.
4. Implement action execution for one state family at a time.
5. Only then decide whether to keep depending on `STS2MCP`, fork it, or write a
   smaller RL-specific C# mod using the same Godot/.NET approach.

