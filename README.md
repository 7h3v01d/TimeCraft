# TimeCraft v1.6

A 3D infinite voxel sandbox built with Python, pyglet, and OpenGL 3.3 core.
No external game engine — everything from terrain generation to mob AI is hand-rolled.

---

## Quick start

```bat
setup.bat       # create venv + install dependencies (first run only)
run.bat         # launch the game
test.bat        # run the test suite (497 tests)
```

Requires Python 3.11+ and a GPU supporting OpenGL 3.3 core.

---

## Controls

| Key / Mouse | Action |
|---|---|
| W A S D | Move |
| Mouse | Look |
| Left click | Break block |
| Right click | Place block |
| Scroll wheel | Cycle hotbar |
| 1 – 9 | Select hotbar slot |
| Space | Jump |
| Left Shift | Crouch |
| R | Sprint |
| Tab | Toggle fly mode |
| G | Fire wormhole gun (portal A → B) |
| F5 | Save world |
| F6 | New world |
| Escape | Release mouse |

---

## Features

### World
- **Infinite procedural terrain** — chunk-based generation with sector eviction; only loaded terrain kept in memory
- **6 biomes** — Tundra, Taiga, Plains, Forest, Desert, Savanna; driven by independent temperature × moisture noise passes
- **15 block types** — Grass, Dirt, Sand, Stone, Brick, Wood, Planks, Leaf, Gravel, Glass, Water, Magic Water, Crystal, Snow, Portal
- **Binary save format** — `.tcw` file, magic `TCWF`, 7 bytes per block; ~5× smaller than JSON; auto-migrates legacy saves
- **Frustum culling** — Gribb/Hartmann plane extraction; ~59% of sectors culled per frame

### Visuals
- **Ambient occlusion** — per-vertex, baked into VBO at chunk generation time
- **Day/night cycle** — sky colour lerps through 4 keypoints (dawn/noon/dusk/midnight); `sun_brightness` uniform drives block lighting
- **Sky objects** — sun disc with halo, moon (opposite phase), 200 stars (fade in at dusk), 28 fluffy cloud clusters (multi-puff alpha-stacked octagons, drifting)
- **Weather system** — rain and snow particle emitters; biome-driven (snow in tundra/taiga, rain in forest/plains/savanna, clear in desert); linear fog shrinks draw distance in heavy weather
- **Fog** — depth-based in default shader; `fog_end` shrinks from 105 blocks (clear) to 18 (peak rain)
- **Water shader** — animated vertex displacement + shimmer via separate GLSL pair
- **Block highlight** — wireframe outline on aimed block (dedicated minimal shader, drawn in 3D pass)
- **Extended draw distance** — `z_far=120`, render distance 6 sectors (96 blocks), frustum culling keeps it responsive

### HUD
- **Inventory hotbar** — 9 slots, atlas texture icons, scroll wheel cycling, white border highlight
- **Minimap** — 120×120 px bottom-right; top_surface dict makes rebuild O(pixels) with no y-scan; rate-limited to 0.5s
- **Portal compass** — screen-edge bearing indicators for each active portal end; shows label + distance; relative to player facing
- **Crosshair** — gapped, outlined, always visible against any background
- **Status toasts** — centre-screen messages for portal events, saves, wormhole travel

### Gameplay
- **Wormhole gun** — press G to fire; first shot = portal A (blue block), second = portal B (orange block); walk through either end to teleport; 30s timer or permanent mode; one pair at a time
- **Block particles** — per-block RGBA burst on break, with gravity and alpha fade
- **Block sounds** — 14 procedurally synthesised wavs (stone, dirt, wood, glass, leaf, water, sand); generated on first launch, cached in `sounds/`
- **Physics** — AABB collision (shared `util.collide()`), gravity, jump, sprint, crouch, fly; portal blocks are walk-through

### Mobs
- **Entity system** — `Entity` base class with AABB physics; `Mob` subclass with IDLE/WALK state machine wander AI; `MobManager` handles spawning, despawning, rate-limited physics
- **Chicken** — yellow pixelart sprite, spawns on grass/sand, quick and jittery
- **Sheep** — white woolly sprite, spawns on grass/snow, slow and calm
- **Textured billboards** — dedicated mob shader samples `mob_atlas.png` (128×64 RGBA sprite sheet); nearest-neighbour filtering for crisp pixels; leg quads animated with `sin(game_time)` walk cycle; `brightness` uniform matches day/night
- **Spawning** — uses `top_surface` dict for O(1) surface lookup; spawn ring 6–48 blocks from player; biome-appropriate surfaces only

---

## Architecture

```
timecraft/
├── main.py           entry point, splash screen, GL setup
├── window.py         pyglet Window subclass — input, camera, draw pipeline
├── model.py          world state, chunk gen, sectors, particles, weather, save/load
├── config.py         all constants — physics, biomes, weather, mobs, sky, fog
├── noise_gen.py      seeded cosine-interpolated noise + get_climate()
├── util.py           cube_vertices, compute_ao, frustum culling, collide(), sectorize()
├── sky.py            SkyRenderer — sun, moon, stars, fluffy clouds
├── mobs.py           Entity, Mob, MobManager — AI, physics, spawning
├── mob_renderer.py   MobRenderer — textured billboard shader + walk animation
├── sounds.py         procedural wav synthesis, SoundManager (14 sounds)
├── texture.png       4×4 block atlas (256×256, 64px tiles)
├── mob_atlas.png     mob sprite sheet (128×64, chicken + sheep)
├── water_vertex.glsl animated water vertex shader
├── water_fragment.glsl water shimmer fragment shader
└── test_timecraft.py 497 tests
```

---

## Test suite

```bat
test.bat
```

497 passing, 1 skipped. The skipped test is a seed-dependent desert spawn search — valid behaviour, not a bug.

---

## What's next

- **Crafting** — combine hotbar blocks with C; config recipe dict
- **Night hostile mobs** — spawns at `sun_brightness < 0.3`, despawns at dawn
- **Chest storage** — right-click chest block opens HUD inventory; persists in binary save
- **More mob types** — cow, pig, hostile zombie
- **Multiplayer** — asyncio client/server, block-delta sync

