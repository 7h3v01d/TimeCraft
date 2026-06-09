# 🌍 TimeCraft v0.7

A 3D voxel sandbox engine built in Python with **pyglet** and **OpenGL 3.3 core** — procedurally generated worlds, custom GLSL shaders, real physics, ambient occlusion, a day/night cycle, and a working save/load system.

This started as a learning project and turned into something genuinely playable. Pulled back out of archive and actively developed.

---

## 🧠 What is TimeCraft?

TimeCraft is a first-person voxel world — think Minecraft-lite, built from scratch. You spawn into a procedurally generated landscape, walk around, build, and destroy. The world persists between sessions via JSON save files, water moves and shimmers, and the sun rises and sets.

The design goal was to understand every layer of a 3D engine: geometry batching, UV-mapped texture atlases, custom GLSL shaders, sector-based chunk visibility, frustum culling, collision detection, and a deterministic tick loop. No game engine. No shortcuts.

---

## ✨ Features

**World generation**
- 128×128 seeded terrain via a custom cosine-interpolated noise generator (`NoiseGen`)
- Biome-like surface rules: sand near water edges, grass/dirt mid-range, gravel beaches, snow-capped peaks above y=30
- Procedural tree generation with wood trunks and 3D leaf canopies
- Rare crystal formations spawning on high terrain
- Water fills low areas; rare glowing magic water spawns randomly

**Rendering**
- OpenGL 3.3 core profile — no legacy pipeline
- Dual shader system: a standard block shader and a dedicated animated water shader
- Water surface animates via sine-wave vertex displacement in GLSL (`water_vertex.glsl`)
- Fragment shimmer effect on all water; extra glow pulse on magic water (`water_fragment.glsl`)
- 4×4 texture atlas with 14 named block types (grass, sand, brick, stone, wood, leaf, water, crystal, magic water, dirt, snow, glass, planks, gravel)
- Per-vertex ambient occlusion baked into the vertex buffer at block-show time — top faces fully lit, sides dimmed, bottom darkest, corners darkened by solid neighbours
- Sector-based geometry visibility with frustum culling — only sectors near the player and within the camera's view frustum are batched and drawn (~59% of background sectors culled)
- Splash screen on launch while world generates

**Lighting & atmosphere**
- Full day/night cycle — 10-minute days with smooth sky colour transitions: dawn (warm orange) → midday (sky blue) → dusk (deep red-orange) → midnight (near-black navy)
- `sun_brightness` uniform applied to both block and water shaders each frame — world darkens naturally at night
- Starts at noon on every fresh launch

**Player & physics**
- First-person camera with yaw/pitch look
- Walk, sprint, crouch, jump, and fly modes
- Gravity, terminal velocity, and per-face AABB collision
- Sprint increases FOV dynamically; crouch lowers eye height
- Fall-out-of-world respawn safety net

**World interaction**
- Left-click to remove blocks (stone is indestructible)
- Right-click (or Ctrl+Left-click) to place the selected block type
- Block-break particle burst — coloured to match the broken block, gravity physics, alpha fade
- 12-slot inventory hotbar with atlas texture icons, centred at screen bottom
- Mouse scroll wheel cycles through hotbar slots
- Gapped crosshair with white fill and dark outline — readable against any background
- FPS counter, position display, and status toast messages

**Save / Load**
- `F5` saves the current world to `world_save.json`
- World auto-loads on next launch if the save file exists
- `F6` deletes the save — next launch regenerates a fresh world from a new seed
- Full block texture roundtrip through named string keys

---

## 🗂️ Project structure

```
timecraft/
├── main.py              # Entry point — splash screen, GL setup, event loop
├── window.py            # pyglet Window subclass — input, camera, physics, draw
├── model.py             # World state — gen, sectors, frustum, particles, save/load
├── config.py            # All constants — physics, AO, hotbar, day/night, shaders
├── noise_gen.py         # Custom seeded cosine-interpolated terrain noise
├── util.py              # cube_vertices(), compute_ao(), frustum culling, sectorize()
├── water_vertex.glsl    # Animated water vertex shader
├── water_fragment.glsl  # Water shimmer + magic water + sun_brightness shader
├── texture.png          # 4×4 block texture atlas
└── test_timecraft.py    # 221 passing tests (pytest)

setup.bat                # One-shot venv setup + dependency install
run.bat                  # Activate venv and launch
test.bat                 # Run pytest
shell.bat                # Open activated venv shell
doctor.bat               # Dependency health check
```

---

## ▶️ Running TimeCraft

**Requirements:** Python 3.11+, Windows (batch scripts; engine is cross-platform)

**Quick start:**
```bat
setup.bat      :: first time only — creates venv, installs deps
run.bat        :: launch the game
```

**Manual:**
```bash
python -m pip install -r requirements.txt
python timecraft/main.py
```

**Dependencies** (`requirements.txt`):
```
pyglet==2.1.14
PyOpenGL==3.1.10
PyOpenGL-accelerate==3.1.10
pytest
```

---

## 🎮 Controls

| Key / Button | Action |
|---|---|
| `W A S D` | Move |
| `Mouse` | Look |
| `Space` | Jump |
| `R` | Sprint |
| `Left Shift` | Crouch |
| `Tab` | Toggle fly mode |
| `C` (hold) | Zoom |
| `1–0` | Select block type |
| `Scroll wheel` | Cycle hotbar |
| `Left click` | Remove block |
| `Right click` | Place block |
| `F5` | Save world |
| `F6` | Clear save (new world on restart) |
| `Esc` | Release mouse |

---

## 🧪 Tests

221 tests, all passing. Run with:
```bat
test.bat
```
or:
```bash
pytest timecraft/test_timecraft.py -v
```

Test coverage spans: `normalize`, `sectorize`, `cube_vertices`, config constants, noise generation, quad index generation, world block operations, sector assignment, save/load roundtrip, spawn point logic, all 14 block type definitions, ambient occlusion (AO neighbour table, compute_ao correctness), particle system (Particle dataclass, spawn, update, colour mapping), hotbar config (atlas cell map, slot geometry), day/night cycle (sun_brightness curve, sky_colour transitions), and frustum culling (plane extraction, AABB tests, sector AABB, model integration).

---

## ⚠️ Known limitations

- World is infinite — terrain generated on demand, distant chunks evicted
- No point light sources — lighting is purely sun-driven ambient
- No entities, mobs, or NPCs
- No multiplayer
- Save format is plain JSON — not optimised for large worlds

---

## 💡 Where this could go

The architecture already supports growth in several directions:

- Chunked infinite world loading (sector system is designed for it)
- Block placing sounds (`pyglet.media` already available)
- Binary save format for performance at scale
- Biome system (second noise pass for moisture/temperature)
- Entity and NPC systems (collision infrastructure already solid)
- Weather and seasonal effects (day/night uniform already flowing)
- A rendered sun disc in the sky

---

## 📜 License

Personal project — unlicensed. Do whatever you like with it.

---

## 🏷️ Status

**Active** — v0.7, 221 tests passing, fully playable.
