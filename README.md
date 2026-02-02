# ⏳ TimeCraft (Archived)

**TimeCraft** is a tile-based sandbox and simulation engine built as a learning project — and a serious one.

It explores **world state, time progression, tile rules, and rendering**, all inside a clean, deterministic update loop.

This project is archived, but fully functional.

---

## 🧠 What is TimeCraft?

TimeCraft started with a simple idea:

> “What if I built a small world that *changes over time* — and I actually understand every part of it?”

The result is a compact simulation engine where:
- the world is a grid of tiles
- time advances in discrete steps
- tiles have identity, rules, and visuals
- rendering and logic are cleanly separated

---

## ✨ Core features

- 🗺️ **Tile-based world**
  - Grid-driven world representation
  - Distinct tile types (dirt, grass, water, stone, etc.)

- ⏱️ **Time-driven simulation**
  - Deterministic update loop
  - World state evolves over time
  - Rules applied consistently per tick

- 🎨 **Texture-based rendering**
  - Visual tiles mapped cleanly from world data
  - Rendering decoupled from simulation logic

- 🧩 **Rule-based behavior**
  - Tiles are data + behavior, not just graphics
  - Easy to extend with new tile types or interactions

- 🖥️ **Interactive loop**
  - Input → update → render
  - Stable, debuggable core loop

---

## 🗂️ Project structure (conceptual)
```text
timecraft/
├── main.py # Game loop + initialization
├── world.py # World grid + state
├── tiles.py # Tile definitions and rules
├── renderer.py # Rendering logic
├── input.py # Player / camera input
└── assets/
└── textures/ # Tile textures
```

(Exact layout may vary — this represents the architecture.)

---

## ▶️ Running TimeCraft

Requirements:
- Python 3.x
- Pygame (or equivalent rendering library)

Run:
```bash
python main.py
```
## ⚠️ Project status
Archived / Learning Engine

- Fully functional core loop
- No save/load system
- No large-scale world generation
- No AI entities or pathfinding
- No packaging or installer

This repo is preserved as a working engine snapshot, not a finished game.

💡 If revisited someday…
Natural extensions would be:

- chunked world loading
- procedural terrain generation
- entity systems
- weather / seasons
- simulation tuning
- save/load support

The core architecture already supports growth.

## 📜 License
Unlicensed (personal archive).

## 🏷️ Status
Archived — solid, educational, and foundational.

TimeCraft represents a deep dive into simulation fundamentals and engine-level thinking.
