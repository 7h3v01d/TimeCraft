# config.py

import math

# Helper functions moved from util.py to break the circular import
def tex_coord(x, y, n=4):
    """ Return the bounding vertices of the texture square. """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m

def tex_coords(top, bottom, side):
    """ Return a list of the texture squares for the top, bottom and side. """
    top = tex_coord(*top)
    bottom = tex_coord(*bottom)
    side = tex_coord(*side)
    result = []
    result.extend(top)
    result.extend(bottom)
    result.extend(side * 4)
    return result

TICKS_PER_SEC = 60

# Size of sectors used to ease block loading.
SECTOR_SIZE = 16

# Movement variables
WALKING_SPEED = 5
FLYING_SPEED = 15
CROUCH_SPEED = 2
SPRINT_SPEED = 7
SPRINT_FOV = SPRINT_SPEED / 2

# Physics variables
GRAVITY = 20.0
MAX_JUMP_HEIGHT = 1.0 # About the height of a block.
JUMP_SPEED = math.sqrt(2 * GRAVITY * MAX_JUMP_HEIGHT)
TERMINAL_VELOCITY = 50

# Player variables
PLAYER_HEIGHT = 2
PLAYER_FOV = 80.0

# Texture path
import os
TEXTURE_PATH = os.path.join(os.path.dirname(__file__), 'texture.png')

# Texture coordinates
GRASS = tex_coords((1, 0), (0, 1), (0, 0))
SAND = tex_coords((1, 1), (1, 1), (1, 1))
BRICK = tex_coords((2, 0), (2, 0), (2, 0))
STONE = tex_coords((2, 1), (2, 1), (2, 1))
WOOD = tex_coords((3, 1), (3, 1), (3, 1))
LEAF = tex_coords((3, 0), (3, 0), (3, 0))
WATER = tex_coords((0, 2), (0, 2), (0, 2))
CRYSTAL = tex_coords((0, 3), (0, 3), (0, 3))
MAGIC_WATER = tex_coords((1, 3), (1, 3), (1, 3))  # Glowing water

DIRT = tex_coords((0, 1), (0, 1), (0, 1))        # atlas (0,1) - reuses dirt face
SNOW = tex_coords((3, 3), (0, 1), (2, 1))         # atlas (3,3) snow top, dirt bottom, stone sides
GLASS = tex_coords((1, 2), (1, 2), (1, 2))        # atlas (1,2)
PLANKS = tex_coords((2, 2), (2, 2), (2, 2))       # atlas (2,2)
GRAVEL = tex_coords((3, 2), (3, 2), (3, 2))       # atlas (3,2)


# Particle colours (RGBA)
PARTICLE_STONE = (100, 100, 100, 255)
PARTICLE_WOOD = (139, 69, 19, 255)
PARTICLE_LEAF = (0, 100, 0, 255)
PARTICLE_CRYSTAL = (255, 255, 255, 200)
PARTICLE_DIRT = (101, 67, 33, 255)
PARTICLE_SNOW = (230, 240, 255, 255)
PARTICLE_GLASS = (180, 220, 255, 200)
PARTICLE_PLANKS = (160, 120, 60, 255)
PARTICLE_GRAVEL = (130, 120, 110, 255)
PARTICLE_GRASS = (80, 140, 50, 255)
PARTICLE_SAND = (210, 190, 130, 255)
PARTICLE_BRICK = (160, 80, 60, 255)
PARTICLE_WATER = (60, 120, 200, 180)
PARTICLE_MAGIC_WATER = (100, 80, 220, 200)

# Particle behaviour
PARTICLE_COUNT = 6          # particles spawned per broken block
PARTICLE_LIFETIME = 0.55    # seconds before a particle disappears
PARTICLE_SIZE = 4           # screen pixels (drawn in 2-D HUD pass)
PARTICLE_SPEED = 4.0        # initial speed magnitude (world units/s)
PARTICLE_GRAVITY = 18.0     # downward acceleration (world units/s²)

# Maps block texture list → particle colour. Built after all texture constants exist.
# tuple() wrapping is needed because lists aren't hashable.
TEXTURE_PARTICLE_MAP = {}  # populated below after constants are defined


def _build_particle_map():
    pairs = [
        (GRASS, PARTICLE_GRASS),
        (SAND, PARTICLE_SAND),
        (BRICK, PARTICLE_BRICK),
        (STONE, PARTICLE_STONE),
        (WOOD, PARTICLE_WOOD),
        (LEAF, PARTICLE_LEAF),
        (WATER, PARTICLE_WATER),
        (CRYSTAL, PARTICLE_CRYSTAL),
        (MAGIC_WATER, PARTICLE_MAGIC_WATER),
        (DIRT, PARTICLE_DIRT),
        (SNOW, PARTICLE_SNOW),
        (GLASS, PARTICLE_GLASS),
        (PLANKS, PARTICLE_PLANKS),
        (GRAVEL, PARTICLE_GRAVEL),
    ]
    for tex, colour in pairs:
        TEXTURE_PARTICLE_MAP[tuple(tex)] = colour


_build_particle_map()

# Block faces
FACES = [
    ( 0, 1, 0),
    ( 0,-1, 0),
    (-1, 0, 0),
    ( 1, 0, 0),
    ( 0, 0, 1),
    ( 0, 0,-1),
]

# ---------------------------------------------------------------------------
# Hotbar UI
# ---------------------------------------------------------------------------
HOTBAR_SLOT_SIZE = 50       # pixel size of each square slot
HOTBAR_PADDING   = 4        # gap between slots (and between slot edge and icon)
HOTBAR_ICON_PAD  = 5        # inset from slot edge to the texture icon
HOTBAR_Y         = 10       # bottom edge y-offset from screen bottom

# Maps a block texture list (as tuple) → atlas (col, row) for the TOP face.
# Used by the hotbar to sample the right cell from the 4×4 texture atlas.
# Row 0 is at the BOTTOM of the atlas image (pyglet convention).
TEXTURE_ATLAS_CELL = {}     # populated by _build_atlas_cell_map() below

def _build_atlas_cell_map():
    pairs = [
        (GRASS,        (1, 0)),
        (SAND,         (1, 1)),
        (BRICK,        (2, 0)),
        (STONE,        (2, 1)),
        (WOOD,         (3, 1)),
        (LEAF,         (3, 0)),
        (WATER,        (0, 2)),
        (CRYSTAL,      (0, 3)),
        (MAGIC_WATER,  (1, 3)),
        (DIRT,         (0, 1)),
        (SNOW,         (3, 3)),
        (GLASS,        (1, 2)),
        (PLANKS,       (2, 2)),
        (GRAVEL,       (3, 2)),
    ]
    for tex, cell in pairs:
        TEXTURE_ATLAS_CELL[tuple(tex)] = cell

_build_atlas_cell_map()
# ---------------------------------------------------------------------------
# Base brightness per face (top=full, bottom=darkest, sides=mid).
# Face order matches cube_vertices: top, bottom, left, right, front, back.
AO_FACE_BASE = [1.00, 0.50, 0.75, 0.75, 0.75, 0.75]

# Darkness subtracted per solid neighbour block touching a vertex corner.
AO_STEP = 0.15   # max 3 neighbours × 0.15 = 0.45 total darkening

# For each (face, vertex) pair: 3 integer neighbour offsets (side_a, side_b, diagonal).
# 6 faces × 4 verts = 24 entries.  Order mirrors cube_vertices face/vert order.
AO_NEIGHBOURS = [
    # Face 0 — top (normal +y)
    [(-1,1,0), (0,1,-1), (-1,1,-1)],   # v0
    [(-1,1,0), (0,1, 1), (-1,1, 1)],   # v1
    [( 1,1,0), (0,1, 1), ( 1,1, 1)],   # v2
    [( 1,1,0), (0,1,-1), ( 1,1,-1)],   # v3
    # Face 1 — bottom (normal -y)
    [(-1,-1,0), (0,-1,-1), (-1,-1,-1)],
    [( 1,-1,0), (0,-1,-1), ( 1,-1,-1)],
    [( 1,-1,0), (0,-1, 1), ( 1,-1, 1)],
    [(-1,-1,0), (0,-1, 1), (-1,-1, 1)],
    # Face 2 — left (normal -x)
    [(-1,0,-1), (-1,-1,0), (-1,-1,-1)],
    [(-1,0, 1), (-1,-1,0), (-1,-1, 1)],
    [(-1,0, 1), (-1, 1,0), (-1, 1, 1)],
    [(-1,0,-1), (-1, 1,0), (-1, 1,-1)],
    # Face 3 — right (normal +x)
    [(1,0, 1), (1,-1,0), (1,-1, 1)],
    [(1,0,-1), (1,-1,0), (1,-1,-1)],
    [(1,0,-1), (1, 1,0), (1, 1,-1)],
    [(1,0, 1), (1, 1,0), (1, 1, 1)],
    # Face 4 — front (normal +z)
    [(-1,0,1), (0,-1,1), (-1,-1,1)],
    [( 1,0,1), (0,-1,1), ( 1,-1,1)],
    [( 1,0,1), (0, 1,1), ( 1, 1,1)],
    [(-1,0,1), (0, 1,1), (-1, 1,1)],
    # Face 5 — back (normal -z)
    [( 1,0,-1), (0,-1,-1), ( 1,-1,-1)],
    [(-1,0,-1), (0,-1,-1), (-1,-1,-1)],
    [(-1,0,-1), (0, 1,-1), (-1, 1,-1)],
    [( 1,0,-1), (0, 1,-1), ( 1, 1,-1)],
]
# ---------------------------------------------------------------------------
# Day / night cycle
# ---------------------------------------------------------------------------
DAY_LENGTH = 600.0        # seconds per full day (10 real minutes)
SUN_MIN_BRIGHTNESS = 0.05 # minimum brightness at midnight

# Sky colour keypoints (r, g, b) — linearly interpolated each frame
SKY_DAWN  = (0.85, 0.50, 0.20)
SKY_DAY   = (0.50, 0.69, 1.00)
SKY_DUSK  = (0.70, 0.30, 0.10)
SKY_NIGHT = (0.03, 0.03, 0.12)


def sun_brightness(game_time):
    """Return sun brightness in [SUN_MIN_BRIGHTNESS, 1.0].

    Uses a (sin+1)/2 curve so noon=1.0, midnight=SUN_MIN_BRIGHTNESS,
    and dawn/dusk sit at a comfortable ~0.5.
    angle=0 → dawn, π/2 → noon, π → dusk, 3π/2 → midnight.
    """
    import math
    angle = (game_time / DAY_LENGTH) * 2.0 * math.pi
    t = (math.sin(angle) + 1.0) * 0.5   # 0..1
    return SUN_MIN_BRIGHTNESS + (1.0 - SUN_MIN_BRIGHTNESS) * t


def sky_colour(game_time):
    """Return (r, g, b) sky clear colour for the given game_time.

    Blends between four keypoints across the day:
      dawn (t=0) → midday (t=DAY_LENGTH/4) → dusk (t=DAY_LENGTH/2)
      → midnight (t=3*DAY_LENGTH/4) → dawn again.
    """
    import math

    def lerp3(a, b, x):
        x = max(0.0, min(1.0, x))
        return (a[0] + (b[0]-a[0])*x,
                a[1] + (b[1]-a[1])*x,
                a[2] + (b[2]-a[2])*x)

    norm = (game_time % DAY_LENGTH) / DAY_LENGTH  # 0..1 across one day
    angle = norm * 2.0 * math.pi

    if angle < math.pi:
        # Day half: dawn → midday → dusk
        phase = angle / math.pi           # 0..1
        if phase < 0.5:
            return lerp3(SKY_DAWN, SKY_DAY, phase * 2.0)
        else:
            return lerp3(SKY_DAY, SKY_DUSK, (phase - 0.5) * 2.0)
    else:
        # Night half: dusk → midnight → dawn
        phase = (angle - math.pi) / math.pi   # 0..1
        if phase < 0.5:
            return lerp3(SKY_DUSK, SKY_NIGHT, phase * 2.0)
        else:
            return lerp3(SKY_NIGHT, SKY_DAWN, (phase - 0.5) * 2.0)

# ---------------------------------------------------------------------------
# Binary save format  (.tcw — TimeCraft World)
# ---------------------------------------------------------------------------
SAVE_MAGIC   = b'TCWF'   # 4-byte file magic
SAVE_VERSION = 1         # bumped if format changes incompatibly

# Stable block ID table — ORDER MUST NEVER CHANGE (it's the on-disk encoding).
# New block types are appended; existing entries are never moved or removed.
BLOCK_IDS = [
    'GRASS', 'SAND', 'BRICK', 'STONE', 'WOOD', 'LEAF',
    'WATER', 'CRYSTAL', 'MAGIC_WATER', 'DIRT', 'SNOW',
    'GLASS', 'PLANKS', 'GRAVEL',
]
