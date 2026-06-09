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


# Particle textures
PARTICLE_STONE = (100, 100, 100, 255)
PARTICLE_WOOD = (139, 69, 19, 255)
PARTICLE_LEAF = (0, 100, 0, 255)
PARTICLE_CRYSTAL = (255, 255, 255, 200)
PARTICLE_DIRT = (101, 67, 33, 255)
PARTICLE_SNOW = (230, 240, 255, 255)
PARTICLE_GLASS = (180, 220, 255, 200)
PARTICLE_PLANKS = (160, 120, 60, 255)
PARTICLE_GRAVEL = (130, 120, 110, 255)

# Block faces
FACES = [
    ( 0, 1, 0),
    ( 0,-1, 0),
    (-1, 0, 0),
    ( 1, 0, 0),
    ( 0, 0, 1),
    ( 0, 0,-1),
]