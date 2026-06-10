# util.py

import math
import config


def cube_vertices(x, y, z, n):
    """ Return the vertices of the cube at position x, y, z with size 2*n. """
    return [
        x-n,y+n,z-n, x-n,y+n,z+n, x+n,y+n,z+n, x+n,y+n,z-n,  # top
        x-n,y-n,z-n, x+n,y-n,z-n, x+n,y-n,z+n, x-n,y-n,z+n,  # bottom
        x-n,y-n,z-n, x-n,y-n,z+n, x-n,y+n,z+n, x-n,y+n,z-n,  # left
        x+n,y-n,z+n, x+n,y-n,z-n, x+n,y+n,z-n, x+n,y+n,z+n,  # right
        x-n,y-n,z+n, x+n,y-n,z+n, x+n,y+n,z+n, x-n,y+n,z+n,  # front
        x+n,y-n,z-n, x-n,y-n,z-n, x-n,y+n,z-n, x+n,y+n,z-n,  # back
    ]


def extract_frustum_planes(vp_matrix):
    """Extract 6 normalised frustum planes from a combined view-projection matrix.

    Uses the Gribb/Hartmann method: each plane is a linear combination of the
    matrix rows.  The matrix must be in column-major order (pyglet Mat4 layout).

    Returns a list of 6 (a, b, c, d) tuples in the order:
      left, right, bottom, top, near, far.
    A point (x,y,z) is inside plane i when a*x + b*y + c*z + d >= 0.
    """
    m = list(vp_matrix)
    # Row i (0-based) in column-major: elements m[i], m[i+4], m[i+8], m[i+12]
    planes = []
    for r1, r2, sign in [
        (3, 0, +1),   # left:   row3 + row0
        (3, 0, -1),   # right:  row3 - row0
        (3, 1, +1),   # bottom: row3 + row1
        (3, 1, -1),   # top:    row3 - row1
        (3, 2, +1),   # near:   row3 + row2
        (3, 2, -1),   # far:    row3 - row2
    ]:
        a = m[r1 + 0]  + sign * m[r2 + 0]
        b = m[r1 + 4]  + sign * m[r2 + 4]
        c = m[r1 + 8]  + sign * m[r2 + 8]
        d = m[r1 + 12] + sign * m[r2 + 12]
        length = math.sqrt(a*a + b*b + c*c)
        if length > 0:
            planes.append((a/length, b/length, c/length, d/length))
        else:
            planes.append((a, b, c, d))
    return planes


def aabb_outside_frustum(planes, min_x, min_y, min_z, max_x, max_y, max_z):
    """Return True if the AABB is entirely outside any one frustum plane.

    Uses the positive-vertex test: for each plane normal, pick the AABB
    corner most in the direction of the normal (the "positive vertex").
    If that corner is still behind the plane, the whole box is outside.

    This is a conservative test — it never incorrectly culls a visible
    sector, but may pass a few sectors that are just off-screen.
    """
    for (a, b, c, d) in planes:
        px = max_x if a >= 0 else min_x
        py = max_y if b >= 0 else min_y
        pz = max_z if c >= 0 else min_z
        if a*px + b*py + c*pz + d < 0:
            return True
    return False


def sector_aabb(sector):
    """Return (min_x, min_y, min_z, max_x, max_y, max_z) for a sector tuple.

    Sectors are (sx, 0, sz) integer coords; each covers SECTOR_SIZE blocks
    in x and z.  Y spans the full world height (0..64).
    """
    sx, _sy, sz = sector
    s = config.SECTOR_SIZE
    return (sx * s, 0, sz * s,
            sx * s + s, 64, sz * s + s)


def compute_ao(position, world):
    """Return 24 AO brightness floats (one per cube vertex) for *position*."""
    bx, by, bz = position
    result = []
    for i, neighbours in enumerate(config.AO_NEIGHBOURS):
        face_idx = i // 4
        base = config.AO_FACE_BASE[face_idx]
        count = sum(
            1 for (dx, dy, dz) in neighbours
            if (bx + dx, by + dy, bz + dz) in world
        )
        result.append(max(0.0, base - config.AO_STEP * count))
    return result


def normalize(position):
    """ Accepts `position` of arbitrary precision and returns the block containing that position. """
    x, y, z = position
    x, y, z = (int(round(x)), int(round(y)), int(round(z)))
    return (x, y, z)


def sectorize(position, sector_size):
    """ Returns a tuple representing the sector for the given `position`. """
    x, y, z = normalize(position)
    x, y, z = x // sector_size, y // sector_size, z // sector_size
    return (x, 0, z)


def compute_ao(position, world):
    """Return 24 AO brightness floats (one per cube vertex) for *position*.

    Each value is in [0.0, 1.0].  Base face brightnesses:
      top=1.0, sides=0.75, bottom=0.50 — reduced further for each of the
      three adjacent blocks (2 axis-aligned + 1 diagonal) that are solid.

    Face/vertex order matches cube_vertices exactly so the result can be
    passed directly as the 'ao' vertex attribute.
    """
    bx, by, bz = position
    result = []
    for i, neighbours in enumerate(config.AO_NEIGHBOURS):
        face_idx = i // 4
        base = config.AO_FACE_BASE[face_idx]
        count = sum(
            1 for (dx, dy, dz) in neighbours
            if (bx + dx, by + dy, bz + dz) in world
        )
        result.append(max(0.0, base - config.AO_STEP * count))
    return result


def normalize(position):
    """ Accepts `position` of arbitrary precision and returns the block containing that position. """
    x, y, z = position
    x, y, z = (int(round(x)), int(round(y)), int(round(z)))
    return (x, y, z)


def sectorize(position, sector_size):
    """ Returns a tuple representing the sector for the given `position`. """
    x, y, z = normalize(position)
    x, y, z = x // sector_size, y // sector_size, z // sector_size
    return (x, 0, z)


def collide(position, height, world, portal_tex=None):
    """AABB collision against the world dict.

    Extracted from Window.collide() so it can be shared by both the
    player controller and mob entities without a Window dependency.

    Returns (new_position, on_ground) where on_ground is True if the
    entity is resting on a solid block.
    """
    pad = 0.25
    p   = list(position)
    np  = normalize(position)
    on_ground = False

    for face in config.FACES:
        for i in range(3):
            if not face[i]:
                continue
            d = (p[i] - np[i]) * face[i]
            if d < pad:
                continue
            for dy in range(int(height) + 1):
                op = list(np)
                op[1] -= dy
                op[i] += face[i]
                pos_key = tuple(op)
                if pos_key not in world:
                    continue
                if portal_tex is not None and world[pos_key] == portal_tex:
                    continue   # portal blocks are walk-through
                p[i] -= (d - pad) * face[i]
                if face == (0, -1, 0):
                    on_ground = True
                break

    return tuple(p), on_ground
