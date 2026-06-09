# model.py

import random
import time
import json
import os
import struct
from collections import deque
from dataclasses import dataclass
from typing import Tuple
import math

import pyglet
from pyglet.gl import gl
from pyglet import image

import config
from noise_gen import NoiseGen


@dataclass
class Particle:
    """A single short-lived block-break particle.

    Position and velocity are in world space.  Rendering is handled by
    Window.draw_particles() which projects the world position through the
    current view/projection matrices and draws a coloured square on the
    2-D HUD pass.
    """
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    colour: Tuple[int, int, int, int]
    lifetime: float = config.PARTICLE_LIFETIME
    age: float = 0.0

    @property
    def alive(self) -> bool:
        return self.age < self.lifetime

    @property
    def alpha_fraction(self) -> float:
        """0.0 (just spawned) → 1.0 (about to die); used to fade out."""
        return self.age / self.lifetime

    def update(self, dt: float) -> None:
        self.vy -= config.PARTICLE_GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        self.age += dt
from util import sectorize, cube_vertices, normalize, compute_ao, \
                  extract_frustum_planes, aabb_outside_frustum, sector_aabb

# Indices to convert 6 quads (each 4 verts) into triangles
# Each face: verts 0,1,2 and 0,2,3
QUAD_INDICES = []
for _face in range(6):
    base = _face * 4
    QUAD_INDICES += [base, base+1, base+2, base, base+2, base+3]

# Map texture list -> save string and back
_TEXTURE_NAMES = {
    'GRASS': config.GRASS, 'SAND': config.SAND, 'BRICK': config.BRICK,
    'STONE': config.STONE, 'WOOD': config.WOOD, 'LEAF': config.LEAF,
    'WATER': config.WATER, 'CRYSTAL': config.CRYSTAL,
    'MAGIC_WATER': config.MAGIC_WATER,
    'DIRT': config.DIRT, 'SNOW': config.SNOW,
    'GLASS': config.GLASS, 'PLANKS': config.PLANKS, 'GRAVEL': config.GRAVEL,
}
_TEXTURE_LOOKUP = {tuple(v): k for k, v in _TEXTURE_NAMES.items()}

# Binary save: stable int → texture and texture → int mappings.
# Derived from config.BLOCK_IDS so the canonical order lives in one place.
_BLOCK_ID_TO_TEX = {i: _TEXTURE_NAMES[name]
                    for i, name in enumerate(config.BLOCK_IDS)
                    if name in _TEXTURE_NAMES}
_TEX_TO_BLOCK_ID = {tuple(tex): i for i, tex in _BLOCK_ID_TO_TEX.items()}

# Struct formats (little-endian)
#   Header: magic(4s) + version(B) + seed(i) + block_count(I)  = 13 bytes
#   Block:  x(h) + y(h) + z(h) + block_id(B)                  =  7 bytes
_HEADER_FMT = struct.Struct('<4sBiI')
_BLOCK_FMT  = struct.Struct('<hhhB')


def _make_default_shader():
    vert_src = """
#version 330 core
in vec3 position;
in vec2 tex_coords;
in float ao;
out vec2 v_texcoord;
out float v_ao;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    v_texcoord = tex_coords;
    v_ao = ao;
}
"""
    frag_src = """
#version 330 core
in vec2 v_texcoord;
in float v_ao;
out vec4 out_color;
uniform sampler2D our_texture;
uniform float sun_brightness;
void main() {
    vec4 tex = texture(our_texture, v_texcoord);
    out_color = vec4(tex.rgb * v_ao * sun_brightness, tex.a);
}
"""
    return pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(vert_src, 'vertex'),
        pyglet.graphics.shader.Shader(frag_src, 'fragment'),
    )


def _make_water_shader(here):
    vert_src = open(os.path.join(here, 'water_vertex.glsl')).read()
    frag_src = open(os.path.join(here, 'water_fragment.glsl')).read()
    return pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(vert_src, 'vertex'),
        pyglet.graphics.shader.Shader(frag_src, 'fragment'),
    )


class TextureBindGroup(pyglet.graphics.Group):
    """Group that binds a texture and sets shader uniforms each draw."""
    def __init__(self, texture, program, order=0, parent=None):
        super().__init__(order=order, parent=parent)
        self.texture = texture
        self.program = program

    def set_state(self):
        self.program.use()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(self.texture.target, self.texture.id)
        self.program['our_texture'] = 0

    def unset_state(self):
        gl.glBindTexture(self.texture.target, 0)
        self.program.stop()

    def __eq__(self, other):
        return (isinstance(other, TextureBindGroup) and
                self.texture == other.texture and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent)

    def __hash__(self):
        return hash((self.texture.id, id(self.program), self.order))


class Model(object):

    SAVE_FILE  = os.path.join(os.path.dirname(__file__), 'world_save.tcw')
    SAVE_FILE_LEGACY = os.path.join(os.path.dirname(__file__), 'world_save.json')

    def __init__(self):
        self.batch = pyglet.graphics.Batch()

        _HERE = os.path.dirname(__file__)
        texture = image.load(config.TEXTURE_PATH).get_texture()

        self.default_shader = _make_default_shader()
        self.water_shader = _make_water_shader(_HERE)

        self.group = TextureBindGroup(texture, self.default_shader)
        self.water_group = TextureBindGroup(texture, self.water_shader)

        self.world = {}
        self.shown = {}
        self._shown = {}
        self.sectors = {}
        self.queue = deque()
        self.particles = []
        self.generated_chunks = set()   # (sx, sz) pairs already terrain-generated
        self.game_time = config.DAY_LENGTH / 4.0  # start at noon
        self.seed = None  # set by _initialize or load_world
        self.frustum_planes = None  # set each frame by set_frustum()
        self._initialize()

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------

    def _initialize(self):
        """Load existing world or start fresh near the origin.

        For a fresh world, just set the seed — actual terrain is generated
        on-demand by generate_chunk() as the player moves.  The initial
        chunks around (0,0) are generated here so the player spawns into
        visible terrain.
        """
        if os.path.exists(self.SAVE_FILE):
            self.load_world()
        elif os.path.exists(self.SAVE_FILE_LEGACY):
            self.load_world()
        else:
            self.seed = random.randint(0, 1_000_000)
            self._gen = NoiseGen(self.seed)
            # Generate a small area around the origin for the spawn region
            for sx in range(-1, 2):
                for sz in range(-1, 2):
                    self.generate_chunk(sx, sz)

    def generate_chunk(self, sx, sz):
        """Generate terrain for sector (sx, 0, sz) if not already generated.

        Covers world columns (sx*S .. sx*S+S-1, sz*S .. sz*S+S-1) where
        S = SECTOR_SIZE.  Skips silently if the chunk was already generated
        or if this world was loaded from a save (in which case the saved
        block state is authoritative).
        """
        if (sx, sz) in self.generated_chunks:
            return
        self.generated_chunks.add((sx, sz))

        if not hasattr(self, '_gen') or self._gen is None:
            return   # loaded world — don't overwrite saved blocks

        S = config.SECTOR_SIZE
        rng = random.Random(self.seed ^ (sx * 73856093) ^ (sz * 19349663))

        for lx in range(S):
            for lz in range(S):
                x = sx * S + lx
                z = sz * S + lz
                h = max(1, int(self._gen.getHeight(x, z)))

                if h < 15:
                    self.add_block((x, h, z), config.SAND, immediate=False)
                    for y in range(h, 15):
                        self.add_block((x, y, z),
                            config.MAGIC_WATER if rng.random() > 0.99 else config.WATER,
                            immediate=False)
                    continue

                if h < 18:
                    self.add_block((x, h, z), config.SAND, immediate=False)

                self.add_block((x, h, z), config.GRASS, immediate=False)
                for y in range(h - 1, 0, -1):
                    self.add_block((x, y, z), config.STONE, immediate=False)

                if h > 20 and rng.random() > 0.99:
                    tree_h = rng.randint(5, 7)
                    for y in range(h + 1, h + tree_h):
                        self.add_block((x, y, z), config.WOOD, immediate=False)
                    leafh = h + tree_h
                    for lz2 in range(z - 2, z + 3):
                        for lx2 in range(x - 2, x + 3):
                            for ly in range(3):
                                if (lx2, leafh + ly, lz2) != (x, leafh + ly, z) \
                                        or rng.random() > 0.1:
                                    self.add_block((lx2, leafh + ly, lz2),
                                                   config.LEAF, immediate=False)

                if h > 18:
                    for dy in range(1, min(4, h)):
                        b = (x, h - dy, z)
                        if b in self.world and self.world[b] == config.STONE:
                            self.add_block(b, config.DIRT, immediate=False)
                            break

                if h > 30:
                    self.add_block((x, h, z), config.SNOW, immediate=False)

                if 18 <= h <= 20 and rng.random() > 0.6:
                    self.add_block((x, h, z), config.GRAVEL, immediate=False)

                if h > 25 and rng.random() > 0.995:
                    self.add_block((x, h - 1, z), config.CRYSTAL, immediate=False)

    # ------------------------------------------------------------------
    # Spawn point
    # ------------------------------------------------------------------

    def get_spawn_point(self):
        """Return (x, y+2, z) at the highest solid block near the world origin."""
        search_radius = 8
        best = None
        for dx in range(-search_radius, search_radius + 1):
            for dz in range(-search_radius, search_radius + 1):
                for y in range(60, 0, -1):
                    if (dx, y, dz) in self.world:
                        tex = self.world[(dx, y, dz)]
                        if tex not in (config.WATER, config.MAGIC_WATER):
                            if best is None or y > best[1]:
                                best = (dx, y, dz)
                            break
        if best:
            return (float(best[0]), float(best[1] + 2), float(best[2]))
        return (0.0, 50.0, 0.0)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_world(self):
        """Serialise world to a binary .tcw file.

        Format (little-endian):
          Header — magic(4s) version(B) seed(i) block_count(I)  [13 bytes]
          Blocks — x(h) y(h) z(h) block_id(B) per block         [7 bytes each]

        Returns the number of blocks written.
        """
        blocks = [
            (pos, _TEX_TO_BLOCK_ID.get(tuple(tex)))
            for pos, tex in self.world.items()
        ]
        blocks = [(pos, bid) for pos, bid in blocks if bid is not None]

        with open(self.SAVE_FILE, 'wb') as f:
            f.write(_HEADER_FMT.pack(
                config.SAVE_MAGIC,
                config.SAVE_VERSION,
                self.seed if self.seed is not None else 0,
                len(blocks),
            ))
            for (x, y, z), bid in blocks:
                f.write(_BLOCK_FMT.pack(x, y, z, bid))

        return len(blocks)

    def load_world(self):
        """Load a world from disk, auto-detecting binary (.tcw) or legacy JSON.

        Binary is tried first via magic-byte check; JSON is the fallback for
        one-time migration of pre-v0.8 saves.  After a successful JSON load
        the world is immediately re-saved in binary format.
        """
        path = self.SAVE_FILE if os.path.exists(self.SAVE_FILE) else self.SAVE_FILE_LEGACY

        with open(path, 'rb') as f:
            magic = f.read(4)

        if magic == config.SAVE_MAGIC:
            self._load_binary(path)
        else:
            self._load_json_legacy(path)
            self.save_world()   # migrate to binary on first load

    def _load_binary(self, path):
        """Read a .tcw binary save file."""
        self._gen = None   # saved state is authoritative — no terrain overwrite
        with open(path, 'rb') as f:
            magic, version, seed, count = _HEADER_FMT.unpack(
                f.read(_HEADER_FMT.size))
            self.seed = seed
            block_size = _BLOCK_FMT.size
            for _ in range(count):
                x, y, z, bid = _BLOCK_FMT.unpack(f.read(block_size))
                tex = _BLOCK_ID_TO_TEX.get(bid, config.STONE)
                self.add_block((x, y, z), tex, immediate=False)
        # Mark all loaded sectors as generated so they aren't re-terrainformed
        for sector in self.sectors:
            self.generated_chunks.add((sector[0], sector[2]))

    def _load_json_legacy(self, path):
        """Read a legacy JSON save (pre-v0.8).  Used for one-time migration."""
        self._gen = None   # saved state is authoritative
        with open(path, 'r') as f:
            data = json.load(f)
        self.seed = data.get('seed')
        for entry in data['blocks']:
            pos = tuple(entry['pos'])
            tex = _TEXTURE_NAMES.get(entry['tex'], config.STONE)
            self.add_block(pos, tex, immediate=False)
        for sector in self.sectors:
            self.generated_chunks.add((sector[0], sector[2]))

    def delete_save(self):
        """Remove save file(s) so next launch regenerates the world."""
        for path in (self.SAVE_FILE, self.SAVE_FILE_LEGACY):
            if os.path.exists(path):
                os.remove(path)

    # ------------------------------------------------------------------
    # Core world logic (unchanged)
    # ------------------------------------------------------------------

    def hit_test(self, position, vector, max_distance=8):
        m = 8
        x, y, z = position
        dx, dy, dz = vector
        previous = None
        for _ in range(max_distance * m):
            key = normalize((x, y, z))
            if key != previous and key in self.world:
                return key, previous
            previous = key
            x, y, z = x + dx / m, y + dy / m, z + dz / m
        return None, None

    def exposed(self, position):
        x, y, z = position
        for dx, dy, dz in config.FACES:
            if (x + dx, y + dy, z + dz) not in self.world:
                return True
        return False

    def add_block(self, position, texture, immediate=True, hit_vector=None):
        if position in self.world:
            self.remove_block(position, immediate)
        self.world[position] = texture
        self.sectors.setdefault(sectorize(position, config.SECTOR_SIZE), []).append(position)
        if immediate:
            if self.exposed(position):
                self.show_block(position)
            self.check_neighbors(position)

    def remove_block(self, position, immediate=True, hit_vector=None):
        self.spawn_particles(position)   # must come before del so texture is still in world
        del self.world[position]
        self.sectors[sectorize(position, config.SECTOR_SIZE)].remove(position)
        if immediate:
            if position in self.shown:
                self.hide_block(position)
            self.check_neighbors(position)

    def check_neighbors(self, position):
        x, y, z = position
        for dx, dy, dz in config.FACES:
            key = (x + dx, y + dy, z + dz)
            if key not in self.world:
                continue
            if self.exposed(key):
                if key not in self.shown:
                    self.show_block(key)
            else:
                if key in self.shown:
                    self.hide_block(key)

    def show_block(self, position, immediate=True):
        texture = self.world[position]
        self.shown[position] = texture
        if immediate:
            self._show_block(position, texture)
        else:
            self._enqueue(self._show_block, position, texture)

    def _show_block(self, position, texture):
        x, y, z = position
        vertex_data = cube_vertices(x, y, z, 0.5)
        texture_data = list(texture)
        ao_data = compute_ao(position, self.world)

        is_water = (texture == config.WATER or texture == config.MAGIC_WATER)
        group = self.water_group if is_water else self.group
        shader = self.water_shader if is_water else self.default_shader

        if is_water:
            vlist = shader.vertex_list_indexed(
                24, gl.GL_TRIANGLES, QUAD_INDICES,
                self.batch, group,
                position=('f', vertex_data),
                tex_coords=('f', texture_data),
            )
        else:
            vlist = shader.vertex_list_indexed(
                24, gl.GL_TRIANGLES, QUAD_INDICES,
                self.batch, group,
                position=('f', vertex_data),
                tex_coords=('f', texture_data),
                ao=('f', ao_data),
            )
        self._shown[position] = vlist

    def hide_block(self, position, immediate=True):
        self.shown.pop(position)
        if immediate:
            self._hide_block(position)
        else:
            self._enqueue(self._hide_block, position)

    def _hide_block(self, position):
        if position in self._shown:
            self._shown.pop(position).delete()

    def show_sector(self, sector):
        """Show all exposed blocks in *sector*, skipping sectors outside the frustum."""
        if self.frustum_planes is not None:
            mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = sector_aabb(sector)
            if aabb_outside_frustum(self.frustum_planes,
                                    mn_x, mn_y, mn_z,
                                    mx_x, mx_y, mx_z):
                return
        for position in self.sectors.get(sector, []):
            if position not in self.shown and self.exposed(position):
                self.show_block(position, False)

    def hide_sector(self, sector):
        for position in self.sectors.get(sector, []):
            if position in self.shown:
                self.hide_block(position, False)

    def set_frustum(self, vp_matrix):
        """Update the stored frustum planes from the current view-projection matrix.

        Called once per frame from Window.on_draw() before change_sectors runs.
        """
        self.frustum_planes = extract_frustum_planes(vp_matrix)

    def change_sectors(self, before, after):
        """Update visible sectors as the player moves.

        - Generates terrain for any newly entered sector not yet generated.
        - Shows blocks in sectors entering the render radius.
        - Hides blocks in sectors leaving the render radius.
        - Evicts blocks from world dict for sectors beyond EVICT_DISTANCE,
          keeping memory bounded regardless of how far the player travels.
        """
        pad   = config.RENDER_DISTANCE
        evict = config.EVICT_DISTANCE

        before_set = set()
        after_set  = set()
        evict_set  = set()

        for dx in range(-evict, evict + 1):
            for dz in range(-evict, evict + 1):
                dist = abs(dx) + abs(dz)   # Manhattan — cheap, good enough
                if after:
                    ax, _ay, az = after
                    s = (ax + dx, 0, az + dz)
                    if dist <= pad:
                        after_set.add(s)
                    if dist > evict:
                        evict_set.add(s)
                if before:
                    bx, _by, bz = before
                    s = (bx + dx, 0, bz + dz)
                    if dist <= pad:
                        before_set.add(s)

        # Generate terrain for sectors newly in range that haven't been gen'd yet
        for sector in after_set - before_set:
            sx, _sy, sz = sector
            self.generate_chunk(sx, sz)

        # Show blocks entering the pad
        for sector in after_set - before_set:
            self.show_sector(sector)

        # Hide blocks leaving the pad
        for sector in before_set - after_set:
            self.hide_sector(sector)

        # Evict blocks far from the player — remove from world dict entirely
        if after:
            ax, _ay, az = after
            for dx in range(-evict, evict + 1):
                for dz in range(-evict, evict + 1):
                    if abs(dx) + abs(dz) > evict:
                        sector = (ax + dx, 0, az + dz)
                        self._evict_sector(sector)

    def _evict_sector(self, sector):
        """Remove all blocks in *sector* from the world dict and GPU batch.

        The generated_chunks entry is kept so the sector isn't re-generated
        if the player returns — it will be loaded from the save instead.
        """
        positions = list(self.sectors.get(sector, []))
        for position in positions:
            if position in self.shown:
                self.hide_block(position, immediate=True)
            if position in self.world:
                del self.world[position]
        if sector in self.sectors:
            del self.sectors[sector]

    def _enqueue(self, func, *args):
        self.queue.append((func, args))

    def _dequeue(self):
        func, args = self.queue.popleft()
        func(*args)

    def process_queue(self):
        start = time.process_time()
        while self.queue and time.process_time() - start < 1.0 / config.TICKS_PER_SEC:
            self._dequeue()
        dt = 1.0 / config.TICKS_PER_SEC
        self.game_time += dt

    def process_entire_queue(self):
        while self.queue:
            self._dequeue()

    # ------------------------------------------------------------------
    # Particle system
    # ------------------------------------------------------------------

    def spawn_particles(self, position):
        """Spawn a burst of particles at *position* coloured to match the block.

        Called by remove_block when a block is broken by the player.
        Does nothing if the block texture has no mapped particle colour.
        """
        tex = self.world.get(position)
        if tex is None:
            return
        colour = config.TEXTURE_PARTICLE_MAP.get(tuple(tex))
        if colour is None:
            return

        x, y, z = position
        for _ in range(config.PARTICLE_COUNT):
            speed = config.PARTICLE_SPEED
            vx = random.uniform(-speed, speed)
            vy = random.uniform(0.5 * speed, speed)   # bias upward
            vz = random.uniform(-speed, speed)
            # Spawn at a random sub-block offset so they don't all stack
            ox = random.uniform(-0.3, 0.3)
            oy = random.uniform(0.0, 0.5)
            oz = random.uniform(-0.3, 0.3)
            self.particles.append(
                Particle(x + ox, y + oy, z + oz, vx, vy, vz, colour)
            )

    def update_particles(self, dt):
        """Advance all particles and discard dead ones.  Call once per frame."""
        live = []
        for p in self.particles:
            p.update(dt)
            if p.alive:
                live.append(p)
        self.particles = live

    def set_shader_uniforms(self, view_matrix, proj_matrix):
        """Called each frame to push view/projection and day/night uniforms into both shaders."""
        view_flat = list(view_matrix)
        proj_flat = list(proj_matrix)
        brightness = config.sun_brightness(self.game_time)
        self.default_shader.use()
        self.default_shader['view'] = view_flat
        self.default_shader['projection'] = proj_flat
        self.default_shader['sun_brightness'] = brightness
        self.default_shader.stop()
        self.water_shader.use()
        self.water_shader['view'] = view_flat
        self.water_shader['projection'] = proj_flat
        self.water_shader['time'] = self.game_time
        self.water_shader['sun_brightness'] = brightness
        self.water_shader.stop()
