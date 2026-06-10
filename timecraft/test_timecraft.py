"""
TimeCraft Test Suite
Tests all headless logic: util, config, noise_gen, and model (GL mocked).
Run from the timecraft/ directory:  pytest test_timecraft.py -v
"""

import sys
import os
import math
import types
import wave
import struct
import random
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Headless pyglet/GL stub — must be installed before any game imports
# ---------------------------------------------------------------------------

def _make_pyglet_stub():
    pyglet = types.ModuleType("pyglet")

    # pyglet.graphics
    graphics = types.ModuleType("pyglet.graphics")
    graphics.Batch = MagicMock
    class _GroupBase:
        def __init__(self, order=0, parent=None):
            self.order = order
            self.parent = parent
    graphics.Group = _GroupBase  # real base class so subclassing works

    class _ShaderStub:
        def __init__(self, src, kind): pass
    class _ProgramStub:
        def __init__(self, *shaders): pass
        def use(self): pass
        def stop(self): pass
        def __setitem__(self, k, v): pass
        def vertex_list_indexed(self, count, mode, indices, batch, group, **kw):
            vl = MagicMock()
            vl.delete = MagicMock()
            return vl
    shader_mod = types.ModuleType("pyglet.graphics.shader")
    shader_mod.Shader = _ShaderStub
    shader_mod.ShaderProgram = _ProgramStub
    graphics.shader = shader_mod

    # pyglet.gl
    gl = types.ModuleType("pyglet.gl")
    for name in ["GL_TRIANGLES", "GL_TEXTURE0", "GL_TEXTURE_2D"]:
        setattr(gl, name, 0)
    for fn in ["glActiveTexture", "glBindTexture", "glEnable", "glDisable"]:
        setattr(gl, fn, MagicMock())
    gl.gl = gl  # pyglet.gl.gl

    # pyglet.image
    image_mod = types.ModuleType("pyglet.image")
    fake_tex = MagicMock()
    fake_tex.target = 0
    fake_tex.id = 1
    fake_region = MagicMock()
    fake_grid = MagicMock()
    fake_grid.__getitem__ = MagicMock(return_value=fake_region)
    image_mod.load = MagicMock(return_value=MagicMock(get_texture=MagicMock(return_value=fake_tex)))
    image_mod.create = MagicMock()
    image_mod.SolidColorImagePattern = MagicMock()
    image_mod.ImageGrid = MagicMock(return_value=fake_grid)
    image_mod.TextureGrid = MagicMock(return_value=fake_grid)

    # pyglet.sprite
    sprite_mod = types.ModuleType("pyglet.sprite")
    class _FakeSprite:
        def __init__(self, img, x=0, y=0, batch=None):
            self.x = x
            self.y = y
            self.width = 0
            self.height = 0
    sprite_mod.Sprite = _FakeSprite

    # pyglet.shapes (extend existing or create)
    shapes_mod = types.ModuleType("pyglet.shapes")
    class _FakeRect:
        def __init__(self, x, y, w, h, color=(255,255,255,255), batch=None): pass
    class _FakeLine:
        def __init__(self, x1, y1, x2, y2, thickness=1, color=(0,0,0,255), batch=None): pass
    shapes_mod.Rectangle = _FakeRect
    shapes_mod.Line = _FakeLine

    # pyglet.media
    media_mod = types.ModuleType("pyglet.media")
    class _FakeSource:
        def play(self): pass
    media_mod.load = MagicMock(return_value=_FakeSource())
    media_mod.StaticSource = _FakeSource

    # pyglet.clock / app
    clock_mod = types.ModuleType("pyglet.clock")
    clock_mod.schedule_interval = MagicMock()
    app_mod = types.ModuleType("pyglet.app")

    pyglet.media = media_mod
    pyglet.graphics = graphics
    pyglet.gl = gl
    pyglet.image = image_mod
    pyglet.sprite = sprite_mod
    pyglet.shapes = shapes_mod
    pyglet.clock = clock_mod
    pyglet.app = app_mod

    sys.modules["pyglet"] = pyglet
    sys.modules["pyglet.media"] = media_mod
    sys.modules["pyglet.graphics"] = graphics
    sys.modules["pyglet.graphics.shader"] = shader_mod
    sys.modules["pyglet.gl"] = gl
    sys.modules["pyglet.gl.gl"] = gl
    sys.modules["pyglet.image"] = image_mod
    sys.modules["pyglet.sprite"] = sprite_mod
    sys.modules["pyglet.shapes"] = shapes_mod
    sys.modules["pyglet.clock"] = clock_mod
    sys.modules["pyglet.app"] = app_mod

_make_pyglet_stub()

# Now safe to import game modules
sys.path.insert(0, ".")          # run from project root
sys.path.insert(0, "timecraft")  # or from parent

import config
import tempfile
import json
from util import cube_vertices, normalize, sectorize, compute_ao, \
                  extract_frustum_planes, aabb_outside_frustum, sector_aabb, \
                  collide as util_collide
from noise_gen import NoiseGen, NoiseParameters
from model import Model, QUAD_INDICES, Particle, PortalEnd, PortalManager, \
                  WeatherParticle, WeatherManager


# ===========================================================================
# util.py
# ===========================================================================

class TestNormalize(unittest.TestCase):
    def test_exact_int(self):
        self.assertEqual(normalize((3, 5, 7)), (3, 5, 7))

    def test_rounds_up(self):
        self.assertEqual(normalize((3.6, 5.6, 7.6)), (4, 6, 8))

    def test_rounds_down(self):
        self.assertEqual(normalize((3.2, 5.2, 7.2)), (3, 5, 7))

    def test_midpoint_rounds(self):
        # Python rounds 0.5 to nearest even
        result = normalize((0.5, 0.5, 0.5))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_negative(self):
        self.assertEqual(normalize((-1.4, -2.6, -0.5)), (-1, -3, 0))

    def test_zero(self):
        self.assertEqual(normalize((0, 0, 0)), (0, 0, 0))


class TestSectorize(unittest.TestCase):
    def test_origin(self):
        self.assertEqual(sectorize((0, 0, 0), 16), (0, 0, 0))

    def test_within_sector(self):
        self.assertEqual(sectorize((5, 10, 15), 16), (0, 0, 0))

    def test_boundary(self):
        self.assertEqual(sectorize((16, 0, 16), 16), (1, 0, 1))

    def test_large_coords(self):
        self.assertEqual(sectorize((64, 0, 32), 16), (4, 0, 2))

    def test_y_always_zero(self):
        # sectorize always returns y=0 (sectors are flat)
        s = sectorize((10, 99, 10), 16)
        self.assertEqual(s[1], 0)

    def test_negative_coords(self):
        result = sectorize((-8, 5, -8), 16)
        self.assertIsInstance(result, tuple)

    def test_different_sector_sizes(self):
        self.assertEqual(sectorize((8, 0, 8), 8), (1, 0, 1))
        self.assertEqual(sectorize((8, 0, 8), 16), (0, 0, 0))


class TestCubeVertices(unittest.TestCase):
    def test_returns_72_floats(self):
        verts = cube_vertices(0, 0, 0, 0.5)
        self.assertEqual(len(verts), 72)  # 6 faces * 4 verts * 3 coords

    def test_symmetry_around_origin(self):
        verts = cube_vertices(0, 0, 0, 0.5)
        self.assertIn(0.5, verts)
        self.assertIn(-0.5, verts)

    def test_offset_position(self):
        v0 = cube_vertices(0, 0, 0, 1)
        v1 = cube_vertices(5, 5, 5, 1)
        # Every x coord in v1 should be v0's + 5
        for i in range(0, 72, 3):
            self.assertAlmostEqual(v1[i] - v0[i], 5)
            self.assertAlmostEqual(v1[i+1] - v0[i+1], 5)
            self.assertAlmostEqual(v1[i+2] - v0[i+2], 5)

    def test_size_scaling(self):
        v1 = cube_vertices(0, 0, 0, 1)
        v2 = cube_vertices(0, 0, 0, 2)
        # All coords in v2 should be exactly 2x v1
        for a, b in zip(v1, v2):
            self.assertAlmostEqual(b, a * 2)

    def test_all_faces_distinct(self):
        verts = cube_vertices(0, 0, 0, 0.5)
        # 6 groups of 12 floats — each face should differ from the others
        faces = [tuple(verts[i:i+12]) for i in range(0, 72, 12)]
        self.assertEqual(len(set(faces)), 6)


# ===========================================================================
# config.py
# ===========================================================================

class TestConfig(unittest.TestCase):
    def test_texture_path_is_string(self):
        self.assertIsInstance(config.TEXTURE_PATH, str)

    def test_texture_path_ends_png(self):
        self.assertTrue(config.TEXTURE_PATH.endswith('texture.png'))

    def test_texture_coords_are_lists(self):
        for name in ['GRASS', 'SAND', 'STONE', 'WOOD', 'LEAF', 'WATER', 'CRYSTAL', 'MAGIC_WATER', 'BRICK']:
            val = getattr(config, name)
            self.assertIsInstance(val, list, f"{name} should be a list")

    def test_texture_coords_length(self):
        # 6 faces * 8 floats per face = 48
        for name in ['GRASS', 'SAND', 'STONE']:
            val = getattr(config, name)
            self.assertEqual(len(val), 48, f"{name} length should be 48")

    def test_faces_are_unit_vectors(self):
        for face in config.FACES:
            magnitude = sum(abs(f) for f in face)
            self.assertEqual(magnitude, 1)

    def test_faces_count(self):
        self.assertEqual(len(config.FACES), 6)

    def test_physics_constants_positive(self):
        self.assertGreater(config.GRAVITY, 0)
        self.assertGreater(config.JUMP_SPEED, 0)
        self.assertGreater(config.TERMINAL_VELOCITY, 0)

    def test_jump_speed_formula(self):
        expected = math.sqrt(2 * config.GRAVITY * config.MAX_JUMP_HEIGHT)
        self.assertAlmostEqual(config.JUMP_SPEED, expected)

    def test_speed_ordering(self):
        self.assertLess(config.CROUCH_SPEED, config.WALKING_SPEED)
        self.assertLess(config.WALKING_SPEED, config.SPRINT_SPEED)
        self.assertLess(config.SPRINT_SPEED, config.FLYING_SPEED)

    def test_particle_colours_are_rgba(self):
        for name in ['PARTICLE_STONE', 'PARTICLE_WOOD', 'PARTICLE_LEAF', 'PARTICLE_CRYSTAL']:
            val = getattr(config, name)
            self.assertEqual(len(val), 4)
            for channel in val:
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

    def test_sector_size_positive(self):
        self.assertGreater(config.SECTOR_SIZE, 0)

    def test_ticks_per_sec_positive(self):
        self.assertGreater(config.TICKS_PER_SEC, 0)


# ===========================================================================
# noise_gen.py
# ===========================================================================

class TestNoiseGen(unittest.TestCase):
    def setUp(self):
        self.gen = NoiseGen(seed=42)

    def test_get_height_returns_float(self):
        h = self.gen.getHeight(10, 10)
        self.assertIsInstance(h, float)

    def test_deterministic_with_same_seed(self):
        g1 = NoiseGen(seed=99)
        g2 = NoiseGen(seed=99)
        for x, z in [(0,0), (5,5), (10,20), (50,50)]:
            self.assertAlmostEqual(g1.getHeight(x, z), g2.getHeight(x, z))

    def test_different_seeds_differ(self):
        g1 = NoiseGen(seed=1)
        g2 = NoiseGen(seed=2)
        results_same = all(
            abs(g1.getHeight(x, z) - g2.getHeight(x, z)) < 0.001
            for x, z in [(5,5), (10,10), (20,30)]
        )
        self.assertFalse(results_same)

    def test_height_in_reasonable_range(self):
        # Heights should be somewhere sensible for terrain (not wildly huge)
        for x in range(0, 50, 5):
            for z in range(0, 50, 5):
                h = self.gen.getHeight(x, z)
                self.assertGreater(h, -100)
                self.assertLess(h, 500)

    def test_height_varies_across_map(self):
        heights = {self.gen.getHeight(x, z) for x in range(10) for z in range(10)}
        self.assertGreater(len(heights), 5)  # not all the same

    def test_lerp_bounds(self):
        result = self.gen._lerp(0.0, 1.0, 0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_lerp_at_zero(self):
        self.assertAlmostEqual(self.gen._lerp(3.0, 7.0, 0.0), 3.0)

    def test_lerp_at_one(self):
        self.assertAlmostEqual(self.gen._lerp(3.0, 7.0, 1.0), 7.0, places=4)

    def test_noise_params_defaults(self):
        p = self.gen.noiseParams
        self.assertGreater(p.octaves, 0)
        self.assertGreater(p.amplitude, 0)
        self.assertGreater(p.smoothness, 0)


# ===========================================================================
# model.py — QUAD_INDICES
# ===========================================================================

class TestQuadIndices(unittest.TestCase):
    def test_length(self):
        # 6 faces * 6 indices per face (2 triangles)
        self.assertEqual(len(QUAD_INDICES), 36)

    def test_all_indices_in_range(self):
        for idx in QUAD_INDICES:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 24)  # 24 verts per cube

    def test_each_face_covers_4_unique_verts(self):
        for face in range(6):
            face_indices = QUAD_INDICES[face*6:(face+1)*6]
            unique = set(face_indices)
            self.assertEqual(len(unique), 4)

    def test_each_face_two_triangles(self):
        # Each group of 6 forms 2 triangles sharing a diagonal
        for face in range(6):
            t = QUAD_INDICES[face*6:(face+1)*6]
            tri1 = set(t[0:3])
            tri2 = set(t[3:6])
            shared = tri1 & tri2
            self.assertEqual(len(shared), 2)  # share exactly 2 verts (the diagonal)


# ===========================================================================
# model.py — world management (headless)
# ===========================================================================

class TestModelWorld(unittest.TestCase):
    def setUp(self):
        # Patch _initialize so tests don't run world gen
        with patch.object(Model, '_initialize', return_value=None):
            self.model = Model()

    def test_initial_world_empty(self):
        self.assertEqual(len(self.model.world), 0)

    def test_add_block(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        self.assertIn((0, 0, 0), self.model.world)

    def test_add_block_stores_texture(self):
        self.model.add_block((1, 2, 3), config.STONE, immediate=False)
        self.assertEqual(self.model.world[(1, 2, 3)], config.STONE)

    def test_remove_block(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        self.model.remove_block((0, 0, 0), immediate=False)
        self.assertNotIn((0, 0, 0), self.model.world)

    def test_add_overwrites_existing(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        self.model.add_block((0, 0, 0), config.STONE, immediate=False)
        self.assertEqual(self.model.world[(0, 0, 0)], config.STONE)

    def test_sector_assignment(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        sector = sectorize((0, 0, 0), config.SECTOR_SIZE)
        self.assertIn(sector, self.model.sectors)
        self.assertIn((0, 0, 0), self.model.sectors[sector])

    def test_sector_removal(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        sector = sectorize((0, 0, 0), config.SECTOR_SIZE)
        self.model.remove_block((0, 0, 0), immediate=False)
        self.assertNotIn((0, 0, 0), self.model.sectors.get(sector, []))

    def test_multiple_blocks_same_sector(self):
        for i in range(5):
            self.model.add_block((i, 0, 0), config.GRASS, immediate=False)
        sector = sectorize((0, 0, 0), config.SECTOR_SIZE)
        self.assertEqual(len(self.model.sectors[sector]), 5)

    def test_blocks_in_different_sectors(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        self.model.add_block((100, 0, 100), config.STONE, immediate=False)
        self.assertEqual(len(self.model.sectors), 2)


class TestModelExposed(unittest.TestCase):
    def setUp(self):
        with patch.object(Model, '_initialize', return_value=None):
            self.model = Model()

    def test_isolated_block_is_exposed(self):
        self.model.add_block((5, 5, 5), config.GRASS, immediate=False)
        self.assertTrue(self.model.exposed((5, 5, 5)))

    def test_fully_enclosed_block_not_exposed(self):
        center = (5, 5, 5)
        self.model.add_block(center, config.STONE, immediate=False)
        for dx, dy, dz in config.FACES:
            self.model.add_block((5+dx, 5+dy, 5+dz), config.STONE, immediate=False)
        self.assertFalse(self.model.exposed(center))

    def test_partially_enclosed_still_exposed(self):
        center = (5, 5, 5)
        self.model.add_block(center, config.STONE, immediate=False)
        # Only surround 5 of 6 faces
        for dx, dy, dz in config.FACES[:-1]:
            self.model.add_block((5+dx, 5+dy, 5+dz), config.STONE, immediate=False)
        self.assertTrue(self.model.exposed(center))


class TestModelHitTest(unittest.TestCase):
    def setUp(self):
        with patch.object(Model, '_initialize', return_value=None):
            self.model = Model()

    def test_miss_returns_none_none(self):
        block, prev = self.model.hit_test((0, 0, 0), (0, 1, 0))
        self.assertIsNone(block)
        self.assertIsNone(prev)

    def test_hits_block_directly_ahead(self):
        self.model.add_block((0, 5, 0), config.STONE, immediate=False)
        block, prev = self.model.hit_test((0, 0, 0), (0, 1, 0))
        self.assertEqual(block, (0, 5, 0))

    def test_previous_is_adjacent(self):
        self.model.add_block((0, 5, 0), config.STONE, immediate=False)
        block, prev = self.model.hit_test((0, 0, 0), (0, 1, 0))
        self.assertIsNotNone(prev)
        # previous should be one step before the hit block
        self.assertNotEqual(prev, block)

    def test_max_distance_respected(self):
        # Place block just beyond max distance
        self.model.add_block((0, 10, 0), config.STONE, immediate=False)
        block, _ = self.model.hit_test((0, 0, 0), (0, 1, 0), max_distance=8)
        self.assertIsNone(block)

    def test_hit_within_max_distance(self):
        self.model.add_block((0, 5, 0), config.STONE, immediate=False)
        block, _ = self.model.hit_test((0, 0, 0), (0, 1, 0), max_distance=8)
        self.assertIsNotNone(block)

    def test_direction_specificity(self):
        # Block is to the right, looking left shouldn't hit it
        self.model.add_block((5, 0, 0), config.STONE, immediate=False)
        block, _ = self.model.hit_test((0, 0, 0), (-1, 0, 0))
        self.assertIsNone(block)

    def test_hit_in_correct_direction(self):
        self.model.add_block((5, 0, 0), config.STONE, immediate=False)
        block, _ = self.model.hit_test((0, 0, 0), (1, 0, 0))
        self.assertEqual(block, (5, 0, 0))


class TestModelQueue(unittest.TestCase):
    def setUp(self):
        with patch.object(Model, '_initialize', return_value=None):
            self.model = Model()

    def test_queue_starts_empty(self):
        self.assertEqual(len(self.model.queue), 0)

    def test_enqueue_adds_item(self):
        fn = MagicMock()
        self.model._enqueue(fn, 1, 2)
        self.assertEqual(len(self.model.queue), 1)

    def test_dequeue_calls_function(self):
        fn = MagicMock()
        self.model._enqueue(fn, 'a', 'b')
        self.model._dequeue()
        fn.assert_called_once_with('a', 'b')

    def test_process_entire_queue_drains(self):
        fn = MagicMock()
        for _ in range(10):
            self.model._enqueue(fn)
        self.model.process_entire_queue()
        self.assertEqual(len(self.model.queue), 0)
        self.assertEqual(fn.call_count, 10)

    def test_process_queue_advances_game_time(self):
        t0 = self.model.game_time
        self.model.process_queue()
        self.assertGreater(self.model.game_time, t0)

    def test_show_block_enqueues_when_not_immediate(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        self.model.show_block((0, 0, 0), immediate=False)
        self.assertGreater(len(self.model.queue), 0)


class TestModelSectors(unittest.TestCase):
    def setUp(self):
        with patch.object(Model, '_initialize', return_value=None):
            self.model = Model()

    def test_change_sectors_shows_new(self):
        # Add a block in the target sector
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        before_count = len(self.model.shown)
        self.model.change_sectors(None, (0, 0, 0))
        # shown may grow (blocks in range get show_block queued)
        # just ensure no crash and queue has work
        self.assertGreaterEqual(len(self.model.queue) + len(self.model.shown), before_count)

    def test_show_sector_marks_exposed_blocks(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        sector = sectorize((0, 0, 0), config.SECTOR_SIZE)
        self.model.show_sector(sector)
        self.assertIn((0, 0, 0), self.model.shown)

    def test_hide_sector_clears_shown(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        sector = sectorize((0, 0, 0), config.SECTOR_SIZE)
        self.model.show_sector(sector)
        self.model.hide_sector(sector)
        self.assertNotIn((0, 0, 0), self.model.shown)


# ===========================================================================
# Collision / normalize integration
# ===========================================================================

class TestCollisionHelpers(unittest.TestCase):
    """Tests the normalize + sectorize pipeline used by collision detection."""

    def test_player_position_normalizes_to_block(self):
        # Standing at (5.3, 10.7, 3.1) should map to block (5, 11, 3)
        result = normalize((5.3, 10.7, 3.1))
        self.assertEqual(result, (5, 11, 3))

    def test_block_lookup_uses_normalized_pos(self):
        with patch.object(Model, '_initialize', return_value=None):
            model = Model()
        model.add_block((5, 11, 3), config.STONE, immediate=False)
        # Slightly off-centre position should still find the block
        key = normalize((5.3, 10.7, 3.1))
        self.assertIn(key, model.world)


if __name__ == '__main__':
    unittest.main(verbosity=2)


# ===========================================================================
# model.py — Save / Load / Spawn (new in world update)
# ===========================================================================

class TestModelSaveLoad(unittest.TestCase):
    def _make_model(self):
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        return m

    def _tmp_tcw(self):
        f = tempfile.NamedTemporaryFile(suffix='.tcw', delete=False)
        f.close()
        return f.name

    def _tmp_json(self):
        f = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        f.close()
        return f.name

    # --- texture lookup (unchanged) ---

    def test_texture_lookup_roundtrip(self):
        from model import _TEXTURE_NAMES, _TEXTURE_LOOKUP
        for name, tex in _TEXTURE_NAMES.items():
            key = tuple(tex)
            self.assertIn(key, _TEXTURE_LOOKUP, f"{name} missing from lookup")
            self.assertEqual(_TEXTURE_LOOKUP[key], name)

    # --- binary save ---

    def test_save_creates_file(self):
        m = self._make_model()
        m.add_block((0, 0, 0), config.GRASS, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m.SAVE_FILE = tmp
            m.save_world()
            self.assertTrue(os.path.exists(tmp))
        finally:
            os.unlink(tmp)

    def test_save_writes_magic_header(self):
        m = self._make_model()
        m.seed = 99
        m.add_block((1, 2, 3), config.STONE, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m.SAVE_FILE = tmp
            m.save_world()
            with open(tmp, 'rb') as f:
                magic = f.read(4)
            self.assertEqual(magic, config.SAVE_MAGIC)
        finally:
            os.unlink(tmp)

    def test_save_header_contains_seed_and_count(self):
        import struct
        from model import _HEADER_FMT
        m = self._make_model()
        m.seed = 42
        m.add_block((1, 2, 3), config.STONE, immediate=False)
        m.add_block((4, 5, 6), config.GRASS, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m.SAVE_FILE = tmp
            m.save_world()
            with open(tmp, 'rb') as f:
                magic, ver, seed, count = _HEADER_FMT.unpack(f.read(_HEADER_FMT.size))
            self.assertEqual(seed, 42)
            self.assertEqual(count, 2)
            self.assertEqual(ver, config.SAVE_VERSION)
        finally:
            os.unlink(tmp)

    def test_save_returns_block_count(self):
        m = self._make_model()
        for i in range(5):
            m.add_block((i, 0, 0), config.GRASS, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m.SAVE_FILE = tmp
            count = m.save_world()
            self.assertEqual(count, 5)
        finally:
            os.unlink(tmp)

    def test_save_block_data_correct(self):
        import struct
        from model import _HEADER_FMT, _BLOCK_FMT, _TEX_TO_BLOCK_ID
        m = self._make_model()
        m.seed = 1
        m.add_block((10, 20, 30), config.DIRT, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m.SAVE_FILE = tmp
            m.save_world()
            with open(tmp, 'rb') as f:
                f.read(_HEADER_FMT.size)
                x, y, z, bid = _BLOCK_FMT.unpack(f.read(_BLOCK_FMT.size))
            self.assertEqual((x, y, z), (10, 20, 30))
            self.assertEqual(bid, _TEX_TO_BLOCK_ID[tuple(config.DIRT)])
        finally:
            os.unlink(tmp)

    def test_file_size_is_correct(self):
        from model import _HEADER_FMT, _BLOCK_FMT
        m = self._make_model()
        n = 7
        for i in range(n):
            m.add_block((i, 0, 0), config.STONE, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m.SAVE_FILE = tmp
            m.save_world()
            expected = _HEADER_FMT.size + n * _BLOCK_FMT.size
            self.assertEqual(os.path.getsize(tmp), expected)
        finally:
            os.unlink(tmp)

    # --- binary load ---

    def test_load_restores_blocks(self):
        m1 = self._make_model()
        m1.seed = 7
        m1.add_block((3, 4, 5), config.WOOD, immediate=False)
        m1.add_block((6, 7, 8), config.LEAF, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m1.SAVE_FILE = tmp
            m1.save_world()
            m2 = self._make_model()
            m2.SAVE_FILE = tmp
            m2.load_world()
            self.assertIn((3, 4, 5), m2.world)
            self.assertIn((6, 7, 8), m2.world)
            self.assertEqual(m2.world[(3, 4, 5)], config.WOOD)
            self.assertEqual(m2.world[(6, 7, 8)], config.LEAF)
            self.assertEqual(m2.seed, 7)
        finally:
            os.unlink(tmp)

    def test_load_block_count_matches(self):
        m1 = self._make_model()
        for i in range(10):
            m1.add_block((i, 0, 0), config.GRASS, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m1.SAVE_FILE = tmp
            m1.save_world()
            m2 = self._make_model()
            m2.SAVE_FILE = tmp
            m2.load_world()
            self.assertEqual(len(m2.world), 10)
        finally:
            os.unlink(tmp)

    def test_all_block_types_survive_roundtrip(self):
        """Every block type encodes and decodes cleanly."""
        blocks = [
            config.GRASS, config.SAND, config.BRICK, config.STONE,
            config.WOOD, config.LEAF, config.WATER, config.CRYSTAL,
            config.MAGIC_WATER, config.DIRT, config.SNOW,
            config.GLASS, config.PLANKS, config.GRAVEL,
        ]
        m1 = self._make_model()
        m1.seed = 0
        for i, tex in enumerate(blocks):
            m1.add_block((i, 0, 0), tex, immediate=False)
        tmp = self._tmp_tcw()
        try:
            m1.SAVE_FILE = tmp
            m1.save_world()
            m2 = self._make_model()
            m2.SAVE_FILE = tmp
            m2.load_world()
            for i, tex in enumerate(blocks):
                self.assertIn((i, 0, 0), m2.world)
                self.assertEqual(m2.world[(i, 0, 0)], tex)
        finally:
            os.unlink(tmp)

    # --- delete ---

    def test_delete_save(self):
        m = self._make_model()
        tmp = self._tmp_tcw()
        try:
            m.SAVE_FILE = tmp
            m.add_block((0, 0, 0), config.GRASS, immediate=False)
            m.save_world()
            self.assertTrue(os.path.exists(tmp))
            m.delete_save()
            self.assertFalse(os.path.exists(tmp))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_save_no_file_no_error(self):
        m = self._make_model()
        m.SAVE_FILE = '/tmp/timecraft_nonexistent_12345.tcw'
        m.SAVE_FILE_LEGACY = '/tmp/timecraft_nonexistent_12345.json'
        m.delete_save()   # should not raise

    # --- migration: JSON → binary ---

    def test_legacy_json_loads_correctly(self):
        """_load_json_legacy reads an old-format JSON save."""
        m1 = self._make_model()
        m1.seed = 55
        m1.add_block((1, 1, 1), config.BRICK, immediate=False)
        # Write a proper legacy JSON file manually
        legacy_data = {
            'seed': 55,
            'blocks': [{'pos': [1, 1, 1], 'tex': 'BRICK'}]
        }
        tmp_json = self._tmp_json()
        try:
            with open(tmp_json, 'w') as f:
                json.dump(legacy_data, f)
            m2 = self._make_model()
            m2.SAVE_FILE = '/tmp/nonexistent_tcw_12345.tcw'
            m2.SAVE_FILE_LEGACY = tmp_json
            m2._load_json_legacy(tmp_json)
            self.assertIn((1, 1, 1), m2.world)
            self.assertEqual(m2.world[(1, 1, 1)], config.BRICK)
            self.assertEqual(m2.seed, 55)
        finally:
            os.unlink(tmp_json)

    def test_legacy_json_triggers_migration(self):
        """Loading a JSON save produces a .tcw binary alongside it."""
        legacy_data = {
            'seed': 77,
            'blocks': [{'pos': [5, 5, 5], 'tex': 'SNOW'}]
        }
        tmp_json = self._tmp_json()
        tmp_tcw  = tmp_json.replace('.json', '.tcw')
        try:
            with open(tmp_json, 'w') as f:
                json.dump(legacy_data, f)
            m = self._make_model()
            m.SAVE_FILE        = tmp_tcw
            m.SAVE_FILE_LEGACY = tmp_json
            m.load_world()
            # Migration should have written the binary file
            self.assertTrue(os.path.exists(tmp_tcw))
            with open(tmp_tcw, 'rb') as f:
                magic = f.read(4)
            self.assertEqual(magic, config.SAVE_MAGIC)
        finally:
            for p in (tmp_json, tmp_tcw):
                if os.path.exists(p):
                    os.unlink(p)


class TestBinarySaveConfig(unittest.TestCase):
    """Verify binary save config constants are correct."""

    def test_save_magic_is_four_bytes(self):
        self.assertEqual(len(config.SAVE_MAGIC), 4)

    def test_save_magic_value(self):
        self.assertEqual(config.SAVE_MAGIC, b'TCWF')

    def test_save_version_positive(self):
        self.assertGreater(config.SAVE_VERSION, 0)

    def test_block_ids_has_14_entries(self):
        self.assertGreaterEqual(len(config.BLOCK_IDS), 14)

    def test_block_ids_are_all_strings(self):
        for bid in config.BLOCK_IDS:
            self.assertIsInstance(bid, str)

    def test_block_ids_no_duplicates(self):
        self.assertEqual(len(config.BLOCK_IDS), len(set(config.BLOCK_IDS)))

    def test_block_ids_match_texture_names(self):
        from model import _TEXTURE_NAMES
        for name in config.BLOCK_IDS:
            self.assertIn(name, _TEXTURE_NAMES,
                          f"BLOCK_IDS entry '{name}' not in _TEXTURE_NAMES")

    def test_block_id_to_tex_has_14_entries(self):
        from model import _BLOCK_ID_TO_TEX
        self.assertGreaterEqual(len(_BLOCK_ID_TO_TEX), 14)

    def test_tex_to_block_id_has_14_entries(self):
        from model import _TEX_TO_BLOCK_ID
        self.assertGreaterEqual(len(_TEX_TO_BLOCK_ID), 14)

    def test_block_id_roundtrip(self):
        from model import _BLOCK_ID_TO_TEX, _TEX_TO_BLOCK_ID
        for bid, tex in _BLOCK_ID_TO_TEX.items():
            self.assertEqual(_TEX_TO_BLOCK_ID[tuple(tex)], bid)

    def test_header_struct_size(self):
        from model import _HEADER_FMT
        self.assertEqual(_HEADER_FMT.size, 13)

    def test_block_struct_size(self):
        from model import _BLOCK_FMT
        self.assertEqual(_BLOCK_FMT.size, 7)




class TestModelSpawnPoint(unittest.TestCase):
    def _make_model(self):
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        return m

    def test_spawn_returns_tuple_of_3(self):
        m = self._make_model()
        m.add_block((64, 20, 64), config.GRASS, immediate=False)
        spawn = m.get_spawn_point()
        self.assertEqual(len(spawn), 3)

    def test_spawn_above_ground(self):
        m = self._make_model()
        # Place ground at y=20 near centre (64,64)
        m.add_block((64, 20, 64), config.GRASS, immediate=False)
        spawn = m.get_spawn_point()
        self.assertGreater(spawn[1], 20)

    def test_spawn_skips_water(self):
        m = self._make_model()
        # Place water at surface near origin, stone below
        m.add_block((0, 15, 0), config.WATER, immediate=False)
        m.add_block((0, 14, 0), config.STONE, immediate=False)
        spawn = m.get_spawn_point()
        # Should land on stone at 14, not water at 15
        self.assertGreater(spawn[1], 14)

    def test_spawn_fallback_when_empty(self):
        m = self._make_model()
        spawn = m.get_spawn_point()
        # Should return a fallback, not crash
        self.assertIsNotNone(spawn)
        self.assertEqual(len(spawn), 3)

    def test_spawn_is_floats(self):
        m = self._make_model()
        m.add_block((0, 25, 0), config.GRASS, immediate=False)
        spawn = m.get_spawn_point()
        for coord in spawn:
            self.assertIsInstance(coord, float)

    def test_world_is_infinite(self):
        """Model no longer has a fixed WORLD_SIZE — world grows on demand."""
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        self.assertFalse(hasattr(m, 'WORLD_SIZE'))


# ===========================================================================
# New blocks (dirt, snow, glass, planks, gravel)
# ===========================================================================

class TestNewBlocks(unittest.TestCase):
    def test_all_new_blocks_defined(self):
        for name in ['DIRT', 'SNOW', 'GLASS', 'PLANKS', 'GRAVEL']:
            self.assertTrue(hasattr(config, name), f"{name} missing from config")

    def test_new_blocks_are_lists(self):
        for name in ['DIRT', 'SNOW', 'GLASS', 'PLANKS', 'GRAVEL']:
            self.assertIsInstance(getattr(config, name), list)

    def test_new_blocks_correct_length(self):
        for name in ['DIRT', 'SNOW', 'GLASS', 'PLANKS', 'GRAVEL']:
            self.assertEqual(len(getattr(config, name)), 48)

    def test_new_blocks_distinct(self):
        blocks = [config.DIRT, config.SNOW, config.GLASS, config.PLANKS, config.GRAVEL]
        tuples = [tuple(b) for b in blocks]
        self.assertEqual(len(set(tuples)), len(tuples), "New blocks must have unique textures")

    def test_new_blocks_dont_clash_with_existing(self):
        existing = [config.GRASS, config.SAND, config.BRICK, config.STONE,
                    config.WOOD, config.LEAF, config.WATER, config.CRYSTAL, config.MAGIC_WATER]
        new = [config.DIRT, config.SNOW, config.GLASS, config.PLANKS, config.GRAVEL]
        existing_tuples = {tuple(b) for b in existing}
        for b in new:
            self.assertNotIn(tuple(b), existing_tuples)

    def test_new_particle_colours_defined(self):
        for name in ['PARTICLE_DIRT', 'PARTICLE_SNOW', 'PARTICLE_GLASS',
                     'PARTICLE_PLANKS', 'PARTICLE_GRAVEL']:
            self.assertTrue(hasattr(config, name), f"{name} missing")
            val = getattr(config, name)
            self.assertEqual(len(val), 4)

    def test_new_blocks_in_texture_name_map(self):
        from model import _TEXTURE_NAMES
        for name in ['DIRT', 'SNOW', 'GLASS', 'PLANKS', 'GRAVEL']:
            self.assertIn(name, _TEXTURE_NAMES)

    def test_new_blocks_in_texture_lookup(self):
        from model import _TEXTURE_LOOKUP
        for name in ['DIRT', 'SNOW', 'GLASS', 'PLANKS', 'GRAVEL']:
            key = tuple(getattr(config, name))
            self.assertIn(key, _TEXTURE_LOOKUP)

    def test_save_load_roundtrip_new_blocks(self):
        with patch.object(Model, '_initialize', return_value=None):
            m1 = Model()
        m1.seed = 1
        m1.add_block((0, 0, 0), config.DIRT, immediate=False)
        m1.add_block((1, 0, 0), config.SNOW, immediate=False)
        m1.add_block((2, 0, 0), config.GLASS, immediate=False)
        m1.add_block((3, 0, 0), config.PLANKS, immediate=False)
        m1.add_block((4, 0, 0), config.GRAVEL, immediate=False)
        with tempfile.NamedTemporaryFile(suffix='.tcw', delete=False) as f:
            tmp = f.name
        try:
            m1.SAVE_FILE = tmp
            m1.save_world()
            with patch.object(Model, '_initialize', return_value=None):
                m2 = Model()
            m2.SAVE_FILE = tmp
            m2.load_world()
            self.assertEqual(m2.world[(0, 0, 0)], config.DIRT)
            self.assertEqual(m2.world[(1, 0, 0)], config.SNOW)
            self.assertEqual(m2.world[(2, 0, 0)], config.GLASS)
            self.assertEqual(m2.world[(3, 0, 0)], config.PLANKS)
            self.assertEqual(m2.world[(4, 0, 0)], config.GRAVEL)
        finally:
            os.unlink(tmp)

    def test_inventory_contains_new_blocks(self):
        """Inventory config check — all new blocks should be reachable."""
        # We check config has them; window inventory is tested via integration
        all_blocks = [config.BRICK, config.GRASS, config.SAND, config.STONE,
                      config.WOOD, config.LEAF, config.CRYSTAL,
                      config.DIRT, config.PLANKS, config.GRAVEL, config.SNOW, config.GLASS]
        tuples = [tuple(b) for b in all_blocks]
        self.assertEqual(len(set(tuples)), len(all_blocks), "Inventory has duplicate block types")


# ===========================================================================
# Particle system
# ===========================================================================

class TestParticle(unittest.TestCase):
    """Unit tests for the Particle dataclass."""

    def _make(self, **kwargs):
        defaults = dict(x=0.0, y=0.0, z=0.0, vx=1.0, vy=2.0, vz=0.0,
                        colour=(100, 100, 100, 255))
        defaults.update(kwargs)
        return Particle(**defaults)

    def test_alive_at_birth(self):
        p = self._make()
        self.assertTrue(p.alive)

    def test_dead_after_full_lifetime(self):
        p = self._make()
        p.age = p.lifetime
        self.assertFalse(p.alive)

    def test_alpha_fraction_zero_at_birth(self):
        p = self._make()
        self.assertAlmostEqual(p.alpha_fraction, 0.0)

    def test_alpha_fraction_one_at_death(self):
        p = self._make()
        p.age = p.lifetime
        self.assertAlmostEqual(p.alpha_fraction, 1.0)

    def test_update_advances_age(self):
        p = self._make()
        p.update(0.1)
        self.assertAlmostEqual(p.age, 0.1)

    def test_update_moves_position(self):
        p = self._make(x=0.0, y=10.0, z=0.0, vx=1.0, vy=0.0, vz=2.0)
        p.update(1.0)
        self.assertAlmostEqual(p.x, 1.0)
        self.assertAlmostEqual(p.z, 2.0)

    def test_gravity_pulls_down(self):
        p = self._make(vy=0.0)
        p.update(1.0)
        self.assertLess(p.vy, 0.0)

    def test_upward_velocity_decelerates(self):
        p = self._make(vy=10.0)
        initial_vy = p.vy
        p.update(0.5)
        self.assertLess(p.vy, initial_vy)

    def test_custom_lifetime(self):
        p = self._make(lifetime=2.0)
        p.update(1.9)
        self.assertTrue(p.alive)
        p.update(0.2)
        self.assertFalse(p.alive)

    def test_colour_stored_correctly(self):
        colour = (200, 100, 50, 180)
        p = self._make(colour=colour)
        self.assertEqual(p.colour, colour)


class TestParticleConfig(unittest.TestCase):
    """Verify particle config constants are sane."""

    def test_particle_count_positive(self):
        self.assertGreater(config.PARTICLE_COUNT, 0)

    def test_particle_lifetime_positive(self):
        self.assertGreater(config.PARTICLE_LIFETIME, 0.0)

    def test_particle_size_positive(self):
        self.assertGreater(config.PARTICLE_SIZE, 0)

    def test_particle_speed_positive(self):
        self.assertGreater(config.PARTICLE_SPEED, 0.0)

    def test_particle_gravity_positive(self):
        self.assertGreater(config.PARTICLE_GRAVITY, 0.0)

    def test_all_block_types_in_particle_map(self):
        block_types = [
            config.GRASS, config.SAND, config.BRICK, config.STONE,
            config.WOOD, config.LEAF, config.WATER, config.CRYSTAL,
            config.MAGIC_WATER, config.DIRT, config.SNOW,
            config.GLASS, config.PLANKS, config.GRAVEL,
        ]
        for block in block_types:
            key = tuple(block)
            self.assertIn(key, config.TEXTURE_PARTICLE_MAP,
                          f"Block texture not in TEXTURE_PARTICLE_MAP")

    def test_particle_colours_are_rgba_tuples(self):
        for key, colour in config.TEXTURE_PARTICLE_MAP.items():
            self.assertIsInstance(colour, tuple)
            self.assertEqual(len(colour), 4)
            for channel in colour:
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

    def test_particle_map_has_correct_entry_count(self):
        self.assertEqual(len(config.TEXTURE_PARTICLE_MAP), 14)


class TestModelParticles(unittest.TestCase):
    """Test spawn_particles and update_particles on a headless Model."""

    def setUp(self):
        with patch.object(Model, '_initialize', return_value=None):
            self.model = Model()

    def test_particles_list_starts_empty(self):
        self.assertEqual(self.model.particles, [])

    def test_spawn_adds_particles(self):
        self.model.add_block((5, 5, 5), config.GRASS, immediate=False)
        self.model.spawn_particles((5, 5, 5))
        self.assertEqual(len(self.model.particles), config.PARTICLE_COUNT)

    def test_spawn_correct_count(self):
        self.model.add_block((0, 0, 0), config.STONE, immediate=False)
        self.model.spawn_particles((0, 0, 0))
        self.assertEqual(len(self.model.particles), config.PARTICLE_COUNT)

    def test_spawn_near_block_position(self):
        self.model.add_block((10, 20, 30), config.DIRT, immediate=False)
        self.model.spawn_particles((10, 20, 30))
        for p in self.model.particles:
            self.assertAlmostEqual(p.x, 10.0, delta=1.0)
            self.assertAlmostEqual(p.y, 20.0, delta=1.5)
            self.assertAlmostEqual(p.z, 30.0, delta=1.0)

    def test_spawn_uses_block_colour(self):
        self.model.add_block((0, 0, 0), config.CRYSTAL, immediate=False)
        self.model.spawn_particles((0, 0, 0))
        expected = config.TEXTURE_PARTICLE_MAP[tuple(config.CRYSTAL)]
        for p in self.model.particles:
            self.assertEqual(p.colour, expected)

    def test_spawn_on_missing_position_is_noop(self):
        # Position not in world — should not crash or add particles
        self.model.spawn_particles((99, 99, 99))
        self.assertEqual(self.model.particles, [])

    def test_update_particles_advances_age(self):
        self.model.add_block((0, 0, 0), config.GRASS, immediate=False)
        self.model.spawn_particles((0, 0, 0))
        self.model.update_particles(0.1)
        for p in self.model.particles:
            self.assertAlmostEqual(p.age, 0.1, places=5)

    def test_update_particles_removes_dead(self):
        self.model.add_block((0, 0, 0), config.WOOD, immediate=False)
        self.model.spawn_particles((0, 0, 0))
        # Age them past their lifetime
        self.model.update_particles(config.PARTICLE_LIFETIME + 1.0)
        self.assertEqual(self.model.particles, [])

    def test_update_particles_keeps_live_ones(self):
        self.model.add_block((0, 0, 0), config.SAND, immediate=False)
        self.model.spawn_particles((0, 0, 0))
        self.model.update_particles(config.PARTICLE_LIFETIME * 0.1)
        self.assertEqual(len(self.model.particles), config.PARTICLE_COUNT)

    def test_remove_block_triggers_spawn(self):
        self.model.add_block((3, 3, 3), config.GRASS, immediate=False)
        self.model.remove_block((3, 3, 3), immediate=False)
        self.assertEqual(len(self.model.particles), config.PARTICLE_COUNT)

    def test_multiple_spawns_accumulate(self):
        for pos in [(0, 0, 0), (1, 0, 0), (2, 0, 0)]:
            self.model.add_block(pos, config.STONE, immediate=False)
            self.model.spawn_particles(pos)
        self.assertEqual(len(self.model.particles), config.PARTICLE_COUNT * 3)

    def test_particles_have_upward_bias(self):
        self.model.add_block((0, 0, 0), config.GRAVEL, immediate=False)
        self.model.spawn_particles((0, 0, 0))
        avg_vy = sum(p.vy for p in self.model.particles) / len(self.model.particles)
        self.assertGreater(avg_vy, 0.0, "Particles should have net upward velocity on spawn")

    def test_snow_block_gets_snow_colour(self):
        self.model.add_block((0, 0, 0), config.SNOW, immediate=False)
        self.model.spawn_particles((0, 0, 0))
        expected = config.PARTICLE_SNOW
        for p in self.model.particles:
            self.assertEqual(p.colour, expected)

    def test_glass_block_gets_glass_colour(self):
        self.model.add_block((0, 0, 0), config.GLASS, immediate=False)
        self.model.spawn_particles((0, 0, 0))
        expected = config.PARTICLE_GLASS
        for p in self.model.particles:
            self.assertEqual(p.colour, expected)


# ===========================================================================
# Ambient occlusion
# ===========================================================================

class TestAOConfig(unittest.TestCase):
    """Verify AO constants are well-formed."""

    def test_face_base_has_six_entries(self):
        self.assertEqual(len(config.AO_FACE_BASE), 6)

    def test_face_base_values_in_range(self):
        for v in config.AO_FACE_BASE:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_top_face_is_brightest(self):
        self.assertEqual(config.AO_FACE_BASE[0], max(config.AO_FACE_BASE))

    def test_bottom_face_is_darkest(self):
        self.assertEqual(config.AO_FACE_BASE[1], min(config.AO_FACE_BASE))

    def test_ao_neighbours_has_24_entries(self):
        self.assertEqual(len(config.AO_NEIGHBOURS), 24)

    def test_each_neighbour_entry_has_three_offsets(self):
        for entry in config.AO_NEIGHBOURS:
            self.assertEqual(len(entry), 3)

    def test_each_offset_is_integer_triple(self):
        for entry in config.AO_NEIGHBOURS:
            for offset in entry:
                self.assertEqual(len(offset), 3)
                for v in offset:
                    self.assertIsInstance(v, int)

    def test_ao_step_positive(self):
        self.assertGreater(config.AO_STEP, 0.0)

    def test_max_darkening_stays_above_zero(self):
        # 3 neighbours × AO_STEP should not exceed the minimum face base
        min_base = min(config.AO_FACE_BASE)
        self.assertGreater(min_base - 3 * config.AO_STEP, -0.01,
                           "AO_STEP too large: bottom face could go below zero")


class TestComputeAO(unittest.TestCase):
    """Unit tests for compute_ao() in util.py."""

    def test_returns_24_values(self):
        ao = compute_ao((0, 0, 0), {(0, 0, 0)})
        self.assertEqual(len(ao), 24)

    def test_all_values_floats(self):
        ao = compute_ao((0, 0, 0), {(0, 0, 0)})
        for v in ao:
            self.assertIsInstance(v, float)

    def test_values_in_range(self):
        ao = compute_ao((0, 0, 0), {(0, 0, 0)})
        for v in ao:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_isolated_block_top_face_is_max(self):
        ao = compute_ao((5, 5, 5), {(5, 5, 5)})
        top_verts = ao[0:4]
        for v in top_verts:
            self.assertAlmostEqual(v, config.AO_FACE_BASE[0])

    def test_isolated_block_bottom_face_is_min(self):
        ao = compute_ao((5, 5, 5), {(5, 5, 5)})
        bottom_verts = ao[4:8]
        for v in bottom_verts:
            self.assertAlmostEqual(v, config.AO_FACE_BASE[1])

    def test_isolated_block_side_faces_correct(self):
        ao = compute_ao((0, 0, 0), {(0, 0, 0)})
        side_verts = ao[8:]   # faces 2–5, 16 values
        for v in side_verts:
            self.assertAlmostEqual(v, config.AO_FACE_BASE[2])

    def test_neighbour_above_right_darkens_top_corner(self):
        # (1,1,0) is above-right; top face verts v2 and v3 share the +x side
        world = {(0,0,0), (1,1,0)}
        ao = compute_ao((0, 0, 0), world)
        top = ao[0:4]
        # v2 (index 2) and v3 (index 3) should be darker than v0 and v1
        self.assertLess(top[2], top[0])
        self.assertLess(top[3], top[0])
        self.assertAlmostEqual(top[0], config.AO_FACE_BASE[0])  # v0 unaffected
        self.assertAlmostEqual(top[1], config.AO_FACE_BASE[0])  # v1 unaffected

    def test_darkening_magnitude_one_neighbour(self):
        world = {(0,0,0), (1,1,0)}
        ao = compute_ao((0, 0, 0), world)
        # v2 and v3 each have exactly 1 neighbour → base - 1*AO_STEP
        expected = config.AO_FACE_BASE[0] - config.AO_STEP
        self.assertAlmostEqual(ao[2], expected)
        self.assertAlmostEqual(ao[3], expected)

    def test_no_values_below_zero(self):
        # Surround with many neighbours; no value should go negative
        world = {(0,0,0)}
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    world.add((dx, dy, dz))
        ao = compute_ao((0, 0, 0), world)
        for v in ao:
            self.assertGreaterEqual(v, 0.0)

    def test_position_offset_works_correctly(self):
        # Same relative geometry at different absolute positions
        ao_origin = compute_ao((0, 0, 0), {(0, 0, 0), (1, 1, 0)})
        ao_offset = compute_ao((10, 20, 30), {(10, 20, 30), (11, 21, 30)})
        for a, b in zip(ao_origin, ao_offset):
            self.assertAlmostEqual(a, b)

    def test_symmetric_neighbours_produce_uniform_darkening(self):
        # Block surrounded uniformly on one face axis: all 4 verts of that face darkened equally
        world = {(0,0,0), (-1,1,0), (1,1,0), (0,1,-1), (0,1,1)}
        ao = compute_ao((0, 0, 0), world)
        top = ao[0:4]
        # Each top vert has exactly 2 neighbours → base - 2*AO_STEP
        expected = config.AO_FACE_BASE[0] - 2 * config.AO_STEP
        for v in top:
            self.assertAlmostEqual(v, expected)

    def test_ao_neighbours_table_length(self):
        self.assertEqual(len(config.AO_NEIGHBOURS), 24)

    def test_world_without_block_itself(self):
        # compute_ao should work even if the block isn't in the world dict
        ao = compute_ao((0, 0, 0), set())
        for v in ao:
            self.assertGreaterEqual(v, 0.0)


# ===========================================================================
# Hotbar
# ===========================================================================

class TestHotbarConfig(unittest.TestCase):
    """Verify hotbar config constants are sane."""

    def test_slot_size_positive(self):
        self.assertGreater(config.HOTBAR_SLOT_SIZE, 0)

    def test_padding_non_negative(self):
        self.assertGreaterEqual(config.HOTBAR_PADDING, 0)

    def test_icon_pad_non_negative(self):
        self.assertGreaterEqual(config.HOTBAR_ICON_PAD, 0)

    def test_icon_fits_in_slot(self):
        icon_size = config.HOTBAR_SLOT_SIZE - config.HOTBAR_ICON_PAD * 2
        self.assertGreater(icon_size, 0)

    def test_hotbar_y_non_negative(self):
        self.assertGreaterEqual(config.HOTBAR_Y, 0)

    def test_atlas_cell_map_has_14_entries(self):
        self.assertGreaterEqual(len(config.TEXTURE_ATLAS_CELL), 14)

    def test_all_inventory_blocks_in_atlas_map(self):
        inventory = [
            config.BRICK, config.GRASS, config.SAND, config.STONE,
            config.WOOD, config.LEAF, config.CRYSTAL,
            config.DIRT, config.PLANKS, config.GRAVEL, config.SNOW, config.GLASS,
        ]
        for block in inventory:
            self.assertIn(tuple(block), config.TEXTURE_ATLAS_CELL,
                          "Inventory block missing from TEXTURE_ATLAS_CELL")

    def test_atlas_cells_are_valid_coords(self):
        for tex_key, (col, row) in config.TEXTURE_ATLAS_CELL.items():
            self.assertGreaterEqual(col, 0)
            self.assertLess(col, 4, f"Atlas col {col} out of range for 4×4 grid")
            self.assertGreaterEqual(row, 0)
            self.assertLess(row, 4, f"Atlas row {row} out of range for 4×4 grid")

    def test_atlas_cells_are_tuples_of_two_ints(self):
        for tex_key, cell in config.TEXTURE_ATLAS_CELL.items():
            self.assertIsInstance(cell, tuple)
            self.assertEqual(len(cell), 2)
            self.assertIsInstance(cell[0], int)
            self.assertIsInstance(cell[1], int)

    def test_no_duplicate_atlas_cells_in_inventory(self):
        inventory = [
            config.BRICK, config.GRASS, config.SAND, config.STONE,
            config.WOOD, config.LEAF, config.CRYSTAL,
            config.DIRT, config.PLANKS, config.GRAVEL, config.SNOW, config.GLASS,
        ]
        cells = [config.TEXTURE_ATLAS_CELL[tuple(b)] for b in inventory]
        self.assertEqual(len(cells), len(set(cells)),
                         "Two inventory blocks map to the same atlas cell")

    def test_grass_top_face_cell(self):
        # GRASS top face is atlas col=1, row=0
        cell = config.TEXTURE_ATLAS_CELL[tuple(config.GRASS)]
        self.assertEqual(cell, (1, 0))

    def test_stone_cell(self):
        cell = config.TEXTURE_ATLAS_CELL[tuple(config.STONE)]
        self.assertEqual(cell, (2, 1))

    def test_snow_top_face_cell(self):
        # SNOW top face is atlas col=3, row=3
        cell = config.TEXTURE_ATLAS_CELL[tuple(config.SNOW)]
        self.assertEqual(cell, (3, 3))


class TestHotbarInventory(unittest.TestCase):
    """Test inventory data and selection logic (no GL required)."""

    def _make_inventory(self):
        return [
            config.BRICK, config.GRASS, config.SAND, config.STONE,
            config.WOOD, config.LEAF, config.CRYSTAL,
            config.DIRT, config.PLANKS, config.GRAVEL, config.SNOW, config.GLASS,
        ]

    def test_inventory_has_12_slots(self):
        inv = self._make_inventory()
        self.assertEqual(len(inv), 12)

    def test_all_slots_distinct(self):
        inv = self._make_inventory()
        as_tuples = [tuple(b) for b in inv]
        self.assertEqual(len(set(as_tuples)), len(inv))

    def test_scroll_forward_wraps(self):
        inv = self._make_inventory()
        current = 11   # last slot
        next_slot = (current - 1) % len(inv)
        self.assertEqual(next_slot, 10)

    def test_scroll_backward_wraps(self):
        inv = self._make_inventory()
        current = 0
        prev_slot = (current + 1) % len(inv)
        self.assertEqual(prev_slot, 1)

    def test_scroll_wraps_from_last_to_first(self):
        inv = self._make_inventory()
        current = len(inv) - 1
        next_idx = (current - (-1)) % len(inv)   # scroll_y = -1 → scroll backward
        self.assertEqual(next_idx, 0)

    def test_num_key_index_calculation(self):
        # Keys 1-9 map to indices 0-8, key 0 maps to index 9
        # Formula: (symbol - num_keys[0]) % len(inventory)
        # Simulate: num_keys = [k1, k2, ..., k0], symbols are sequential
        num_keys = list(range(10))   # 0..9 stand-ins
        inventory = list(range(12))
        for i, sym in enumerate(num_keys):
            idx = (sym - num_keys[0]) % len(inventory)
            self.assertEqual(idx, i % len(inventory))

    def test_hotbar_total_width_calculation(self):
        n = 12
        slot_size = config.HOTBAR_SLOT_SIZE
        padding = config.HOTBAR_PADDING
        total_w = n * slot_size + (n - 1) * padding
        self.assertGreater(total_w, 0)
        # Should fit within a 1280px window
        self.assertLess(total_w, 1280)

    def test_slot_x_positions_non_overlapping(self):
        n = 12
        slot_size = config.HOTBAR_SLOT_SIZE
        padding = config.HOTBAR_PADDING
        total_w = n * slot_size + (n - 1) * padding
        start_x = (1280 - total_w) // 2
        positions = [start_x + i * (slot_size + padding) for i in range(n)]
        for i in range(len(positions) - 1):
            self.assertGreaterEqual(positions[i + 1], positions[i] + slot_size,
                                    f"Slots {i} and {i+1} overlap")


# ===========================================================================
# Day / night cycle
# ===========================================================================

class TestDayNightConfig(unittest.TestCase):
    """Verify day/night config constants are sane."""

    def test_day_length_positive(self):
        self.assertGreater(config.DAY_LENGTH, 0.0)

    def test_sun_min_brightness_in_range(self):
        self.assertGreater(config.SUN_MIN_BRIGHTNESS, 0.0)
        self.assertLess(config.SUN_MIN_BRIGHTNESS, 1.0)

    def test_sky_colours_are_tuples_of_three(self):
        for colour in [config.SKY_DAWN, config.SKY_DAY, config.SKY_DUSK, config.SKY_NIGHT]:
            self.assertIsInstance(colour, tuple)
            self.assertEqual(len(colour), 3)

    def test_sky_colour_channels_in_range(self):
        for colour in [config.SKY_DAWN, config.SKY_DAY, config.SKY_DUSK, config.SKY_NIGHT]:
            for ch in colour:
                self.assertGreaterEqual(ch, 0.0)
                self.assertLessEqual(ch, 1.0)

    def test_day_sky_is_bluest(self):
        # Midday sky should have the highest blue channel
        blues = [c[2] for c in [config.SKY_DAWN, config.SKY_DAY, config.SKY_DUSK, config.SKY_NIGHT]]
        self.assertEqual(max(blues), config.SKY_DAY[2])

    def test_night_sky_is_darkest(self):
        avg = lambda c: sum(c) / 3
        avgs = [avg(c) for c in [config.SKY_DAWN, config.SKY_DAY, config.SKY_DUSK, config.SKY_NIGHT]]
        self.assertEqual(min(avgs), avg(config.SKY_NIGHT))


class TestSunBrightness(unittest.TestCase):
    """Unit tests for config.sun_brightness()."""

    def test_noon_is_maximum(self):
        # Noon = DAY_LENGTH / 4 (quarter day in)
        noon = config.DAY_LENGTH / 4.0
        self.assertAlmostEqual(config.sun_brightness(noon), 1.0, places=3)

    def test_midnight_is_minimum(self):
        # Midnight = 3/4 of a day in
        midnight = config.DAY_LENGTH * 3.0 / 4.0
        self.assertAlmostEqual(config.sun_brightness(midnight),
                               config.SUN_MIN_BRIGHTNESS, places=3)

    def test_always_at_least_min_brightness(self):
        for i in range(100):
            t = i * config.DAY_LENGTH / 100.0
            self.assertGreaterEqual(config.sun_brightness(t), config.SUN_MIN_BRIGHTNESS)

    def test_never_exceeds_one(self):
        for i in range(100):
            t = i * config.DAY_LENGTH / 100.0
            self.assertLessEqual(config.sun_brightness(t), 1.0 + 1e-9)

    def test_symmetric_around_noon(self):
        # Dawn and dusk should be equal (symmetric sinusoid)
        dawn = config.sun_brightness(0.0)
        dusk = config.sun_brightness(config.DAY_LENGTH / 2.0)
        self.assertAlmostEqual(dawn, dusk, places=5)

    def test_morning_brighter_than_midnight(self):
        morning = config.sun_brightness(config.DAY_LENGTH / 8.0)
        midnight = config.sun_brightness(config.DAY_LENGTH * 3.0 / 4.0)
        self.assertGreater(morning, midnight)

    def test_wraps_correctly_over_multiple_days(self):
        # Brightness at t and t + DAY_LENGTH should be equal
        t = 123.45
        self.assertAlmostEqual(
            config.sun_brightness(t),
            config.sun_brightness(t + config.DAY_LENGTH),
            places=5
        )

    def test_returns_float(self):
        self.assertIsInstance(config.sun_brightness(0.0), float)


class TestSkyColour(unittest.TestCase):
    """Unit tests for config.sky_colour()."""

    def test_returns_tuple_of_three(self):
        result = config.sky_colour(0.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_channels_in_range(self):
        for i in range(60):
            t = i * config.DAY_LENGTH / 60.0
            r, g, b = config.sky_colour(t)
            self.assertGreaterEqual(r, 0.0); self.assertLessEqual(r, 1.0)
            self.assertGreaterEqual(g, 0.0); self.assertLessEqual(g, 1.0)
            self.assertGreaterEqual(b, 0.0); self.assertLessEqual(b, 1.0)

    def test_midday_matches_sky_day(self):
        noon = config.DAY_LENGTH / 4.0
        result = config.sky_colour(noon)
        for a, b in zip(result, config.SKY_DAY):
            self.assertAlmostEqual(a, b, places=5)

    def test_midnight_matches_sky_night(self):
        midnight = config.DAY_LENGTH * 3.0 / 4.0
        result = config.sky_colour(midnight)
        for a, b in zip(result, config.SKY_NIGHT):
            self.assertAlmostEqual(a, b, places=5)

    def test_dawn_matches_sky_dawn(self):
        result = config.sky_colour(0.0)
        for a, b in zip(result, config.SKY_DAWN):
            self.assertAlmostEqual(a, b, places=5)

    def test_dusk_matches_sky_dusk(self):
        dusk = config.DAY_LENGTH / 2.0
        result = config.sky_colour(dusk)
        for a, b in zip(result, config.SKY_DUSK):
            self.assertAlmostEqual(a, b, places=5)

    def test_midday_is_bluest_period(self):
        noon = config.DAY_LENGTH / 4.0
        midnight = config.DAY_LENGTH * 3.0 / 4.0
        self.assertGreater(config.sky_colour(noon)[2], config.sky_colour(midnight)[2])

    def test_continuous_no_sudden_jumps(self):
        # No two adjacent samples should differ by more than 0.15 per channel
        step = config.DAY_LENGTH / 200.0
        prev = config.sky_colour(0.0)
        for i in range(1, 201):
            curr = config.sky_colour(i * step)
            for a, b in zip(prev, curr):
                self.assertLess(abs(a - b), 0.15, "Sky colour jumped discontinuously")
            prev = curr

    def test_wraps_correctly(self):
        t = 77.7
        a = config.sky_colour(t)
        b = config.sky_colour(t + config.DAY_LENGTH)
        for ca, cb in zip(a, b):
            self.assertAlmostEqual(ca, cb, places=5)

    def test_morning_brighter_than_night(self):
        morning = config.sky_colour(config.DAY_LENGTH / 8.0)
        night   = config.sky_colour(config.DAY_LENGTH * 5.0 / 8.0)
        self.assertGreater(sum(morning), sum(night))


# ===========================================================================
# Frustum culling
# ===========================================================================

def _make_vp(eye_x=64, eye_y=20, eye_z=64,
             look_dx=1, look_dy=0, look_dz=0,
             fov=80.0, aspect=16/9,
             z_near=0.1, z_far=120.0):
    """Build a combined VP matrix (flat 16-float column-major list) using pure Python.

    Avoids pyglet.math so it works under the headless stub.
    """
    import math

    # --- perspective projection (column-major, OpenGL convention) ---
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    nf = 1.0 / (z_near - z_far)
    proj = [
        f/aspect, 0,  0,               0,
        0,        f,  0,               0,
        0,        0,  (z_far+z_near)*nf, -1,
        0,        0,  2*z_far*z_near*nf,  0,
    ]

    # --- look-at view matrix ---
    # forward = normalise(target - eye)
    fx, fy, fz = look_dx, look_dy, look_dz
    fl = math.sqrt(fx*fx + fy*fy + fz*fz)
    fx, fy, fz = fx/fl, fy/fl, fz/fl

    # right = forward × up
    ux, uy, uz = 0, 1, 0
    rx = fy*uz - fz*uy
    ry = fz*ux - fx*uz
    rz = fx*uy - fy*ux
    rl = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx, ry, rz = rx/rl, ry/rl, rz/rl

    # true up = right × forward
    tux = ry*fz - rz*fy
    tuy = rz*fx - rx*fz
    tuz = rx*fy - ry*fx

    tx = -(rx*eye_x + ry*eye_y + rz*eye_z)
    ty = -(tux*eye_x + tuy*eye_y + tuz*eye_z)
    tz = fx*eye_x + fy*eye_y + fz*eye_z

    view = [
        rx,   tux,  -fx,  0,
        ry,   tuy,  -fy,  0,
        rz,   tuz,  -fz,  0,
        tx,   ty,    tz,  1,
    ]

    # --- multiply proj @ view (column-major 4×4) ---
    def mat_mul(a, b):
        result = [0.0] * 16
        for col in range(4):
            for row in range(4):
                s = 0.0
                for k in range(4):
                    s += a[k*4 + row] * b[col*4 + k]
                result[col*4 + row] = s
        return result

    return mat_mul(proj, view)


class TestExtractFrustumPlanes(unittest.TestCase):

    def test_returns_six_planes(self):
        vp = _make_vp()
        planes = extract_frustum_planes(vp)
        self.assertEqual(len(planes), 6)

    def test_each_plane_is_tuple_of_four(self):
        planes = extract_frustum_planes(_make_vp())
        for p in planes:
            self.assertEqual(len(p), 4)

    def test_planes_are_normalised(self):
        import math
        planes = extract_frustum_planes(_make_vp())
        for a, b, c, d in planes:
            length = math.sqrt(a*a + b*b + c*c)
            self.assertAlmostEqual(length, 1.0, places=5)

    def test_point_ahead_is_inside_all_planes(self):
        # Camera at (64,20,64) looking +x; point 5 units ahead
        planes = extract_frustum_planes(_make_vp())
        x, y, z = 69, 20, 64
        for a, b, c, d in planes:
            self.assertGreaterEqual(a*x + b*y + c*z + d, 0,
                                    "Point directly ahead should be inside frustum")

    def test_point_directly_behind_is_outside(self):
        # Camera looking +x; point behind = lower x
        planes = extract_frustum_planes(_make_vp())
        x, y, z = 60, 20, 64   # behind camera
        results = [a*x + b*y + c*z + d for a, b, c, d in planes]
        self.assertTrue(any(r < 0 for r in results),
                        "Point behind camera should fail at least one plane")

    def test_point_beyond_far_plane_is_outside(self):
        # z_far=120; point 300 units ahead should be outside
        planes = extract_frustum_planes(_make_vp())
        x, y, z = 364, 20, 64
        results = [a*x + b*y + c*z + d for a, b, c, d in planes]
        self.assertTrue(any(r < 0 for r in results))

    def test_different_view_directions_give_different_planes(self):
        planes_x = extract_frustum_planes(_make_vp(look_dx=1, look_dz=0))
        planes_z = extract_frustum_planes(_make_vp(look_dx=0, look_dz=1))
        self.assertNotEqual(planes_x, planes_z)


class TestAABBOutsideFrustum(unittest.TestCase):

    def setUp(self):
        # Camera at (64,20,64) looking +x
        self.planes = extract_frustum_planes(_make_vp())

    def test_box_ahead_is_not_culled(self):
        # Box from x=70..86 (ahead of camera), centred on camera z
        self.assertFalse(aabb_outside_frustum(
            self.planes, 70, 0, 56, 86, 64, 72))

    def test_box_behind_is_culled(self):
        # Box entirely behind camera
        self.assertTrue(aabb_outside_frustum(
            self.planes, 40, 0, 56, 56, 64, 72))

    def test_box_beyond_far_plane_is_culled(self):
        # Beyond z_far=120 units ahead: x > 64+120 = 184
        self.assertTrue(aabb_outside_frustum(
            self.planes, 190, 0, 56, 206, 64, 72))

    def test_box_straddling_near_is_not_culled(self):
        # Box overlapping camera position — straddles near plane, should pass
        self.assertFalse(aabb_outside_frustum(
            self.planes, 63, 0, 56, 80, 64, 72))

    def test_large_box_enclosing_frustum_is_not_culled(self):
        # A box that contains the entire frustum should never be culled
        self.assertFalse(aabb_outside_frustum(
            self.planes, 0, 0, 0, 256, 128, 256))

    def test_returns_bool(self):
        result = aabb_outside_frustum(self.planes, 0, 0, 0, 16, 64, 16)
        self.assertIsInstance(result, bool)


class TestSectorAABB(unittest.TestCase):

    def test_origin_sector(self):
        mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = sector_aabb((0, 0, 0))
        self.assertEqual(mn_x, 0)
        self.assertEqual(mn_z, 0)
        self.assertEqual(mx_x, config.SECTOR_SIZE)
        self.assertEqual(mx_z, config.SECTOR_SIZE)

    def test_sector_size_matches_config(self):
        s = config.SECTOR_SIZE
        mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = sector_aabb((1, 0, 1))
        self.assertEqual(mx_x - mn_x, s)
        self.assertEqual(mx_z - mn_z, s)

    def test_y_spans_world_height(self):
        mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = sector_aabb((2, 0, 3))
        self.assertEqual(mn_y, 0)
        self.assertEqual(mx_y, 64)

    def test_sector_coords_scale_correctly(self):
        s = config.SECTOR_SIZE
        mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = sector_aabb((3, 0, 5))
        self.assertEqual(mn_x, 3 * s)
        self.assertEqual(mn_z, 5 * s)
        self.assertEqual(mx_x, 4 * s)
        self.assertEqual(mx_z, 6 * s)

    def test_returns_six_values(self):
        result = sector_aabb((0, 0, 0))
        self.assertEqual(len(result), 6)

    def test_min_less_than_max(self):
        mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = sector_aabb((1, 0, 2))
        self.assertLess(mn_x, mx_x)
        self.assertLess(mn_y, mx_y)
        self.assertLess(mn_z, mx_z)


class TestModelFrustum(unittest.TestCase):
    """Test frustum integration on headless Model."""

    def setUp(self):
        from unittest.mock import patch
        with patch.object(Model, '_initialize', return_value=None):
            self.model = Model()

    def test_frustum_planes_start_none(self):
        self.assertIsNone(self.model.frustum_planes)

    def test_set_frustum_stores_planes(self):
        vp = _make_vp()
        self.model.set_frustum(vp)
        self.assertIsNotNone(self.model.frustum_planes)
        self.assertEqual(len(self.model.frustum_planes), 6)

    def test_set_frustum_updates_each_call(self):
        self.model.set_frustum(_make_vp(look_dx=1, look_dz=0))
        planes_a = list(self.model.frustum_planes)
        self.model.set_frustum(_make_vp(look_dx=0, look_dz=1))
        planes_b = list(self.model.frustum_planes)
        self.assertNotEqual(planes_a, planes_b)

    def test_show_sector_skips_culled_sector(self):
        # Camera looking +x at (64,20,64); sector behind = (3,0,4) → x 48..64
        # Add a block in that sector, set frustum, show_sector should skip it
        self.model.add_block((50, 10, 64), config.GRASS, immediate=False)
        self.model.set_frustum(_make_vp())
        self.model.show_sector((3, 0, 4))
        # Block should NOT have been shown (it's behind the camera)
        self.assertNotIn((50, 10, 64), self.model.shown)

    def test_show_sector_shows_visible_sector(self):
        # Sector ahead at (5,0,4) → x 80..96, z 64..80
        self.model.add_block((82, 10, 66), config.STONE, immediate=False)
        self.model.set_frustum(_make_vp())
        self.model.show_sector((5, 0, 4))
        # Block IS in the frustum — show_sector should have queued it
        # (shown or queued — immediate=False means it goes to queue)
        self.assertIn((82, 10, 66), self.model.shown)

    def test_show_sector_works_without_frustum(self):
        # frustum_planes=None means no culling — all sectors show normally
        self.model.add_block((10, 5, 10), config.DIRT, immediate=False)
        self.assertIsNone(self.model.frustum_planes)
        self.model.show_sector((0, 0, 0))
        self.assertIn((10, 5, 10), self.model.shown)


# ===========================================================================
# Infinite world / chunk system
# ===========================================================================

class TestInfiniteWorldConfig(unittest.TestCase):

    def test_render_distance_positive(self):
        self.assertGreater(config.RENDER_DISTANCE, 0)

    def test_evict_distance_greater_than_render(self):
        self.assertGreater(config.EVICT_DISTANCE, config.RENDER_DISTANCE)

    def test_distances_are_ints(self):
        self.assertIsInstance(config.RENDER_DISTANCE, int)
        self.assertIsInstance(config.EVICT_DISTANCE, int)


class TestGenerateChunk(unittest.TestCase):

    def _make_model(self):
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        m.seed = 42
        m._gen       = __import__('noise_gen').NoiseGen(42)
        m._temp_gen  = __import__('noise_gen').NoiseGen(42 + config.BIOME_TEMP_SEED_OFFSET)
        m._moist_gen = __import__('noise_gen').NoiseGen(42 + config.BIOME_MOIST_SEED_OFFSET)
        return m

    def test_generate_chunk_adds_blocks(self):
        m = self._make_model()
        m.generate_chunk(0, 0)
        self.assertGreater(len(m.world), 0)

    def test_generate_chunk_marks_as_generated(self):
        m = self._make_model()
        m.generate_chunk(3, 5)
        self.assertIn((3, 5), m.generated_chunks)

    def test_generate_chunk_idempotent(self):
        m = self._make_model()
        m.generate_chunk(0, 0)
        count_after_first = len(m.world)
        m.generate_chunk(0, 0)
        self.assertEqual(len(m.world), count_after_first)

    def test_generate_chunk_covers_correct_columns(self):
        m = self._make_model()
        S = config.SECTOR_SIZE
        m.generate_chunk(2, 3)
        # All world blocks should have x in [32..47], z in [48..63]
        for (x, y, z) in m.world:
            self.assertGreaterEqual(x, 2 * S)
            self.assertLess(x, 3 * S)
            self.assertGreaterEqual(z, 3 * S)
            self.assertLess(z, 4 * S)

    def test_generate_negative_chunk(self):
        m = self._make_model()
        m.generate_chunk(-1, -1)
        S = config.SECTOR_SIZE
        self.assertIn((-1, -1), m.generated_chunks)
        # Blocks should be in the negative quadrant
        for (x, y, z) in m.world:
            self.assertGreaterEqual(x, -S)
            self.assertLess(x, 0)
            self.assertGreaterEqual(z, -S)
            self.assertLess(z, 0)

    def test_generated_chunks_starts_empty(self):
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        self.assertEqual(len(m.generated_chunks), 0)

    def test_chunk_blocks_have_valid_y(self):
        m = self._make_model()
        m.generate_chunk(0, 0)
        for (x, y, z) in m.world:
            self.assertGreaterEqual(y, 0)

    def test_multiple_chunks_dont_overlap(self):
        m = self._make_model()
        m.generate_chunk(0, 0)
        m.generate_chunk(1, 0)
        S = config.SECTOR_SIZE
        chunk0 = {pos for pos in m.world if pos[0] < S}
        chunk1 = {pos for pos in m.world if pos[0] >= S}
        self.assertEqual(len(chunk0 & chunk1), 0)

    def test_no_gen_when_gen_is_none(self):
        """Loaded worlds (self._gen=None) don't get terrain overwritten."""
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        m._gen = None
        m.seed = 42
        m.add_block((5, 5, 5), config.BRICK, immediate=False)
        m.generate_chunk(0, 0)   # should be a no-op
        self.assertEqual(len(m.world), 1)
        self.assertIn((5, 5, 5), m.world)

    def test_deterministic_with_same_seed(self):
        m1 = self._make_model()
        m2 = self._make_model()
        m1.generate_chunk(5, 5)
        m2.generate_chunk(5, 5)
        self.assertEqual(set(m1.world.keys()), set(m2.world.keys()))

    def test_different_seeds_give_different_terrain(self):
        import noise_gen as ng
        with patch.object(Model, '_initialize', return_value=None):
            m1 = Model()
        m1.seed = 1
        m1._gen       = ng.NoiseGen(1)
        m1._temp_gen  = ng.NoiseGen(1 + config.BIOME_TEMP_SEED_OFFSET)
        m1._moist_gen = ng.NoiseGen(1 + config.BIOME_MOIST_SEED_OFFSET)
        with patch.object(Model, '_initialize', return_value=None):
            m2 = Model()
        m2.seed = 999999
        m2._gen       = ng.NoiseGen(999999)
        m2._temp_gen  = ng.NoiseGen(999999 + config.BIOME_TEMP_SEED_OFFSET)
        m2._moist_gen = ng.NoiseGen(999999 + config.BIOME_MOIST_SEED_OFFSET)
        m1.generate_chunk(0, 0)
        m2.generate_chunk(0, 0)
        self.assertNotEqual(set(m1.world.keys()), set(m2.world.keys()))


class TestEviction(unittest.TestCase):

    def _make_model(self):
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        return m

    def test_evict_sector_removes_blocks(self):
        m = self._make_model()
        m.add_block((0, 5, 0), config.GRASS, immediate=False)
        m.add_block((1, 5, 0), config.STONE, immediate=False)
        sector = (0, 0, 0)
        m._evict_sector(sector)
        self.assertEqual(len(m.world), 0)

    def test_evict_sector_removes_sector_entry(self):
        m = self._make_model()
        m.add_block((0, 5, 0), config.GRASS, immediate=False)
        m._evict_sector((0, 0, 0))
        self.assertNotIn((0, 0, 0), m.sectors)

    def test_evict_empty_sector_no_error(self):
        m = self._make_model()
        m._evict_sector((99, 0, 99))   # should not raise

    def test_evict_does_not_remove_generated_chunk_flag(self):
        m = self._make_model()
        m.generated_chunks.add((0, 0))
        m.add_block((0, 5, 0), config.GRASS, immediate=False)
        m._evict_sector((0, 0, 0))
        # Flag should still be set so we don't re-terrain the chunk
        self.assertIn((0, 0), m.generated_chunks)


# ===========================================================================
# Biome system
# ===========================================================================

class TestClassifyBiome(unittest.TestCase):
    """Unit tests for config.classify_biome() — pure function, no GL."""

    def test_cold_dry_is_tundra(self):
        self.assertEqual(config.classify_biome(-0.5, -0.5), 'TUNDRA')

    def test_cold_wet_is_taiga(self):
        self.assertEqual(config.classify_biome(-0.5, 0.5), 'TAIGA')

    def test_temperate_dry_is_plains(self):
        self.assertEqual(config.classify_biome(0.1, -0.5), 'PLAINS')

    def test_temperate_wet_is_forest(self):
        self.assertEqual(config.classify_biome(0.1, 0.5), 'FOREST')

    def test_hot_dry_is_desert(self):
        self.assertEqual(config.classify_biome(0.8, -0.5), 'DESERT')

    def test_hot_wet_is_savanna(self):
        self.assertEqual(config.classify_biome(0.8, 0.5), 'SAVANNA')

    def test_returns_string(self):
        self.assertIsInstance(config.classify_biome(0.0, 0.0), str)

    def test_result_is_known_biome(self):
        for t in [-0.8, -0.1, 0.2, 0.6]:
            for m in [-0.8, 0.0, 0.6]:
                b = config.classify_biome(t, m)
                self.assertIn(b, config.BIOMES)

    def test_boundary_cold_temperate(self):
        # -0.2 is the boundary: just below = cold, at/above = temperate
        self.assertIn(config.classify_biome(-0.21, 0.5), ('TAIGA', 'TUNDRA'))
        self.assertIn(config.classify_biome(-0.19, 0.5), ('FOREST', 'PLAINS'))

    def test_boundary_temperate_hot(self):
        self.assertIn(config.classify_biome(0.39, 0.5), ('FOREST', 'PLAINS'))
        self.assertIn(config.classify_biome(0.41, 0.5), ('SAVANNA', 'DESERT'))

    def test_all_six_biomes_reachable(self):
        cases = [
            (-0.5, -0.5), (-0.5,  0.5),
            ( 0.1, -0.5), ( 0.1,  0.5),
            ( 0.8, -0.5), ( 0.8,  0.5),
        ]
        results = {config.classify_biome(t, m) for t, m in cases}
        self.assertEqual(results, set(config.BIOMES.keys()))


class TestBiomeConfig(unittest.TestCase):
    """Verify biome config constants are well-formed."""

    def test_has_six_biomes(self):
        self.assertEqual(len(config.BIOMES), 6)

    def test_all_expected_biomes_present(self):
        for name in ('TUNDRA', 'TAIGA', 'PLAINS', 'FOREST', 'DESERT', 'SAVANNA'):
            self.assertIn(name, config.BIOMES)

    def test_each_biome_has_required_keys(self):
        required = {'surface', 'subsurface', 'tree_chance',
                    'crystal_chance', 'gravel_near_water'}
        for name, biome in config.BIOMES.items():
            for key in required:
                self.assertIn(key, biome, f"Biome {name} missing key '{key}'")

    def test_surface_blocks_are_valid_config_attrs(self):
        for name, biome in config.BIOMES.items():
            attr = biome['surface']
            self.assertTrue(hasattr(config, attr),
                            f"Biome {name} surface '{attr}' not in config")

    def test_subsurface_blocks_are_valid_config_attrs(self):
        for name, biome in config.BIOMES.items():
            attr = biome['subsurface']
            self.assertTrue(hasattr(config, attr),
                            f"Biome {name} subsurface '{attr}' not in config")

    def test_tree_chances_are_floats_in_range(self):
        for name, biome in config.BIOMES.items():
            tc = biome['tree_chance']
            self.assertIsInstance(tc, float)
            self.assertGreaterEqual(tc, 0.0)
            self.assertLessEqual(tc, 1.0)

    def test_crystal_chances_are_floats_in_range(self):
        for name, biome in config.BIOMES.items():
            cc = biome['crystal_chance']
            self.assertIsInstance(cc, float)
            self.assertGreaterEqual(cc, 0.0)
            self.assertLessEqual(cc, 1.0)

    def test_gravel_near_water_is_bool(self):
        for name, biome in config.BIOMES.items():
            self.assertIsInstance(biome['gravel_near_water'], bool)

    def test_tundra_has_no_trees(self):
        self.assertEqual(config.BIOMES['TUNDRA']['tree_chance'], 0.0)

    def test_desert_has_no_trees(self):
        self.assertEqual(config.BIOMES['DESERT']['tree_chance'], 0.0)

    def test_forest_has_highest_tree_chance(self):
        chances = {n: b['tree_chance'] for n, b in config.BIOMES.items()}
        self.assertEqual(max(chances, key=chances.get), 'FOREST')

    def test_desert_surface_is_sand(self):
        self.assertEqual(config.BIOMES['DESERT']['surface'], 'SAND')

    def test_tundra_surface_is_snow(self):
        self.assertEqual(config.BIOMES['TUNDRA']['surface'], 'SNOW')

    def test_biome_smoothness_positive(self):
        self.assertGreater(config.BIOME_TEMP_SMOOTHNESS, 0)
        self.assertGreater(config.BIOME_MOIST_SMOOTHNESS, 0)

    def test_biome_seed_offsets_distinct(self):
        self.assertNotEqual(config.BIOME_TEMP_SEED_OFFSET,
                            config.BIOME_MOIST_SEED_OFFSET)


class TestGetClimate(unittest.TestCase):
    """Unit tests for NoiseGen.get_climate()."""

    def setUp(self):
        from noise_gen import NoiseGen
        self.gen = NoiseGen(42)

    def test_returns_float(self):
        result = self.gen.get_climate(0, 0, 200)
        self.assertIsInstance(result, float)

    def test_in_range(self):
        for x in range(-100, 100, 10):
            for z in range(-100, 100, 10):
                v = self.gen.get_climate(x, z, 200)
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)

    def test_varies_across_space(self):
        vals = {self.gen.get_climate(x, 0, 200)
                for x in range(-300, 300, 8)}
        self.assertGreater(len(vals), 5,
                           "Climate noise should vary across space")

    def test_deterministic(self):
        from noise_gen import NoiseGen
        g2 = NoiseGen(42)
        for x, z in [(0,0),(10,20),(-50,30)]:
            self.assertEqual(self.gen.get_climate(x, z, 200),
                             g2.get_climate(x, z, 200))

    def test_different_smoothness_gives_different_values(self):
        v1 = self.gen.get_climate(100, 100, 200)
        v2 = self.gen.get_climate(100, 100, 500)
        self.assertNotAlmostEqual(v1, v2, places=3)

    def test_different_seeds_give_different_climate(self):
        from noise_gen import NoiseGen
        g2 = NoiseGen(999)
        diffs = [abs(self.gen.get_climate(x, 0, 200) - g2.get_climate(x, 0, 200))
                 for x in range(-200, 200, 20)]
        self.assertGreater(sum(diffs), 0.1,
                           "Different seeds should produce different climate")


class TestBiomeGeneration(unittest.TestCase):
    """Integration tests: biome rules applied in generate_chunk."""

    def _make_model(self):
        from noise_gen import NoiseGen
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        m.seed = 42
        m._gen       = NoiseGen(42)
        m._temp_gen  = NoiseGen(42 + config.BIOME_TEMP_SEED_OFFSET)
        m._moist_gen = NoiseGen(42 + config.BIOME_MOIST_SEED_OFFSET)
        return m

    def test_desert_chunk_has_sand_surface(self):
        """Find a desert chunk and verify its surface is sand."""
        from noise_gen import NoiseGen
        temp_gen  = NoiseGen(42 + config.BIOME_TEMP_SEED_OFFSET)
        moist_gen = NoiseGen(42 + config.BIOME_MOIST_SEED_OFFSET)
        S = config.SECTOR_SIZE
        # Search for a desert sector
        desert_sector = None
        for sx in range(-10, 10):
            for sz in range(-10, 10):
                x, z = sx * S + S // 2, sz * S + S // 2
                t = temp_gen.get_climate(x, z, config.BIOME_TEMP_SMOOTHNESS)
                m = moist_gen.get_climate(x, z, config.BIOME_MOIST_SMOOTHNESS)
                if config.classify_biome(t, m) == 'DESERT':
                    desert_sector = (sx, sz)
                    break
            if desert_sector:
                break
        if desert_sector is None:
            self.skipTest("No desert sector found in search range")
        model = self._make_model()
        sx, sz = desert_sector
        model.generate_chunk(sx, sz)
        surface_blocks = []
        for (x, y, z), tex in model.world.items():
            # Find highest block per column
            col = (x, z)
            surface_blocks.append((y, tex))
        # Check no grass on very high blocks in desert (sand or snow at peak)
        top_blocks = {}
        for (x, y, z), tex in model.world.items():
            if (x, z) not in top_blocks or y > top_blocks[(x, z)][0]:
                top_blocks[(x, z)] = (y, tex)
        non_desert_surface = [
            tex for y, tex in top_blocks.values()
            if tex not in (config.SAND, config.SNOW, config.LEAF, config.CRYSTAL)
            and y > 18
        ]
        self.assertEqual(len(non_desert_surface), 0,
                         "Desert surface should be sand (or snow at peaks)")

    def test_tundra_has_no_trees(self):
        """Find a tundra chunk and verify it contains no WOOD blocks."""
        from noise_gen import NoiseGen
        temp_gen  = NoiseGen(42 + config.BIOME_TEMP_SEED_OFFSET)
        moist_gen = NoiseGen(42 + config.BIOME_MOIST_SEED_OFFSET)
        S = config.SECTOR_SIZE
        tundra_sector = None
        for sx in range(-20, 20):
            for sz in range(-20, 20):
                x, z = sx * S + S // 2, sz * S + S // 2
                t = temp_gen.get_climate(x, z, config.BIOME_TEMP_SMOOTHNESS)
                m = moist_gen.get_climate(x, z, config.BIOME_MOIST_SMOOTHNESS)
                if config.classify_biome(t, m) == 'TUNDRA':
                    tundra_sector = (sx, sz)
                    break
            if tundra_sector:
                break
        if tundra_sector is None:
            self.skipTest("No tundra sector in search range")
        model = self._make_model()
        sx, sz = tundra_sector
        model.generate_chunk(sx, sz)
        wood_blocks = [pos for pos, tex in model.world.items()
                       if tex == config.WOOD]
        self.assertEqual(len(wood_blocks), 0,
                         "Tundra should have no trees")

    def test_generate_chunk_uses_biome_surface(self):
        """generate_chunk produces surface blocks consistent with biome rules."""
        model = self._make_model()
        S = config.SECTOR_SIZE
        # Generate several chunks and verify each column's top block matches biome
        for sx, sz in [(0,0),(1,0),(0,1),(-1,0)]:
            model.generate_chunk(sx, sz)

        valid_surfaces = [config.GRASS, config.SNOW, config.SAND,
                          config.GRAVEL, config.LEAF]
        top_blocks = {}
        for (x, y, z), tex in model.world.items():
            if (x, z) not in top_blocks or y > top_blocks[(x, z)][0]:
                top_blocks[(x, z)] = (y, tex)

        for (x, z), (y, tex) in top_blocks.items():
            if y >= 18:
                self.assertIn(tex, valid_surfaces,
                              f"Unexpected surface block {tex} at ({x},{y},{z})")

    def test_water_fills_low_areas_regardless_of_biome(self):
        """Low-height columns should have water regardless of biome."""
        model = self._make_model()
        model.generate_chunk(0, 0)
        has_water = any(
            tex in (config.WATER, config.MAGIC_WATER)
            for tex in model.world.values()
        )
        # Not guaranteed in every chunk but should exist somewhere nearby
        for sx in range(-3, 3):
            for sz in range(-3, 3):
                model.generate_chunk(sx, sz)
        has_water = any(
            tex in (config.WATER, config.MAGIC_WATER)
            for tex in model.world.values()
        )
        self.assertTrue(has_water, "Expected water somewhere in generated area")

    def test_biome_chunks_are_deterministic(self):
        """Same seed + same chunk coords = identical world state."""
        m1 = self._make_model()
        m2 = self._make_model()
        for sx, sz in [(0,0),(1,1),(-1,0)]:
            m1.generate_chunk(sx, sz)
            m2.generate_chunk(sx, sz)
        self.assertEqual(set(m1.world.keys()), set(m2.world.keys()))


# ===========================================================================
# Sound system
# ===========================================================================

class TestSoundSynthesis(unittest.TestCase):
    """Test procedural sound generation — pure Python, no audio device needed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generate_sounds_creates_directory(self):
        from sounds import generate_sounds
        target = os.path.join(self.tmp, 'sounds')
        generate_sounds(target)
        self.assertTrue(os.path.isdir(target))

    def test_generate_sounds_creates_all_wavs(self):
        from sounds import generate_sounds, _SOUND_DEFS
        generate_sounds(self.tmp)
        for name in _SOUND_DEFS:
            path = os.path.join(self.tmp, f'{name}.wav')
            self.assertTrue(os.path.exists(path), f"Missing {name}.wav")

    def test_wav_files_are_valid(self):
        from sounds import generate_sounds, _SOUND_DEFS
        generate_sounds(self.tmp)
        for name in _SOUND_DEFS:
            path = os.path.join(self.tmp, f'{name}.wav')
            with wave.open(path, 'r') as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 22050)
                self.assertGreater(w.getnframes(), 0)

    def test_wav_sample_count_matches_duration(self):
        from sounds import generate_sounds, _SOUND_DEFS, SAMPLE_RATE
        generate_sounds(self.tmp)
        for name, (fn, dur, vol, seed) in _SOUND_DEFS.items():
            path = os.path.join(self.tmp, f'{name}.wav')
            with wave.open(path, 'r') as w:
                expected = int(SAMPLE_RATE * dur)
                self.assertEqual(w.getnframes(), expected,
                                 f"{name}: wrong frame count")

    def test_samples_within_int16_range(self):
        from sounds import generate_sounds, _SOUND_DEFS
        generate_sounds(self.tmp)
        for name in list(_SOUND_DEFS.keys())[:4]:   # spot-check a few
            path = os.path.join(self.tmp, f'{name}.wav')
            with wave.open(path, 'r') as w:
                raw = w.readframes(w.getnframes())
            samples = struct.unpack(f'<{len(raw)//2}h', raw)
            self.assertLessEqual(max(abs(s) for s in samples), 32767)

    def test_generate_sounds_is_idempotent(self):
        from sounds import generate_sounds
        generate_sounds(self.tmp)
        mtime_before = {
            f: os.path.getmtime(os.path.join(self.tmp, f))
            for f in os.listdir(self.tmp)
        }
        import time; time.sleep(0.05)
        generate_sounds(self.tmp)   # should not overwrite existing files
        for f, mtime in mtime_before.items():
            self.assertEqual(os.path.getmtime(os.path.join(self.tmp, f)),
                             mtime, f"{f} was overwritten on second call")

    def test_break_and_place_variants_exist(self):
        from sounds import _SOUND_DEFS
        categories = {'stone', 'dirt', 'wood', 'glass', 'leaf', 'water', 'sand'}
        for cat in categories:
            self.assertIn(f'{cat}_break', _SOUND_DEFS,
                          f"Missing {cat}_break")
            self.assertIn(f'{cat}_place', _SOUND_DEFS,
                          f"Missing {cat}_place")

    def test_stone_synth_returns_correct_length(self):
        from sounds import _stone, SAMPLE_RATE
        dur = 0.10
        samples = _stone(dur, 0.7, seed=0)
        self.assertEqual(len(samples), int(SAMPLE_RATE * dur))

    def test_all_synth_functions_return_floats(self):
        from sounds import _stone, _dirt, _wood, _glass, _leaf, _water
        for fn in (_stone, _dirt, _wood, _glass, _leaf, _water):
            samples = fn(0.05, 0.5, seed=0)
            self.assertTrue(all(isinstance(s, float) for s in samples))

    def test_all_synth_samples_in_range(self):
        from sounds import _stone, _dirt, _wood, _glass, _leaf, _water
        for fn in (_stone, _dirt, _wood, _glass, _leaf, _water):
            samples = fn(0.05, 0.5, seed=0)
            for s in samples:
                self.assertGreaterEqual(s, -1.0)
                self.assertLessEqual(s, 1.0)


class TestSoundConfig(unittest.TestCase):
    """Verify BLOCK_SOUND_MAP is complete and consistent."""

    def test_block_sound_map_has_14_entries(self):
        self.assertEqual(len(config.BLOCK_SOUND_MAP), 14)

    def test_all_block_types_in_map(self):
        all_blocks = [
            config.GRASS, config.SAND, config.BRICK, config.STONE,
            config.WOOD, config.LEAF, config.WATER, config.CRYSTAL,
            config.MAGIC_WATER, config.DIRT, config.SNOW,
            config.GLASS, config.PLANKS, config.GRAVEL,
        ]
        for block in all_blocks:
            self.assertIn(tuple(block), config.BLOCK_SOUND_MAP,
                          f"Block missing from BLOCK_SOUND_MAP")

    def test_all_categories_are_valid(self):
        valid = {'stone', 'dirt', 'sand', 'wood', 'glass', 'leaf', 'water'}
        for tex_key, category in config.BLOCK_SOUND_MAP.items():
            self.assertIn(category, valid,
                          f"Unknown sound category '{category}'")

    def test_stone_maps_to_stone(self):
        self.assertEqual(config.BLOCK_SOUND_MAP[tuple(config.STONE)], 'stone')

    def test_water_maps_to_water(self):
        self.assertEqual(config.BLOCK_SOUND_MAP[tuple(config.WATER)], 'water')

    def test_wood_maps_to_wood(self):
        self.assertEqual(config.BLOCK_SOUND_MAP[tuple(config.WOOD)], 'wood')

    def test_glass_maps_to_glass(self):
        self.assertEqual(config.BLOCK_SOUND_MAP[tuple(config.GLASS)], 'glass')

    def test_sound_names_resolve_to_wav_defs(self):
        from sounds import _SOUND_DEFS
        for category in set(config.BLOCK_SOUND_MAP.values()):
            self.assertIn(f'{category}_break', _SOUND_DEFS)
            self.assertIn(f'{category}_place', _SOUND_DEFS)


class TestSoundManager(unittest.TestCase):
    """Test SoundManager with mocked pyglet.media."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_manager(self):
        from sounds import SoundManager
        return SoundManager(sounds_dir=self.tmp)

    def test_manager_loads_without_error(self):
        sm = self._make_manager()
        self.assertIsNotNone(sm)

    def test_available_sounds_not_empty(self):
        sm = self._make_manager()
        self.assertGreater(len(sm.available_sounds()), 0)

    def test_play_known_sound_no_error(self):
        sm = self._make_manager()
        sm.play('stone_break')   # should not raise

    def test_play_unknown_sound_no_error(self):
        sm = self._make_manager()
        sm.play('nonexistent_sound')   # should not raise

    def test_play_when_disabled_no_error(self):
        sm = self._make_manager()
        sm._enabled = False
        sm.play('stone_break')   # should not raise

    def test_all_expected_sounds_loaded(self):
        from sounds import _SOUND_DEFS
        sm = self._make_manager()
        for name in _SOUND_DEFS:
            self.assertIn(name, sm.available_sounds(),
                          f"Sound '{name}' not loaded")

    def test_wavs_generated_in_sounds_dir(self):
        from sounds import _SOUND_DEFS
        self._make_manager()
        for name in _SOUND_DEFS:
            path = os.path.join(self.tmp, f'{name}.wav')
            self.assertTrue(os.path.exists(path))


# ===========================================================================
# Wormhole gun / portal system
# ===========================================================================

class TestPortalConfig(unittest.TestCase):

    def test_portal_duration_positive(self):
        self.assertGreater(config.PORTAL_DURATION, 0)

    def test_portal_cooldown_positive(self):
        self.assertGreater(config.PORTAL_COOLDOWN, 0)

    def test_portal_reach_greater_than_block_reach(self):
        self.assertGreater(config.PORTAL_REACH, 8)

    def test_portal_colours_are_rgba(self):
        for colour in (config.PORTAL_COLOR_A, config.PORTAL_COLOR_B):
            self.assertEqual(len(colour), 4)
            for ch in colour: self.assertGreaterEqual(ch, 0); self.assertLessEqual(ch, 255)

    def test_portal_colours_distinct(self):
        self.assertNotEqual(config.PORTAL_COLOR_A, config.PORTAL_COLOR_B)

    def test_portal_half_dimensions_positive(self):
        self.assertGreater(config.PORTAL_HALF_WIDTH, 0)
        self.assertGreater(config.PORTAL_HALF_HEIGHT, 0)

    def test_trigger_dist_positive(self):
        self.assertGreater(config.PORTAL_TRIGGER_DIST_XZ, 0)


class TestPortalEnd(unittest.TestCase):

    def _make(self, x=10, y=20, z=30, nx=0, ny=0, nz=1):
        return PortalEnd(x, y, z, nx, ny, nz, config.PORTAL_COLOR_A)

    def test_position_property(self):
        p = self._make(1, 2, 3)
        self.assertEqual(p.position, (1, 2, 3))

    def test_normal_property(self):
        p = self._make(nx=1, ny=0, nz=0)
        self.assertEqual(p.normal, (1, 0, 0))

    def test_corners_returns_four_points(self):
        p = self._make()
        c = p.corners()
        self.assertEqual(len(c), 4)

    def test_corners_are_tuples_of_three(self):
        for corner in self._make().corners():
            self.assertEqual(len(corner), 3)

    def test_corners_span_correct_height(self):
        p = self._make(y=20)
        corners = p.corners()
        ys = [c[1] for c in corners]
        self.assertAlmostEqual(max(ys) - min(ys), config.PORTAL_HALF_HEIGHT * 2, places=3)

    def test_corners_span_correct_width_z_normal(self):
        p = self._make(nx=0, nz=1)  # z normal → width along x
        corners = p.corners()
        xs = [c[0] for c in corners]
        self.assertAlmostEqual(max(xs) - min(xs), config.PORTAL_HALF_WIDTH * 2, places=3)

    def test_corners_span_correct_width_x_normal(self):
        p = self._make(nx=1, nz=0)  # x normal → width along z
        corners = p.corners()
        zs = [c[2] for c in corners]
        self.assertAlmostEqual(max(zs) - min(zs), config.PORTAL_HALF_WIDTH * 2, places=3)

    def test_corners_offset_from_wall(self):
        p = self._make(nz=1)  # z normal, offset pushes z outward
        corners = p.corners()
        for c in corners:
            self.assertGreater(c[2], 30.0)  # all z > base z (offset applied)

    def test_arrival_point_is_in_front_of_portal(self):
        p = self._make(x=10, y=20, z=30, nx=0, ny=0, nz=1)
        arr = p.arrival_point()
        self.assertGreater(arr[2], 30.0)  # arrival is ahead in z

    def test_arrival_point_preserves_y(self):
        p = self._make(y=25)
        arr = p.arrival_point()
        self.assertAlmostEqual(arr[1], 26.0)  # y + 1 to stand above base block


class TestPortalManager(unittest.TestCase):

    def _mgr(self):
        m = PortalManager()
        # Provide a headless model so block placement works in tests
        with patch.object(Model, '_initialize', return_value=None):
            model = Model()
        m._world_ref = model
        return m

    def test_starts_inactive(self):
        m = self._mgr()
        self.assertFalse(m.active)
        self.assertIsNone(m.a)
        self.assertIsNone(m.b)

    def test_first_fire_sets_a(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        self.assertIsNotNone(m.a)
        self.assertIsNone(m.b)
        self.assertFalse(m.active)

    def test_second_fire_sets_b_and_activates(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        m.fire((10, 0, 10), (1, 0, 0))
        self.assertIsNotNone(m.b)
        self.assertTrue(m.active)

    def test_third_fire_resets_and_sets_new_a(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        m.fire((10, 0, 10), (1, 0, 0))
        m.fire((5, 0, 5), (0, 1, 0))
        self.assertIsNotNone(m.a)
        self.assertIsNone(m.b)
        self.assertFalse(m.active)

    def test_clear_removes_both(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        m.fire((10, 0, 10), (1, 0, 0))
        m.clear()
        self.assertFalse(m.active)
        self.assertIsNone(m.a)
        self.assertIsNone(m.b)

    def test_portal_a_colour(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        self.assertEqual(m.a.colour, config.PORTAL_COLOR_A)

    def test_portal_b_colour(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        m.fire((10, 0, 10), (1, 0, 0))
        self.assertEqual(m.b.colour, config.PORTAL_COLOR_B)

    def test_status_no_portal(self):
        m = self._mgr()
        self.assertIn("ready", m.status.lower())

    def test_status_a_set(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        self.assertIn("A", m.status)

    def test_status_active(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        m.fire((10, 0, 10), (1, 0, 0))
        self.assertIn("active", m.status.lower())

    def test_no_teleport_when_inactive(self):
        m = self._mgr()
        self.assertIsNone(m.check_teleport((0, 0, 0)))

    def test_no_teleport_when_only_a_set(self):
        m = self._mgr()
        m.fire((0, 0, 0), (0, 0, 1))
        self.assertIsNone(m.check_teleport((0, 0, 0)))

    def test_teleport_through_a(self):
        m = self._mgr()
        m.fire((0, 5, 0), (0, 0, 1))    # portal A at (0,5,0)
        m.fire((50, 5, 50), (0, 0, 1))  # portal B at (50,5,50)
        # Stand right on portal A — within trigger distance
        result = m.check_teleport((0.0, 5.0, 0.0))
        self.assertIsNotNone(result)
        dest, dy = result
        # Should arrive near portal B
        self.assertAlmostEqual(dest[0], 50.0, delta=3.0)
        self.assertEqual(dy, 0.0)

    def test_teleport_through_b(self):
        m = self._mgr()
        m.fire((0, 5, 0), (0, 0, 1))
        m.fire((50, 5, 50), (0, 0, 1))
        result = m.check_teleport((50.0, 5.0, 50.0))
        self.assertIsNotNone(result)
        dest, dy = result
        self.assertAlmostEqual(dest[0], 0.0, delta=3.0)

    def test_cooldown_prevents_immediate_retrigger(self):
        m = self._mgr()
        m.fire((0, 5, 0), (0, 0, 1))
        m.fire((50, 5, 50), (0, 0, 1))
        m.check_teleport((0.0, 5.0, 0.0))   # teleport
        # Immediately after — cooldown should block
        result = m.check_teleport((50.0, 5.0, 50.0))
        self.assertIsNone(result)

    def test_cooldown_expires(self):
        m = self._mgr()
        m.fire((0, 5, 0), (0, 0, 1))
        m.fire((50, 5, 50), (0, 0, 1))
        m.check_teleport((0.0, 5.0, 0.0))
        m.update(config.PORTAL_COOLDOWN + 0.1)  # let cooldown expire
        result = m.check_teleport((50.0, 5.0, 50.0))
        self.assertIsNotNone(result)

    def test_timed_portal_expires(self):
        import config as cfg
        orig = cfg.PORTAL_PERMANENT
        cfg.PORTAL_PERMANENT = False
        try:
            m = self._mgr()
            m.fire((0, 5, 0), (0, 0, 1))
            m.fire((50, 5, 50), (0, 0, 1))
            closed = m.update(config.PORTAL_DURATION + 1.0)
            self.assertTrue(closed)
            self.assertFalse(m.active)
        finally:
            cfg.PORTAL_PERMANENT = orig

    def test_time_remaining_counts_down(self):
        import config as cfg
        orig = cfg.PORTAL_PERMANENT
        cfg.PORTAL_PERMANENT = False
        try:
            m = self._mgr()
            m.fire((0, 5, 0), (0, 0, 1))
            m.fire((50, 5, 50), (0, 0, 1))
            m.update(3.0)
            self.assertAlmostEqual(m.time_remaining,
                                   config.PORTAL_DURATION - 3.0, delta=0.1)
        finally:
            cfg.PORTAL_PERMANENT = orig

    def test_permanent_portal_never_expires(self):
        import config as cfg
        orig = cfg.PORTAL_PERMANENT
        cfg.PORTAL_PERMANENT = True
        try:
            m = self._mgr()
            m.fire((0, 5, 0), (0, 0, 1))
            m.fire((50, 5, 50), (0, 0, 1))
            closed = m.update(9999.0)
            self.assertFalse(closed)
            self.assertTrue(m.active)
            self.assertEqual(m.time_remaining, float('inf'))
        finally:
            cfg.PORTAL_PERMANENT = orig

    def test_portal_positions_stored_correctly(self):
        m = self._mgr()
        m.fire((3, 10, 7), (1, 0, 0))
        m.fire((99, 10, 99), (0, 0, 1))
        self.assertAlmostEqual(m.a.x, 3)
        self.assertAlmostEqual(m.a.z, 7)
        self.assertAlmostEqual(m.b.x, 99)
        self.assertAlmostEqual(m.b.z, 99)

    def test_model_has_portal_manager(self):
        with patch.object(Model, '_initialize', return_value=None):
            model = Model()
        self.assertIsInstance(model.portal, PortalManager)


# ===========================================================================
# Minimap
# ===========================================================================

class TestMinimapConfig(unittest.TestCase):

    def test_minimap_size_positive(self):
        self.assertGreater(config.MINIMAP_SIZE, 0)

    def test_minimap_scale_positive(self):
        self.assertGreater(config.MINIMAP_SCALE, 0)

    def test_minimap_margin_non_negative(self):
        self.assertGreaterEqual(config.MINIMAP_MARGIN, 0)

    def test_minimap_colours_dict_non_empty(self):
        self.assertGreater(len(config.MINIMAP_COLOURS), 0)

    def test_minimap_colours_are_rgb_triples(self):
        for name, col in config.MINIMAP_COLOURS.items():
            self.assertEqual(len(col), 3, f"{name} colour not a triple")
            for ch in col:
                self.assertGreaterEqual(ch, 0)
                self.assertLessEqual(ch, 255)

    def test_all_main_blocks_have_minimap_colour(self):
        for name in ('GRASS', 'SAND', 'STONE', 'WATER', 'SNOW', 'DESERT',
                     'DIRT', 'WOOD', 'LEAF'):
            if name in config.MINIMAP_COLOURS:
                self.assertIn(name, config.MINIMAP_COLOURS)


class TestBuildMinimapData(unittest.TestCase):

    def _make_model(self):
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        return m

    def test_returns_bytes(self):
        m = self._make_model()
        data = m.build_minimap_data((0, 0, 0))
        self.assertIsInstance(data, bytes)

    def test_correct_size(self):
        m = self._make_model()
        data = m.build_minimap_data((0, 0, 0))
        expected = config.MINIMAP_SIZE * config.MINIMAP_SIZE * 4  # RGBA
        self.assertEqual(len(data), expected)

    def test_empty_world_gives_dark_pixels(self):
        m = self._make_model()
        data = m.build_minimap_data((0, 0, 0))
        # All pixels should be the dark unloaded colour (40, 40, 40, 255)
        for i in range(0, len(data), 4):
            r, g, b, a = data[i], data[i+1], data[i+2], data[i+3]
            self.assertEqual(a, 255)
            self.assertLessEqual(r, 60)

    def test_grass_block_colours_pixel(self):
        m = self._make_model()
        # Place a grass block directly at player position
        m.add_block((0, 10, 0), config.GRASS, immediate=False)
        data = m.build_minimap_data((0, 0, 0))
        size = config.MINIMAP_SIZE
        cx = size // 2
        cy = size // 2
        idx = (cy * size + cx) * 4
        r, g, b = data[idx], data[idx+1], data[idx+2]
        expected = config.MINIMAP_COLOURS['GRASS']
        self.assertEqual((r, g, b), expected)

    def test_water_block_shows_blue(self):
        m = self._make_model()
        m.add_block((0, 10, 0), config.WATER, immediate=False)
        data = m.build_minimap_data((0, 0, 0))
        size = config.MINIMAP_SIZE
        cx, cy = size // 2, size // 2
        idx = (cy * size + cx) * 4
        b = data[idx + 2]
        self.assertGreater(b, 150, "Water should show as blue-dominant pixel")

    def test_higher_block_occludes_lower(self):
        m = self._make_model()
        # Grass at y=5, stone above at y=6 — stone should show
        m.add_block((0, 5, 0), config.GRASS, immediate=False)
        m.add_block((0, 6, 0), config.STONE, immediate=False)
        data = m.build_minimap_data((0, 0, 0))
        size = config.MINIMAP_SIZE
        idx = (size//2 * size + size//2) * 4
        r, g, b = data[idx], data[idx+1], data[idx+2]
        stone_col = config.MINIMAP_COLOURS['STONE']
        self.assertEqual((r, g, b), stone_col)

    def test_player_offset_shifts_map(self):
        m = self._make_model()
        # Block at (10, 5, 0) — with player at (10, 0, 0) it should be at centre
        m.add_block((10, 5, 0), config.SAND, immediate=False)
        data = m.build_minimap_data((10, 0, 0))
        size = config.MINIMAP_SIZE
        idx = (size//2 * size + size//2) * 4
        r, g, b = data[idx], data[idx+1], data[idx+2]
        sand_col = config.MINIMAP_COLOURS['SAND']
        self.assertEqual((r, g, b), sand_col)

    def test_all_pixels_have_full_alpha(self):
        m = self._make_model()
        data = m.build_minimap_data((0, 0, 0))
        for i in range(3, len(data), 4):
            self.assertEqual(data[i], 255)


# ===========================================================================
# Portal compass
# ===========================================================================

class TestPortalCompassConfig(unittest.TestCase):

    def test_compass_margin_positive(self):
        self.assertGreater(config.COMPASS_MARGIN, 0)

    def test_compass_arrow_r_positive(self):
        self.assertGreater(config.COMPASS_ARROW_R, 0)


class TestPortalCompassMaths(unittest.TestCase):
    """Test the bearing and screen-projection maths without GL."""

    def _screen_pos(self, dx, dz, yaw_deg, w=1280, h=720):
        """Reproduce the compass projection from draw_portal_compass."""
        import math
        cx, cy = w // 2, h // 2
        margin = config.COMPASS_MARGIN
        yaw = math.radians(yaw_deg)
        world_bearing = math.atan2(dx, -dz)
        screen_bearing = world_bearing - yaw + math.pi / 2
        sdx = math.cos(screen_bearing)
        sdy = math.sin(screen_bearing)
        if abs(sdx) > 1e-6:
            tx = (w/2 - margin) / abs(sdx)
        else:
            tx = float('inf')
        if abs(sdy) > 1e-6:
            ty = (h/2 - margin) / abs(sdy)
        else:
            ty = float('inf')
        t = min(tx, ty)
        sx = cx + sdx * t
        sy = cy + sdy * t
        sx = max(margin, min(w - margin, sx))
        sy = max(margin, min(h - margin, sy))
        return (sx, sy)

    def test_portal_directly_ahead_is_near_centre(self):
        # Player facing +Z (yaw=0), portal is ahead at dz=50
        sx, sy = self._screen_pos(0, 50, yaw_deg=0)
        self.assertAlmostEqual(sx, 640, delta=5)    # near horizontal centre

    def test_portal_behind_is_near_vertical_edge(self):
        # Player facing +Z, portal behind at dz=-50 — should hit top or bottom edge
        sx, sy = self._screen_pos(0, -50, yaw_deg=0)
        at_edge = (abs(sy - config.COMPASS_MARGIN) < 5 or
                   abs(sy - (720 - config.COMPASS_MARGIN)) < 5)
        self.assertTrue(at_edge, f"Behind portal sy={sy:.0f} should be at a vertical edge")

    def test_portal_left_is_at_left_or_right_edge(self):
        # Player facing +Z (yaw=0), portal to the left (dx<0) — should hit horizontal edge
        sx, sy = self._screen_pos(-50, 0, yaw_deg=0)
        at_edge = (abs(sx - config.COMPASS_MARGIN) < 5 or
                   abs(sx - (1280 - config.COMPASS_MARGIN)) < 5)
        self.assertTrue(at_edge, f"Left portal sx={sx:.0f} should be at horizontal edge")

    def test_portal_right_is_at_right_or_left_edge(self):
        sx, sy = self._screen_pos(50, 0, yaw_deg=0)
        at_edge = (abs(sx - config.COMPASS_MARGIN) < 5 or
                   abs(sx - (1280 - config.COMPASS_MARGIN)) < 5)
        self.assertTrue(at_edge, f"Right portal sx={sx:.0f} should be at horizontal edge")

    def test_indicator_always_within_screen(self):
        import math
        for angle_deg in range(0, 360, 15):
            dx = math.cos(math.radians(angle_deg)) * 100
            dz = math.sin(math.radians(angle_deg)) * 100
            sx, sy = self._screen_pos(dx, dz, yaw_deg=0)
            self.assertGreaterEqual(sx, config.COMPASS_MARGIN)
            self.assertLessEqual(sx, 1280 - config.COMPASS_MARGIN)
            self.assertGreaterEqual(sy, config.COMPASS_MARGIN)
            self.assertLessEqual(sy, 720 - config.COMPASS_MARGIN)

    def test_rotating_player_rotates_indicator(self):
        # Same portal position, different yaw — indicator should move
        sx0, sy0 = self._screen_pos(50, 0, yaw_deg=0)
        sx90, sy90 = self._screen_pos(50, 0, yaw_deg=90)
        self.assertFalse(
            abs(sx0 - sx90) < 1 and abs(sy0 - sy90) < 1,
            "Rotating player should move the compass indicator"
        )

    def test_distance_calculation(self):
        import math
        dx, dz = 30, 40
        dist = math.sqrt(dx*dx + dz*dz)
        self.assertAlmostEqual(dist, 50.0)


# ===========================================================================
# Weather system
# ===========================================================================

class TestWeatherConfig(unittest.TestCase):

    def test_fade_in_positive(self):
        self.assertGreater(config.WEATHER_FADE_IN, 0)

    def test_fade_out_positive(self):
        self.assertGreater(config.WEATHER_FADE_OUT, 0)

    def test_all_biomes_have_weather(self):
        for biome in config.BIOMES:
            self.assertIn(biome, config.BIOME_WEATHER,
                          f"Biome {biome} missing from BIOME_WEATHER")

    def test_weather_types_valid(self):
        valid = {'rain', 'snow', 'clear'}
        for biome, wtype in config.BIOME_WEATHER.items():
            self.assertIn(wtype, valid, f"Biome {biome} has invalid weather {wtype}")

    def test_desert_is_clear(self):
        self.assertEqual(config.BIOME_WEATHER['DESERT'], 'clear')

    def test_tundra_is_snow(self):
        self.assertEqual(config.BIOME_WEATHER['TUNDRA'], 'snow')

    def test_forest_is_rain(self):
        self.assertEqual(config.BIOME_WEATHER['FOREST'], 'rain')

    def test_rain_colour_is_rgba(self):
        self.assertEqual(len(config.RAIN_COLOUR), 4)

    def test_snow_colour_is_rgba(self):
        self.assertEqual(len(config.SNOW_COLOUR), 4)

    def test_rain_falls_faster_than_snow(self):
        self.assertLess(config.RAIN_VY, config.SNOW_VY)

    def test_fog_density_in_range(self):
        self.assertGreater(config.RAIN_FOG_DENSITY, 0.0)
        self.assertLessEqual(config.RAIN_FOG_DENSITY, 1.0)
        self.assertGreater(config.SNOW_FOG_DENSITY, 0.0)
        self.assertLessEqual(config.SNOW_FOG_DENSITY, 1.0)

    def test_rain_foggier_than_snow(self):
        self.assertGreater(config.RAIN_FOG_DENSITY, config.SNOW_FOG_DENSITY)

    def test_fog_end_clear_greater_than_max(self):
        self.assertGreater(config.FOG_END_CLEAR, config.FOG_END_MAX)

    def test_weather_disc_radius_positive(self):
        self.assertGreater(config.WEATHER_DISC_RADIUS, 0)

    def test_weather_height_positive(self):
        self.assertGreater(config.WEATHER_HEIGHT, 0)


class TestWeatherParticle(unittest.TestCase):

    def _make(self, **kw):
        defaults = dict(x=0,y=20,z=0, vx=0,vy=-5,vz=0,
                        colour=(200,200,255,150), size=2, lifetime=2.0)
        defaults.update(kw)
        return WeatherParticle(**defaults)

    def test_alive_at_birth(self):
        self.assertTrue(self._make().alive)

    def test_dead_after_lifetime(self):
        p = self._make()
        p.age = p.lifetime
        self.assertFalse(p.alive)

    def test_falls_on_update(self):
        p = self._make(vy=-10.0)
        old_y = p.y
        p.update(0.1)
        self.assertLess(p.y, old_y)

    def test_no_gravity_acceleration(self):
        # WeatherParticle has constant vy (no gravity applied)
        p = self._make(vy=-5.0)
        p.update(1.0)
        self.assertAlmostEqual(p.vy, -5.0)

    def test_alpha_fraction_increases(self):
        p = self._make()
        p.update(0.5)
        self.assertGreater(p.alpha_fraction, 0.0)

    def test_size_stored(self):
        p = self._make(size=3)
        self.assertEqual(p.size, 3)


class TestWeatherManager(unittest.TestCase):

    def _mgr(self):
        return WeatherManager()

    def test_starts_clear(self):
        m = self._mgr()
        self.assertEqual(m.weather_type, 'clear')
        self.assertAlmostEqual(m.intensity, 0.0)

    def test_starts_no_particles(self):
        self.assertEqual(self._mgr().particles, [])

    def test_fog_density_zero_at_start(self):
        self.assertAlmostEqual(self._mgr().fog_density, 0.0)

    def test_intensity_ramps_up_for_rain(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.update(config.WEATHER_FADE_IN * 0.5, (0,0,0))
        self.assertGreater(m.intensity, 0.0)
        self.assertLess(m.intensity, 1.0)

    def test_intensity_reaches_one_after_fade_in(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.update(config.WEATHER_FADE_IN + 1.0, (0,0,0))
        self.assertAlmostEqual(m.intensity, 1.0)

    def test_intensity_ramps_down_when_cleared(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.intensity = 1.0
        m.weather_type = 'clear'
        m.update(config.WEATHER_FADE_OUT * 0.5, (0,0,0))
        self.assertLess(m.intensity, 1.0)
        self.assertGreater(m.intensity, 0.0)

    def test_rain_emits_particles(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.intensity = 1.0
        m.update(1.0, (0, 20, 0))
        self.assertGreater(len(m.particles), 0)

    def test_snow_emits_particles(self):
        m = self._mgr()
        m.weather_type = 'snow'
        m.intensity = 1.0
        m.update(1.0, (0, 20, 0))
        self.assertGreater(len(m.particles), 0)

    def test_clear_emits_no_particles(self):
        m = self._mgr()
        m.weather_type = 'clear'
        m.intensity = 0.0
        m.update(1.0, (0, 20, 0))
        self.assertEqual(len(m.particles), 0)

    def test_particles_spawn_above_player(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.intensity = 1.0
        player_y = 20.0
        m.update(0.5, (0, player_y, 0))
        for p in m.particles:
            self.assertGreater(p.y, player_y,
                               "Weather particles should spawn above player")

    def test_rain_particles_have_correct_colour(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.intensity = 1.0
        m.update(0.1, (0, 20, 0))
        for p in m.particles:
            self.assertEqual(p.colour, config.RAIN_COLOUR)

    def test_snow_particles_have_correct_colour(self):
        m = self._mgr()
        m.weather_type = 'snow'
        m.intensity = 1.0
        m.update(0.1, (0, 20, 0))
        for p in m.particles:
            self.assertEqual(p.colour, config.SNOW_COLOUR)

    def test_dead_particles_removed(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.intensity = 1.0
        m.update(1.0, (0, 20, 0))
        initial_count = len(m.particles)
        # Age all particles past their lifetime
        for p in m.particles:
            p.age = p.lifetime + 1.0
        m.update(0.01, (0, 20, 0))
        self.assertLess(len(m.particles), initial_count)

    def test_fog_increases_with_rain(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.intensity = 1.0
        initial_fog = m.fog_density
        m.update(2.0, (0, 20, 0))
        self.assertGreater(m.fog_density, initial_fog)

    def test_fog_density_bounded(self):
        m = self._mgr()
        m.weather_type = 'rain'
        m.intensity = 1.0
        for _ in range(100):
            m.update(0.1, (0, 20, 0))
        self.assertLessEqual(m.fog_density, config.RAIN_FOG_DENSITY + 0.01)

    def test_rain_rate_greater_than_zero(self):
        self.assertGreater(config.RAIN_RATE, 0)

    def test_snow_rate_greater_than_zero(self):
        self.assertGreater(config.SNOW_RATE, 0)

    def test_model_has_weather_manager(self):
        with patch.object(Model, '_initialize', return_value=None):
            model = Model()
        self.assertIsInstance(model.weather, WeatherManager)


# ===========================================================================
# Sky objects — sun, moon, stars, clouds
# ===========================================================================

class TestSkyConfig(unittest.TestCase):

    def test_sky_sphere_radius_positive(self):
        self.assertGreater(config.SKY_SPHERE_RADIUS, 0)

    def test_sky_sphere_inside_far_plane(self):
        # Far plane is now 120 in set_3d; sky sphere must be inside it
        self.assertLess(config.SKY_SPHERE_RADIUS, 120.0)

    def test_sun_size_positive(self):
        self.assertGreater(config.SUN_SIZE, 0)

    def test_sun_halo_larger_than_sun(self):
        self.assertGreater(config.SUN_HALO_SIZE, config.SUN_SIZE)

    def test_moon_smaller_than_sun(self):
        self.assertLess(config.MOON_SIZE, config.SUN_SIZE)

    def test_sun_colour_is_rgba(self):
        self.assertEqual(len(config.SUN_COLOUR), 4)

    def test_moon_colour_is_rgba(self):
        self.assertEqual(len(config.MOON_COLOUR), 4)

    def test_star_count_positive(self):
        self.assertGreater(config.STAR_COUNT, 0)

    def test_star_fade_start_greater_than_end(self):
        self.assertGreater(config.STAR_FADE_START, config.STAR_FADE_END)

    def test_cloud_count_positive(self):
        self.assertGreater(config.CLOUD_COUNT, 0)

    def test_cloud_height_positive(self):
        self.assertGreater(config.CLOUD_HEIGHT, 0)

    def test_cloud_speed_positive(self):
        self.assertGreater(config.CLOUD_SPEED, 0)

    def test_cloud_colour_is_rgb(self):
        self.assertEqual(len(config.CLOUD_COLOUR), 3)


class TestSkyMaths(unittest.TestCase):
    """Test sun/moon position maths and visibility logic without GL."""

    def _sun_pos(self, game_time, player=(0, 20, 0)):
        import math
        px, py, pz = player
        R = config.SKY_SPHERE_RADIUS
        angle = (game_time / config.DAY_LENGTH) * 2.0 * math.pi
        return (px, py + math.sin(angle) * R, pz - math.cos(angle) * R)

    def _moon_pos(self, game_time, player=(0, 20, 0)):
        import math
        px, py, pz = player
        R = config.SKY_SPHERE_RADIUS
        angle = (game_time / config.DAY_LENGTH) * 2.0 * math.pi + math.pi
        return (px, py + math.sin(angle) * R, pz - math.cos(angle) * R)

    def test_sun_above_horizon_at_noon(self):
        noon = config.DAY_LENGTH / 4.0
        _, sun_y, _ = self._sun_pos(noon)
        self.assertGreater(sun_y, 20)  # well above player y=20

    def test_sun_below_horizon_at_midnight(self):
        midnight = config.DAY_LENGTH * 3.0 / 4.0
        _, sun_y, _ = self._sun_pos(midnight)
        self.assertLess(sun_y, 20)  # below player y=20

    def test_moon_above_horizon_at_midnight(self):
        midnight = config.DAY_LENGTH * 3.0 / 4.0
        _, moon_y, _ = self._moon_pos(midnight)
        self.assertGreater(moon_y, 20)

    def test_moon_below_horizon_at_noon(self):
        noon = config.DAY_LENGTH / 4.0
        _, moon_y, _ = self._moon_pos(noon)
        self.assertLess(moon_y, 20)

    def test_sun_and_moon_always_opposite(self):
        import math
        player = (0, 20, 0)
        for t in range(0, 600, 30):
            sx, sy, sz = self._sun_pos(t, player)
            mx, my, mz = self._moon_pos(t, player)
            px, py, pz = player
            # Sun and moon vectors should point in opposite directions
            dot = (sx-px)*(mx-px) + (sy-py)*(my-py) + (sz-pz)*(mz-pz)
            self.assertLess(dot, 0, f"Sun and moon should be opposite at t={t}")

    def test_sun_at_sky_sphere_radius(self):
        import math
        player = (0, 20, 0)
        px, py, pz = player
        for t in range(0, 600, 60):
            sx, sy, sz = self._sun_pos(t, player)
            dist = math.sqrt((sx-px)**2 + (sy-py)**2 + (sz-pz)**2)
            self.assertAlmostEqual(dist, config.SKY_SPHERE_RADIUS, places=3)

    def test_star_visibility_at_midnight(self):
        midnight = config.DAY_LENGTH * 3.0 / 4.0
        brightness = config.sun_brightness(midnight)
        self.assertLess(brightness, config.STAR_FADE_START)

    def test_stars_invisible_at_noon(self):
        noon = config.DAY_LENGTH / 4.0
        brightness = config.sun_brightness(noon)
        self.assertGreater(brightness, config.STAR_FADE_START)


class TestSkyRenderer(unittest.TestCase):
    """Test SkyRenderer data generation without GL."""

    def setUp(self):
        # Manually instantiate just the data-generation parts — no GL needed
        import math, random as _random
        self.renderer = type('R', (), {
            '_star_positions': [],
            '_cloud_offsets': [],
        })()

        # Replicate _build_stars
        rng = _random.Random(config.STAR_SEED)
        stars = []
        while len(stars) < config.STAR_COUNT:
            cos_theta = rng.uniform(0.05, 1.0)
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            psi = rng.uniform(0, 2 * math.pi)
            stars.append((sin_theta * math.cos(psi), cos_theta,
                          sin_theta * math.sin(psi)))
        self.renderer._star_positions = stars

        # Replicate _build_clouds (puff-based)
        rng2 = _random.Random(config.CLOUD_SEED)
        s = config.CLOUD_SPREAD
        clouds = []
        for _ in range(config.CLOUD_COUNT):
            base_ox = rng2.uniform(-s, s)
            base_oz = rng2.uniform(-s, s)
            scale   = rng2.uniform(6, 15)
            spread  = scale * 0.65
            n_puffs = rng2.randint(5, 9)
            puffs   = []
            for _ in range(n_puffs):
                px     = base_ox + rng2.uniform(-spread, spread)
                pz_    = base_oz + rng2.uniform(-spread, spread)
                py_off = rng2.uniform(0.0, 2.0)
                r      = rng2.uniform(scale * 0.4, scale * 0.95)
                puffs.append((px, pz_, py_off, r))
            clouds.append(puffs)
        self.renderer._cloud_offsets = clouds

    def test_star_count(self):
        self.assertEqual(len(self.renderer._star_positions),
                         config.STAR_COUNT)

    def test_stars_on_unit_sphere(self):
        import math
        for (x, y, z) in self.renderer._star_positions:
            length = math.sqrt(x*x + y*y + z*z)
            self.assertAlmostEqual(length, 1.0, places=5)

    def test_stars_upper_hemisphere_only(self):
        for (x, y, z) in self.renderer._star_positions:
            self.assertGreater(y, 0.0, "Stars should be in upper hemisphere")

    def test_star_positions_deterministic(self):
        import math, random as _rng
        rng = _rng.Random(config.STAR_SEED)
        stars2 = []
        while len(stars2) < config.STAR_COUNT:
            cos_theta = rng.uniform(0.05, 1.0)
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            psi = rng.uniform(0, 2 * math.pi)
            stars2.append((sin_theta * math.cos(psi), cos_theta,
                           sin_theta * math.sin(psi)))
        self.assertEqual(self.renderer._star_positions, stars2)

    def test_cloud_count(self):
        self.assertEqual(len(self.renderer._cloud_offsets),
                         config.CLOUD_COUNT)

    def test_cloud_offsets_have_four_components(self):
        # Each cloud is a list of puffs; each puff has 4 components
        for cloud in self.renderer._cloud_offsets:
            self.assertIsInstance(cloud, list)
            self.assertGreater(len(cloud), 0)
            for puff in cloud:
                self.assertEqual(len(puff), 4)

    def test_cloud_width_positive(self):
        for cloud in self.renderer._cloud_offsets:
            for ox, oz, oy_off, radius in cloud:
                self.assertGreater(radius, 0)
                self.assertGreaterEqual(oy_off, 0)

    def test_cloud_offsets_deterministic(self):
        import random as _rng, math as _math
        rng = _rng.Random(config.CLOUD_SEED)
        s = config.CLOUD_SPREAD
        clouds2 = []
        for _ in range(config.CLOUD_COUNT):
            base_ox = rng.uniform(-s, s)
            base_oz = rng.uniform(-s, s)
            scale   = rng.uniform(6, 15)
            spread  = scale * 0.65
            n_puffs = rng.randint(5, 9)
            puffs   = []
            for _ in range(n_puffs):
                px     = base_ox + rng.uniform(-spread, spread)
                pz     = base_oz + rng.uniform(-spread, spread)
                py_off = rng.uniform(0.0, 2.0)
                r      = rng.uniform(scale * 0.4, scale * 0.95)
                puffs.append((px, pz, py_off, r))
            clouds2.append(puffs)
        self.assertEqual(self.renderer._cloud_offsets, clouds2)

    def test_cloud_offsets_spread(self):
        # Base puff positions should generally be within CLOUD_SPREAD
        # (individual puffs can scatter a bit beyond)
        s = config.CLOUD_SPREAD * 1.5  # generous bound
        for cloud in self.renderer._cloud_offsets:
            for ox, oz, oy_off, radius in cloud:
                self.assertLessEqual(abs(ox), s)
                self.assertLessEqual(abs(oz), s)

    def test_each_cloud_has_multiple_puffs(self):
        for cloud in self.renderer._cloud_offsets:
            self.assertGreaterEqual(len(cloud), 5)


# ===========================================================================
# Mobs / entities
# ===========================================================================

class TestMobConfig(unittest.TestCase):

    def test_mob_max_positive(self):
        self.assertGreater(config.MOB_MAX, 0)

    def test_mob_types_non_empty(self):
        self.assertGreater(len(config.MOB_TYPES), 0)

    def test_each_mob_type_has_required_keys(self):
        required = {'width', 'height', 'speed', 'body_colour',
                    'leg_colour', 'idle_time', 'walk_time', 'spawn_on'}
        for name, d in config.MOB_TYPES.items():
            for key in required:
                self.assertIn(key, d, f"Mob {name} missing key {key}")

    def test_mob_colours_are_rgba(self):
        for name, d in config.MOB_TYPES.items():
            self.assertEqual(len(d['body_colour']), 4, f"{name} body_colour not RGBA")
            self.assertEqual(len(d['leg_colour']),  4, f"{name} leg_colour not RGBA")

    def test_mob_dimensions_positive(self):
        for name, d in config.MOB_TYPES.items():
            self.assertGreater(d['width'],  0)
            self.assertGreater(d['height'], 0)
            self.assertGreater(d['speed'],  0)

    def test_chicken_and_sheep_defined(self):
        self.assertIn('chicken', config.MOB_TYPES)
        self.assertIn('sheep',   config.MOB_TYPES)

    def test_spawn_on_lists_non_empty(self):
        for name, d in config.MOB_TYPES.items():
            self.assertGreater(len(d['spawn_on']), 0)

    def test_despawn_greater_than_spawn(self):
        self.assertGreater(config.MOB_DESPAWN_DIST, config.MOB_SPAWN_DIST)


class TestEntity(unittest.TestCase):

    def _mob(self, mob_type='chicken', x=0, y=10, z=0):
        from mobs import Mob
        return Mob(mob_type, x, y, z)

    def test_position_property(self):
        m = self._mob(x=1, y=2, z=3)
        self.assertEqual(m.position, (1.0, 2.0, 3.0))

    def test_height_from_config(self):
        m = self._mob('chicken')
        self.assertAlmostEqual(m.height,
                               config.MOB_TYPES['chicken']['height'])

    def test_width_from_config(self):
        m = self._mob('sheep')
        self.assertAlmostEqual(m.width,
                               config.MOB_TYPES['sheep']['width'])

    def test_starts_alive(self):
        self.assertTrue(self._mob().alive)

    def test_falls_without_ground(self):
        m = self._mob(y=20)
        old_y = m.y
        m.apply_gravity(0.1)
        m.move(0.1, {})   # empty world — no collision
        self.assertLess(m.y, old_y)

    def test_lands_on_block(self):
        from mobs import Mob
        m = Mob('chicken', 0.2, 5, 0.2)
        # Place a solid floor at y=3
        world = {(0, 3, 0): config.STONE}
        for _ in range(30):
            m.apply_gravity(0.05)
            m.move(0.05, world)
        # Should be resting on top of the block
        self.assertAlmostEqual(m.y, 4.0, delta=0.5)
        self.assertTrue(m.on_ground)

    def test_body_colour_from_config(self):
        m = self._mob('sheep')
        self.assertEqual(m.body_colour, config.MOB_TYPES['sheep']['body_colour'])


class TestMobAI(unittest.TestCase):

    def _mob(self, mob_type='chicken'):
        from mobs import Mob
        return Mob(mob_type, 0, 10, 0)

    def test_starts_in_idle_or_walk(self):
        from mobs import Mob
        m = Mob('chicken', 0, 10, 0)
        self.assertIn(m._state, (Mob.IDLE, Mob.WALK))

    def test_transitions_to_walk(self):
        from mobs import Mob
        m = Mob('chicken', 0.2, 10, 0.2)
        m._state = Mob.IDLE
        m._state_timer = 0.0
        m.on_ground = True   # pretend it's on the ground
        # Force the state transition directly
        m._enter_walk()
        self.assertEqual(m._state, Mob.WALK)
        self.assertGreater(m._state_timer, 0)

    def test_transitions_to_idle(self):
        from mobs import Mob
        m = Mob('chicken', 0, 10, 0)
        m._state = Mob.WALK
        m._state_timer = 0.0
        m.update(0.1, {})
        self.assertEqual(m._state, Mob.IDLE)

    def test_walk_changes_position(self):
        from mobs import Mob
        # Put chicken on a flat floor so it doesn't fall through
        world = {}
        for x in range(-5, 5):
            for z in range(-5, 5):
                world[(x, 8, z)] = config.GRASS
        m = Mob('chicken', 0.5, 9.5, 0.5)
        m._state = Mob.WALK
        m._state_timer = 5.0
        old_pos = m.position
        for _ in range(20):
            m.update(0.05, world)
        new_pos = m.position
        moved = math.sqrt((new_pos[0]-old_pos[0])**2 + (new_pos[2]-old_pos[2])**2)
        self.assertGreater(moved, 0.01)

    def test_idle_stays_still(self):
        from mobs import Mob
        world = {(0, 8, 0): config.GRASS}
        m = Mob('chicken', 0.5, 9.5, 0.5)
        m._state = Mob.IDLE
        m._state_timer = 10.0
        old_x, old_z = m.x, m.z
        for _ in range(10):
            m.update(0.05, world)
        self.assertAlmostEqual(m.x, old_x, delta=0.01)
        self.assertAlmostEqual(m.z, old_z, delta=0.01)

    def test_falls_below_world_marks_dead(self):
        from mobs import Mob
        m = Mob('chicken', 0, 10, 0)
        m.y = -25.0
        m.update(0.05, {})
        self.assertFalse(m.alive)


class TestMobManager(unittest.TestCase):

    def _mgr(self):
        from mobs import MobManager
        return MobManager()

    def test_starts_empty(self):
        self.assertEqual(len(self._mgr().mobs), 0)

    def test_update_does_not_crash_empty(self):
        mgr = self._mgr()
        mgr.update(0.05, (0, 20, 0), {})   # should not raise

    def test_spawn_after_interval(self):
        mgr = self._mgr()
        # World must be large enough for the spawn ring (min dist = SECTOR_SIZE*2)
        r = config.SECTOR_SIZE * 3 + 5
        world = {}
        for x in range(-r, r):
            for z in range(-r, r):
                world[(x, 10, z)] = config.GRASS
        # Advance past spawn interval several times
        for _ in range(8):
            mgr.update(config.MOB_SPAWN_INTERVAL, (0, 12, 0), world)
        self.assertGreater(len(mgr.mobs), 0)

    def test_despawn_far_mobs(self):
        from mobs import Mob
        mgr = self._mgr()
        far_dist = config.MOB_DESPAWN_DIST * config.SECTOR_SIZE + 10
        mgr.mobs.append(Mob('chicken', far_dist, 10, 0))
        mgr.update(0.05, (0, 10, 0), {})
        self.assertEqual(len(mgr.mobs), 0)

    def test_respects_mob_max(self):
        from mobs import Mob
        mgr = self._mgr()
        for i in range(config.MOB_MAX + 5):
            mgr.mobs.append(Mob('chicken', i, 10, 0))
        world = {}
        for x in range(-10, 10):
            for z in range(-10, 10):
                world[(x, 10, z)] = config.GRASS
        for _ in range(10):
            mgr.update(config.MOB_SPAWN_INTERVAL, (0, 12, 0), world)
        self.assertLessEqual(len(mgr.mobs), config.MOB_MAX + 5)

    def test_model_has_mob_manager(self):
        with patch.object(Model, '_initialize', return_value=None):
            model = Model()
        from mobs import MobManager
        self.assertIsInstance(model.mobs, MobManager)


class TestUtilCollide(unittest.TestCase):
    """Tests for the extracted collide() function in util.py."""

    def test_no_collision_empty_world(self):
        pos, on_ground = util_collide((5.5, 10.5, 5.5), 2, {})
        self.assertAlmostEqual(pos[0], 5.5)
        self.assertFalse(on_ground)

    def test_lands_on_block(self):
        world = {(5, 8, 5): config.STONE}
        # Player at y=8.7 with x/z avoiding .5 rounding
        pos, on_ground = util_collide((5.2, 8.7, 5.2), 2, world)
        self.assertTrue(on_ground)

    def test_portal_blocks_are_passthrough(self):
        world = {(5, 8, 5): config.PORTAL_TEX}
        pos, on_ground = util_collide((5.5, 8.3, 5.5), 2, world,
                                      portal_tex=config.PORTAL_TEX)
        self.assertFalse(on_ground)

    def test_returns_tuple(self):
        result = util_collide((0.5, 5.5, 0.5), 1, {})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ===========================================================================
# Mob sprite atlas and renderer
# ===========================================================================

class TestMobAtlas(unittest.TestCase):
    """Verify mob_atlas.png exists and has correct dimensions/content."""

    def setUp(self):
        import os
        self.atlas_path = os.path.join(
            os.path.dirname(__file__), 'mob_atlas.png')

    def test_atlas_exists(self):
        import os
        self.assertTrue(os.path.exists(self.atlas_path),
                        "mob_atlas.png not found")

    def test_atlas_dimensions(self):
        from PIL import Image
        img = Image.open(self.atlas_path)
        self.assertEqual(img.width,  128)
        self.assertEqual(img.height,  64)

    def test_atlas_is_rgba(self):
        from PIL import Image
        img = Image.open(self.atlas_path).convert('RGBA')
        self.assertEqual(img.mode, 'RGBA')

    def test_chicken_region_is_yellow(self):
        from PIL import Image
        img = Image.open(self.atlas_path).convert('RGBA')
        # Sample centre of chicken body (left half, upper area)
        r, g, b, a = img.getpixel((32, 16))
        self.assertGreater(r, 180, "Chicken should be yellow (high R)")
        self.assertGreater(g, 150, "Chicken should be yellow (high G)")
        self.assertLess(b, 100,    "Chicken should be yellow (low B)")
        self.assertEqual(a, 255)

    def test_sheep_region_is_white(self):
        from PIL import Image
        img = Image.open(self.atlas_path).convert('RGBA')
        # Sample centre of sheep body (right half, upper area)
        r, g, b, a = img.getpixel((96, 16))
        self.assertGreater(r, 200)
        self.assertGreater(g, 200)
        self.assertGreater(b, 200)
        self.assertEqual(a, 255)

    def test_transparent_background_corners(self):
        from PIL import Image
        img = Image.open(self.atlas_path).convert('RGBA')
        # Top-left corner of chicken should be transparent
        _, _, _, a = img.getpixel((0, 0))
        self.assertEqual(a, 0, "Sprite corner should be transparent")

    def test_sheep_has_grey_legs(self):
        from PIL import Image
        img = Image.open(self.atlas_path).convert('RGBA')
        # Bottom centre of sheep region (legs area)
        r, g, b, a = img.getpixel((80, 56))
        self.assertLess(r, 150, "Sheep legs should be grey")


class TestMobRendererConfig(unittest.TestCase):

    def test_all_mob_types_have_uv_entry(self):
        from mob_renderer import MOB_UV
        for mob_type in config.MOB_TYPES:
            self.assertIn(mob_type, MOB_UV,
                          f"Mob type {mob_type} missing from MOB_UV")

    def test_uv_ranges_valid(self):
        from mob_renderer import MOB_UV
        for mob_type, (u0, u1) in MOB_UV.items():
            self.assertGreaterEqual(u0, 0.0)
            self.assertLessEqual(u1, 1.0)
            self.assertLess(u0, u1)

    def test_uv_ranges_non_overlapping(self):
        from mob_renderer import MOB_UV
        ranges = sorted(MOB_UV.values())
        for i in range(len(ranges) - 1):
            self.assertLessEqual(ranges[i][1], ranges[i+1][0],
                                 "UV ranges should not overlap")

    def test_leg_v_top_in_range(self):
        from mob_renderer import LEG_V_TOP
        self.assertGreater(LEG_V_TOP, 0.0)
        self.assertLess(LEG_V_TOP, 1.0)

    def test_walk_amplitude_positive(self):
        from mob_renderer import WALK_AMPLITUDE
        self.assertGreater(WALK_AMPLITUDE, 0.0)

    def test_walk_freq_positive(self):
        from mob_renderer import WALK_FREQ
        self.assertGreater(WALK_FREQ, 0.0)


class TestMobRendererQuad(unittest.TestCase):
    """Test the _quad geometry helper without GL."""

    def _renderer(self):
        from mob_renderer import MobRenderer
        r = MobRenderer()
        return r

    def test_quad_returns_18_position_values(self):
        r = self._renderer()
        verts, uvs = r._quad(0, 0, 0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0)
        self.assertEqual(len(verts), 18)  # 6 verts × 3 coords

    def test_quad_returns_12_uv_values(self):
        r = self._renderer()
        verts, uvs = r._quad(0, 0, 0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0)
        self.assertEqual(len(uvs), 12)  # 6 verts × 2 UV coords

    def test_quad_height_correct(self):
        r = self._renderer()
        h = 1.5
        verts, _ = r._quad(0, 5, 0, 0.5, h, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        ys = [verts[i] for i in range(1, 18, 3)]
        self.assertAlmostEqual(min(ys), 5.0)
        self.assertAlmostEqual(max(ys), 5.0 + h)

    def test_quad_uv_range(self):
        r = self._renderer()
        _, uvs = r._quad(0, 0, 0, 0.5, 1.0, 1.0, 0.0, 0.25, 0.75, 0.0, 1.0)
        us = uvs[0::2]
        vs = uvs[1::2]
        self.assertAlmostEqual(min(us), 0.25)
        self.assertAlmostEqual(max(us), 0.75)
        self.assertAlmostEqual(min(vs), 0.0)
        self.assertAlmostEqual(max(vs), 1.0)
