"""
TimeCraft Test Suite
Tests all headless logic: util, config, noise_gen, and model (GL mocked).
Run from the timecraft/ directory:  pytest test_timecraft.py -v
"""

import sys
import os
import math
import types
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
    image_mod.load = MagicMock(return_value=MagicMock(get_texture=MagicMock(return_value=fake_tex)))
    image_mod.create = MagicMock()
    image_mod.SolidColorImagePattern = MagicMock()

    # pyglet.clock / app
    clock_mod = types.ModuleType("pyglet.clock")
    clock_mod.schedule_interval = MagicMock()
    app_mod = types.ModuleType("pyglet.app")

    pyglet.graphics = graphics
    pyglet.gl = gl
    pyglet.image = image_mod
    pyglet.clock = clock_mod
    pyglet.app = app_mod

    sys.modules["pyglet"] = pyglet
    sys.modules["pyglet.graphics"] = graphics
    sys.modules["pyglet.graphics.shader"] = shader_mod
    sys.modules["pyglet.gl"] = gl
    sys.modules["pyglet.gl.gl"] = gl
    sys.modules["pyglet.image"] = image_mod
    sys.modules["pyglet.clock"] = clock_mod
    sys.modules["pyglet.app"] = app_mod

_make_pyglet_stub()

# Now safe to import game modules
sys.path.insert(0, ".")          # run from project root
sys.path.insert(0, "timecraft")  # or from parent

import config
import tempfile
import json
from util import cube_vertices, normalize, sectorize
from noise_gen import NoiseGen, NoiseParameters
from model import Model, QUAD_INDICES


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

    def test_texture_lookup_roundtrip(self):
        """Every named texture survives save->load name conversion."""
        from model import _TEXTURE_NAMES, _TEXTURE_LOOKUP
        for name, tex in _TEXTURE_NAMES.items():
            key = tuple(tex)
            self.assertIn(key, _TEXTURE_LOOKUP, f"{name} missing from lookup")
            self.assertEqual(_TEXTURE_LOOKUP[key], name)

    def test_save_creates_file(self):
        m = self._make_model()
        m.add_block((0, 0, 0), config.GRASS, immediate=False)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
        try:
            m.SAVE_FILE = tmp
            m.save_world()
            self.assertTrue(os.path.exists(tmp))
        finally:
            os.unlink(tmp)

    def test_save_format(self):
        m = self._make_model()
        m.seed = 42
        m.add_block((1, 2, 3), config.STONE, immediate=False)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            tmp = f.name
        try:
            m.SAVE_FILE = tmp
            m.save_world()
            with open(tmp) as f:
                data = json.load(f)
            self.assertEqual(data['seed'], 42)
            self.assertEqual(len(data['blocks']), 1)
            self.assertEqual(data['blocks'][0]['pos'], [1, 2, 3])
            self.assertEqual(data['blocks'][0]['tex'], 'STONE')
        finally:
            os.unlink(tmp)

    def test_save_returns_block_count(self):
        m = self._make_model()
        for i in range(5):
            m.add_block((i, 0, 0), config.GRASS, immediate=False)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
        try:
            m.SAVE_FILE = tmp
            count = m.save_world()
            self.assertEqual(count, 5)
        finally:
            os.unlink(tmp)

    def test_load_restores_blocks(self):
        # Save a world then load it into a fresh model
        m1 = self._make_model()
        m1.seed = 7
        m1.add_block((3, 4, 5), config.WOOD, immediate=False)
        m1.add_block((6, 7, 8), config.LEAF, immediate=False)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
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
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
        try:
            m1.SAVE_FILE = tmp
            m1.save_world()

            m2 = self._make_model()
            m2.SAVE_FILE = tmp
            m2.load_world()
            self.assertEqual(len(m2.world), 10)
        finally:
            os.unlink(tmp)

    def test_delete_save(self):
        m = self._make_model()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
        try:
            m.SAVE_FILE = tmp
            m.add_block((0,0,0), config.GRASS, immediate=False)
            m.save_world()
            self.assertTrue(os.path.exists(tmp))
            m.delete_save()
            self.assertFalse(os.path.exists(tmp))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_save_no_file_no_error(self):
        m = self._make_model()
        m.SAVE_FILE = '/tmp/timecraft_nonexistent_12345.json'
        m.delete_save()  # should not raise


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
        cx = cz = m.WORLD_SIZE // 2
        # Place water at surface, stone below
        m.add_block((cx, 15, cz), config.WATER, immediate=False)
        m.add_block((cx, 14, cz), config.STONE, immediate=False)
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
        m.add_block((64, 25, 64), config.GRASS, immediate=False)
        spawn = m.get_spawn_point()
        for coord in spawn:
            self.assertIsInstance(coord, float)

    def test_world_size_is_128(self):
        with patch.object(Model, '_initialize', return_value=None):
            m = Model()
        self.assertEqual(m.WORLD_SIZE, 128)


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
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
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
