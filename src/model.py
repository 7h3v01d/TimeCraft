# model.py

import random
import time
from collections import deque
import math

import pyglet
from pyglet.gl import gl
from pyglet import image
from pyglet.graphics import TextureGroup

import config
from noise_gen import NoiseGen
from util import sectorize, cube_vertices, normalize

class Particle:
    def __init__(self, x, y, z, vx, vy, vz, texture, lifetime=0.5):
        self.sprite = pyglet.sprite.Sprite(texture, x=x, y=y, z=z)
        self.vx, self.vy, self.vz = vx, vy, vz
        self.lifetime = lifetime
        self.time = 0

    def update(self, dt):
        self.time += dt
        self.sprite.x += self.vx * dt
        self.sprite.y += self.vy * dt
        self.sprite.z += self.vz * dt
        self.sprite.opacity = 255 * (1 - self.time / self.lifetime)
        if self.sprite.image.get_texture().owner == config.PARTICLE_CRYSTAL:
            self.sprite.scale = 1.0 + 0.2 * math.sin(self.time * 10)
        return self.time < self.lifetime

class Model(object):
    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        self.group = TextureGroup(image.load(config.TEXTURE_PATH).get_texture())
        self.water_shader = pyglet.graphics.shader.ShaderProgram(
            pyglet.graphics.shader.Shader(open('water_vertex.glsl').read(), 'vertex'),
            pyglet.graphics.shader.Shader(open('water_fragment.glsl').read(), 'fragment')
        )
        self.world = {}
        self.shown = {}
        self._shown = {}
        self.sectors = {}
        self.queue = deque()
        self.particles = []
        self.particle_textures = {
            config.STONE: pyglet.image.create(4, 4, pyglet.image.SolidColorImagePattern(config.PARTICLE_STONE)),
            config.WOOD: pyglet.image.create(4, 4, pyglet.image.SolidColorImagePattern(config.PARTICLE_WOOD)),
            config.LEAF: pyglet.image.create(4, 4, pyglet.image.SolidColorImagePattern(config.PARTICLE_LEAF)),
            config.CRYSTAL: pyglet.image.create(4, 4, pyglet.image.SolidColorImagePattern(config.PARTICLE_CRYSTAL)),
        }
        self.game_time = 0
        self._initialize()

    def _initialize(self):
        """ Generate the world terrain. """
        gen = NoiseGen(self.seed if hasattr(self, 'seed') else random.randint(0, 1000000))
        n = 128
        s = 1
        y = 0

        heightMap = [int(gen.getHeight(x, z)) for x in range(n) for z in range(n)]

        for x in range(0, n, s):
            for z in range(0, n, s):
                h = heightMap[x * n + z]
                if h < 15:
                    self.add_block((x, h, z), config.SAND, immediate=False)
                    for y in range(h, 15):
                        self.add_block((x, y, z), config.MAGIC_WATER if random.random() > 0.99 else config.WATER, immediate=False)
                    continue
                if h < 18:
                    self.add_block((x, h, z), config.SAND, immediate=False)
                
                self.add_block((x, h, z), config.GRASS, immediate=False)
                for y in range(h - 1, 0, -1):
                    self.add_block((x, y, z), config.STONE, immediate=False)
                
                if h > 20 and random.random() > 0.99:
                    treeHeight = random.randint(5, 7)
                    for y in range(h + 1, h + treeHeight):
                        self.add_block((x, y, z), config.WOOD, immediate=False)
                    leafh = h + treeHeight
                    for lz in range(z - 2, z + 3):
                        for lx in range(x - 2, x + 3):
                            for ly in range(3):
                                if (lx, leafh + ly, lz) != (x, leafh+ly, z) or random.random() > 0.1:
                                    self.add_block((lx, leafh + ly, lz), config.LEAF, immediate=False)
                
                if h > 25 and random.random() > 0.995:
                    self.add_block((x, h - 1, z), config.CRYSTAL, immediate=False)

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
        x, y, z = position
        for _ in range(5):
            vx = random.uniform(-0.3, 0.3)
            vy = random.uniform(0, 0.5)
            vz = random.uniform(-0.3, 0.3)
            if hit_vector:
                vx += hit_vector[0] * 0.2
                vy += hit_vector[1] * 0.2
                vz += hit_vector[2] * 0.2
            particle_tex = self.particle_textures.get(texture, self.particle_textures[config.STONE])
            self.particles.append(Particle(x, y, z, vx, vy, vz, particle_tex))

    def remove_block(self, position, immediate=True, hit_vector=None):
        texture = self.world[position]
        del self.world[position]
        self.sectors[sectorize(position, config.SECTOR_SIZE)].remove(position)
        if immediate:
            if position in self.shown:
                self.hide_block(position)
            self.check_neighbors(position)
        x, y, z = position
        for _ in range(5):
            vx = random.uniform(-0.3, 0.3)
            vy = random.uniform(0, 0.5)
            vz = random.uniform(-0.3, 0.3)
            if hit_vector:
                vx += hit_vector[0] * 0.2
                vy += hit_vector[1] * 0.2
                vz += hit_vector[2] * 0.2
            particle_tex = self.particle_textures.get(texture, self.particle_textures[config.STONE])
            self.particles.append(Particle(x, y, z, vx, vy, vz, particle_tex))

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
        if texture in (config.WATER, config.MAGIC_WATER):
            with self.water_shader:
                self.water_shader.uniformf('time', self.game_time)
                self.water_shader.uniformi('is_magic_water', 1 if texture == config.MAGIC_WATER else 0)
                self._shown[position] = self.batch.add(24, gl.GL_QUADS, self.group,
                                                      ('v3f/static', vertex_data),
                                                      ('t2f/static', texture_data))
        else:
            self._shown[position] = self.batch.add(24, gl.GL_QUADS, self.group,
                                                  ('v3f/static', vertex_data),
                                                  ('t2f/static', texture_data))

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
        for position in self.sectors.get(sector, []):
            if position not in self.shown and self.exposed(position):
                self.show_block(position, False)

    def hide_sector(self, sector):
        for position in self.sectors.get(sector, []):
            if position in self.shown:
                self.hide_block(position, False)

    def change_sectors(self, before, after):
        before_set = set()
        after_set = set()
        pad = 4
        for dx in range(-pad, pad + 1):
            for dy in [0]:
                for dz in range(-pad, pad + 1):
                    if dx ** 2 + dy ** 2 + dz ** 2 > (pad + 1) ** 2:
                        continue
                    if before:
                        x, y, z = before
                        before_set.add((x + dx, y + dy, z + dz))
                    if after:
                        x, y, z = after
                        after_set.add((x + dx, y + dy, z + dz))
        show = after_set - before_set
        hide = before_set - after_set
        for sector in show:
            self.show_sector(sector)
        for sector in hide:
            self.hide_sector(sector)

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
        self.particles = [p for p in self.particles if p.update(dt)]