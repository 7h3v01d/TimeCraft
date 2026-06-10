# mobs.py
#
# Entity base class, Mob subclass, and MobManager.
# Physics uses util.collide() — the same AABB system as the player.
# Rendering is handled by Window.draw_mobs() via billboard quads.

import math
import random

import config
from util import collide, normalize, sectorize


# ---------------------------------------------------------------------------
# Entity base
# ---------------------------------------------------------------------------

class Entity:
    """A world-space object with position, velocity, and gravity."""

    def __init__(self, x, y, z):
        self.x   = float(x)
        self.y   = float(y)
        self.z   = float(z)
        self.dx  = 0.0   # horizontal velocity
        self.dz  = 0.0
        self.dy  = 0.0   # vertical velocity (gravity accumulates here)
        self.on_ground = False

    @property
    def position(self):
        return (self.x, self.y, self.z)

    def apply_gravity(self, dt):
        self.dy -= dt * config.GRAVITY
        self.dy  = max(self.dy, -config.TERMINAL_VELOCITY)

    def move(self, dt, world, portal_tex=None):
        """Apply velocity and resolve collisions."""
        nx = self.x + self.dx * dt
        ny = self.y + self.dy * dt
        nz = self.z + self.dz * dt
        new_pos, on_ground = collide(
            (nx, ny, nz), self.height, world, portal_tex)
        self.x, self.y, self.z = new_pos
        self.on_ground = on_ground
        if on_ground:
            self.dy = 0.0

    @property
    def height(self):
        return 1.0   # overridden by Mob


# ---------------------------------------------------------------------------
# Mob
# ---------------------------------------------------------------------------

class Mob(Entity):
    """A wandering mob with simple IDLE/WALK AI.

    State machine:
      IDLE  — stands still for a random duration
      WALK  — walks in a random horizontal direction for a random duration
    On collision with terrain the mob picks a new random direction.
    """

    # AI states
    IDLE = 'idle'
    WALK = 'walk'

    def __init__(self, mob_type, x, y, z):
        super().__init__(x, y, z)
        self.mob_type   = mob_type
        self._def       = config.MOB_TYPES[mob_type]
        self._state     = self.IDLE
        self._state_timer = random.uniform(*self._def['idle_time'])
        self._yaw       = random.uniform(0, 2 * math.pi)   # facing direction
        self._walk_yaw  = self._yaw
        self._alive     = True

    # ------------------------------------------------------------------
    # Properties from mob definition
    # ------------------------------------------------------------------

    @property
    def height(self):
        return self._def['height']

    @property
    def width(self):
        return self._def['width']

    @property
    def speed(self):
        return self._def['speed']

    @property
    def body_colour(self):
        return self._def['body_colour']

    @property
    def leg_colour(self):
        return self._def['leg_colour']

    @property
    def alive(self):
        return self._alive

    # ------------------------------------------------------------------
    # AI update
    # ------------------------------------------------------------------

    def update(self, dt, world, portal_tex=None):
        """Advance AI state, apply gravity, move, resolve collisions."""
        self._state_timer -= dt

        if self._state == self.IDLE:
            self.dx = 0.0
            self.dz = 0.0
            if self._state_timer <= 0:
                self._enter_walk()

        elif self._state == self.WALK:
            self.dx = math.cos(self._walk_yaw) * self.speed
            self.dz = math.sin(self._walk_yaw) * self.speed
            if self._state_timer <= 0:
                self._enter_idle()

        # Gravity
        self.apply_gravity(dt)

        # Remember position before move to detect wall collisions
        old_x, old_z = self.x, self.z
        self.move(dt, world, portal_tex)

        # If we hit a wall while walking, pick a new direction next tick
        if self._state == self.WALK:
            moved_dist = math.sqrt((self.x-old_x)**2 + (self.z-old_z)**2)
            expected   = self.speed * dt * 0.3   # expect at least 30% of intended
            if moved_dist < expected:
                self._enter_idle(short=True)

        # Safety — fell out of world
        if self.y < -20:
            self._alive = False

    def _enter_idle(self, short=False):
        self._state = self.IDLE
        lo, hi = self._def['idle_time']
        self._state_timer = random.uniform(lo * 0.3, hi * 0.5) \
            if short else random.uniform(lo, hi)
        self.dx = 0.0
        self.dz = 0.0

    def _enter_walk(self):
        self._state = self.WALK
        lo, hi = self._def['walk_time']
        self._state_timer = random.uniform(lo, hi)
        self._walk_yaw = random.uniform(0, 2 * math.pi)
        self._yaw = self._walk_yaw


# ---------------------------------------------------------------------------
# MobManager
# ---------------------------------------------------------------------------

class MobManager:
    """Manages the mob lifecycle: spawning, updating, despawning."""

    def __init__(self):
        self.mobs: list[Mob] = []
        self._spawn_timer = 0.0
        self._physics_tick = 0

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, dt, player_pos, world, portal_tex=None,
               top_surface=None):
        """Tick mob AI (rate-limited), handle spawning and despawning."""
        self._spawn_timer  += dt
        self._physics_tick += 1

        # Mob physics every MOB_PHYSICS_RATE frames
        if self._physics_tick >= config.MOB_PHYSICS_RATE:
            self._physics_tick = 0
            physics_dt = dt * config.MOB_PHYSICS_RATE
            for mob in self.mobs:
                mob.update(physics_dt, world, portal_tex)

        # Remove dead or far mobs
        px, _py, pz = player_pos
        despawn_dist = config.MOB_DESPAWN_DIST * config.SECTOR_SIZE
        self.mobs = [
            m for m in self.mobs
            if m.alive and
               math.sqrt((m.x-px)**2 + (m.z-pz)**2) < despawn_dist
        ]

        # Spawn attempt
        if (self._spawn_timer >= config.MOB_SPAWN_INTERVAL and
                len(self.mobs) < config.MOB_MAX):
            self._spawn_timer = 0.0
            self._try_spawn(player_pos, world, top_surface)

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def _try_spawn(self, player_pos, world, top_surface=None):
        """Attempt to spawn a mob on a valid surface near the player.

        Uses top_surface dict (O(1) per column) when available, falls back
        to scanning world dict.  Spawn ring: 6..48 blocks from player.
        """
        px, _py, pz = player_pos
        lo = 6.0                                 # close enough to see
        hi = config.MOB_SPAWN_DIST * config.SECTOR_SIZE

        # Pick several random candidates and use the first valid one
        for _ in range(8):
            angle  = random.uniform(0, 2 * math.pi)
            radius = random.uniform(lo, hi)
            tx     = px + math.cos(angle) * radius
            tz     = pz + math.sin(angle) * radius
            bx, bz = int(round(tx)), int(round(tz))

            # Find surface via top_surface dict (fast) or world scan (fallback)
            if top_surface is not None:
                entry = top_surface.get((bx, bz))
                if entry is None:
                    continue
                surface_y, surface_tex = entry
            else:
                surface_y = surface_tex = None
                for y in range(60, 0, -1):
                    if (bx, y, bz) in world:
                        surface_y   = y
                        surface_tex = world[(bx, y, bz)]
                        break
                if surface_y is None:
                    continue

            # Don't spawn on water or portal
            if surface_tex in (config.WATER, config.MAGIC_WATER,
                                config.PORTAL_TEX):
                continue

            # Identify block name
            surface_name = ''
            for bname in config.BLOCK_IDS:
                tex = getattr(config, bname, None)
                if tex is not None and tuple(tex) == tuple(surface_tex):
                    surface_name = bname
                    break

            candidates = [
                t for t, d in config.MOB_TYPES.items()
                if surface_name in d['spawn_on']
            ]
            if not candidates:
                continue

            mob_type = random.choice(candidates)
            # Spawn 1.2 blocks above surface — close enough for fast settling
            self.mobs.append(Mob(mob_type,
                                 bx + 0.2,
                                 surface_y + 1.2,
                                 bz + 0.2))
            return
