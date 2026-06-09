# window.py

import math
import pyglet
from pyglet.gl import gl
from pyglet.window import key, mouse
from pyglet.math import Mat4, Vec3
import config
from model import Model
from util import sectorize, cube_vertices, normalize


class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.exclusive = False
        self.flying = False
        self.jumping = False
        self.jumped = False
        self.crouch = False
        self.sprinting = False
        self.fov_offset = 0
        self.collision_types = {"top": False, "bottom": False, "right": False, "left": False}
        self.strafe = [0, 0]
        self.position = (64, 50, 64)  # overwritten after model loads
        self.rotation = (0, 0)
        self.sector = None
        self.reticle_batch = None
        self.dy = 0
        self.fps = 0.0
        self.inventory = [
            config.BRICK, config.GRASS, config.SAND, config.STONE,
            config.WOOD, config.LEAF, config.CRYSTAL,
            config.DIRT, config.PLANKS, config.GRAVEL, config.SNOW, config.GLASS,
        ]
        self.block = self.inventory[0]
        self.num_keys = [
            key._1, key._2, key._3, key._4, key._5,
            key._6, key._7, key._8, key._9, key._0]
        self.model = Model()
        self.position = self.model.get_spawn_point()
        self.label = pyglet.text.Label('', font_name='Arial', font_size=18,
                                       x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
                                       color=(0, 0, 0, 255))
        self._status_message = ''
        self._status_timer = 0.0
        pyglet.clock.schedule_interval(self.update, 1.0 / config.TICKS_PER_SEC)

    def set_exclusive_mouse(self, exclusive):
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self):
        x, y = self.rotation
        m = math.cos(math.radians(y))
        dy = math.sin(math.radians(y))
        dx = math.cos(math.radians(x - 90)) * m
        dz = math.sin(math.radians(x - 90)) * m
        return (dx, dy, dz)

    def get_motion_vector(self):
        if any(self.strafe):
            x, y = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            y_angle = math.radians(y)
            x_angle = math.radians(x + strafe)
            if self.flying:
                m = math.cos(y_angle)
                dy = math.sin(y_angle)
                if self.strafe[1]:
                    dy = 0.0
                    m = 1
                if self.strafe[0] > 0:
                    dy *= -1
                dx = math.cos(x_angle) * m
                dz = math.sin(x_angle) * m
            else:
                dy = 0.0
                dx = math.cos(x_angle)
                dz = math.sin(x_angle)
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        return (dx, dy, dz)

    def update(self, dt):
        if dt > 0:
            self.fps = 1.0 / dt
        if self._status_timer > 0:
            self._status_timer -= dt
        self.model.process_queue()
        sector = sectorize(self.position, config.SECTOR_SIZE)
        if sector != self.sector:
            self.model.change_sectors(self.sector, sector)
            if self.sector is None:
                self.model.process_entire_queue()
            self.sector = sector
        m = 8
        dt = min(dt, 0.2)
        for _ in range(m):
            self._update(dt / m)

    def _update(self, dt):
        if self.flying:
            speed = config.FLYING_SPEED
        elif self.sprinting:
            speed = config.SPRINT_SPEED
        elif self.crouch:
            speed = config.CROUCH_SPEED
        else:
            speed = config.WALKING_SPEED

        if self.jumping and self.collision_types["top"]:
            self.dy = config.JUMP_SPEED
            self.jumped = True
        elif self.collision_types["top"]:
            self.jumped = False

        if self.jumped:
            speed += 0.7

        d = dt * speed
        dx, dy, dz = self.get_motion_vector()
        dx, dy, dz = dx * d, dy * d, dz * d

        if not self.flying:
            self.dy -= dt * config.GRAVITY
            self.dy = max(self.dy, -config.TERMINAL_VELOCITY)
            dy += self.dy * dt

        old_pos = self.position
        x, y, z = old_pos
        x, y, z = self.collide((x + dx, y + dy, z + dz), config.PLAYER_HEIGHT)
        self.position = (x, y, z)

        # Safety net: if fallen out of world, respawn
        if self.position[1] < -20:
            self.position = self.model.get_spawn_point()
            self.dy = 0

        if old_pos[0] - self.position[0] == 0 and old_pos[2] - self.position[2] == 0:
            if self.sprinting:
                self.sprinting = False
                self.fov_offset -= config.SPRINT_FOV

    def collide(self, position, height):
        pad = 0.25
        p = list(position)
        np = normalize(position)
        self.collision_types = {"top": False, "bottom": False, "right": False, "left": False}
        for face in config.FACES:
            for i in range(3):
                if not face[i]:
                    continue
                d = (p[i] - np[i]) * face[i]
                if d < pad:
                    continue
                for dy in range(int(height)):
                    op = list(np)
                    op[1] -= dy
                    op[i] += face[i]
                    if tuple(op) not in self.model.world:
                        continue
                    p[i] -= (d - pad) * face[i]
                    if face == (0, -1, 0):
                        self.collision_types["top"] = True
                        self.dy = 0
                    if face == (0, 1, 0):
                        self.collision_types["bottom"] = True
                        self.dy = 0
                    break
        return tuple(p)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.exclusive:
            vector = self.get_sight_vector()
            block, previous = self.model.hit_test(self.position, vector)
            if (button == mouse.RIGHT) or ((button == mouse.LEFT) and (modifiers & key.MOD_CTRL)):
                if previous:
                    self.model.add_block(previous, self.block, hit_vector=vector)
            elif button == pyglet.window.mouse.LEFT and block:
                texture = self.model.world[block]
                if texture != config.STONE:
                    self.model.remove_block(block, hit_vector=vector)
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.exclusive:
            m = 0.15
            x, y = self.rotation
            x, y = x + dx * m, y + dy * m
            y = max(-90, min(90, y))
            self.rotation = (x, y)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.W: self.strafe[0] -= 1
        elif symbol == key.S: self.strafe[0] += 1
        elif symbol == key.A: self.strafe[1] -= 1
        elif symbol == key.D: self.strafe[1] += 1
        elif symbol == key.C: self.fov_offset -= 60.0
        elif symbol == key.SPACE: self.jumping = True
        elif symbol == key.ESCAPE: self.set_exclusive_mouse(False)
        elif symbol == key.LSHIFT:
            self.crouch = True
            if self.sprinting:
                self.fov_offset -= config.SPRINT_FOV
                self.sprinting = False
        elif symbol == key.R:
            if not self.crouch and not self.sprinting:
                self.fov_offset += config.SPRINT_FOV
                self.sprinting = True
        elif symbol == key.TAB: self.flying = not self.flying
        elif symbol == key.F5: self._save_world()
        elif symbol == key.F6: self._new_world()
        elif symbol in self.num_keys:
            index = (symbol - self.num_keys[0]) % len(self.inventory)
            self.block = self.inventory[index]

    def on_key_release(self, symbol, modifiers):
        if symbol == key.W: self.strafe[0] += 1
        elif symbol == key.S: self.strafe[0] -= 1
        elif symbol == key.A: self.strafe[1] += 1
        elif symbol == key.D: self.strafe[1] += 1
        elif symbol == key.SPACE: self.jumping = False
        elif symbol == key.LSHIFT: self.crouch = False
        elif symbol == key.C: self.fov_offset += 60.0

    def on_resize(self, width, height):
        self.label.y = height - 10
        self._rebuild_reticle()

    def _rebuild_reticle(self):
        x, y = self.width // 2, self.height // 2
        n = 10
        self.reticle_batch = pyglet.graphics.Batch()
        pyglet.shapes.Line(x - n, y, x + n, y, thickness=1, color=(0, 0, 0, 255), batch=self.reticle_batch)
        pyglet.shapes.Line(x, y - n, x, y + n, thickness=1, color=(0, 0, 0, 255), batch=self.reticle_batch)

    def set_2d(self):
        width, height = self.get_size()
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.projection = Mat4.orthogonal_projection(0, max(1, width), 0, max(1, height), -1, 1)
        self.view = Mat4()

    def set_3d(self):
        width, height = self.get_size()
        gl.glEnable(gl.GL_DEPTH_TEST)
        fov = config.PLAYER_FOV + self.fov_offset
        aspect = width / float(height)
        self.projection = Mat4.perspective_projection(aspect, z_near=0.1, z_far=60.0, fov=fov)

        rx, ry = self.rotation
        px, py, pz = self.position
        eye_y = py + (0.2 if self.crouch else 0)

        # Build view matrix from yaw (rx) and pitch (ry)
        cos_rx = math.cos(math.radians(rx))
        sin_rx = math.sin(math.radians(rx))
        cos_ry = math.cos(math.radians(ry))
        sin_ry = math.sin(math.radians(ry))
        look_x = sin_rx * cos_ry
        look_y = sin_ry
        look_z = -cos_rx * cos_ry
        eye = Vec3(px, eye_y, pz)
        target = Vec3(px + look_x, eye_y + look_y, pz + look_z)
        up = Vec3(0, 1, 0)
        self.view = Mat4.look_at(eye, target, up)

    def on_draw(self):
        self.clear()
        self.set_3d()
        # Push view/projection matrices into both shaders before drawing
        self.model.set_shader_uniforms(self.view, self.projection)
        self.model.batch.draw()
        self.set_2d()
        self.draw_label()
        self.draw_reticle()

    def _save_world(self):
        count = self.model.save_world()
        self._status_message = f'World saved ({count} blocks)'
        self._status_timer = 3.0

    def _new_world(self):
        self.model.delete_save()
        self._status_message = 'Restart to generate new world'
        self._status_timer = 3.0

    def draw_label(self):
        x, y, z = self.position
        self.label.text = '%02d (%.2f, %.2f, %.2f) %d / %d' % (
            self.fps, x, y, z,
            len(self.model._shown), len(self.model.world))
        self.label.draw()
        if self._status_timer > 0:
            if not hasattr(self, '_status_label'):
                self._status_label = pyglet.text.Label(
                    '', font_name='Arial', font_size=16,
                    x=self.width // 2, y=40,
                    anchor_x='center', anchor_y='center',
                    color=(255, 255, 100, 255))
            self._status_label.text = self._status_message
            self._status_label.x = self.width // 2
            self._status_label.draw()

    def draw_reticle(self):
        if self.reticle_batch is None:
            self._rebuild_reticle()
        self.reticle_batch.draw()
