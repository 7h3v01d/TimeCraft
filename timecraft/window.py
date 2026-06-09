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
        self._hotbar_batch = None
        self._hotbar_slot_shapes = []   # background rects + border rects
        self._hotbar_sprites = []       # one Sprite per slot (texture icon)
        self._build_hotbar()
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
        self.model.update_particles(dt)
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

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.exclusive:
            current = self.inventory.index(self.block)
            n = len(self.inventory)
            self.block = self.inventory[(current - int(scroll_y)) % n]

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
        self._build_hotbar()

    def _build_hotbar(self):
        """(Re)build the hotbar batch.  Called on init and every resize."""
        n          = len(self.inventory)
        slot_size  = config.HOTBAR_SLOT_SIZE
        padding    = config.HOTBAR_PADDING
        icon_pad   = config.HOTBAR_ICON_PAD
        icon_size  = slot_size - icon_pad * 2
        total_w    = n * slot_size + (n - 1) * padding
        start_x    = (self.width - total_w) // 2
        base_y     = config.HOTBAR_Y

        # Load the 4×4 atlas and build a TextureGrid for cell lookups.
        # We cache this on self so it isn't reloaded every resize.
        if not hasattr(self, '_atlas_texture'):
            raw = pyglet.image.load(config.TEXTURE_PATH)
            grid = pyglet.image.ImageGrid(raw, rows=4, columns=4)
            self._atlas_grid = pyglet.image.TextureGrid(grid)

        self._hotbar_batch       = pyglet.graphics.Batch()
        self._hotbar_slot_shapes = []
        self._hotbar_sprites     = []

        for i, tex in enumerate(self.inventory):
            sx = start_x + i * (slot_size + padding)
            sy = base_y

            # Slot background — semi-transparent dark square
            bg = pyglet.shapes.Rectangle(
                sx, sy, slot_size, slot_size,
                color=(30, 30, 30, 160),
                batch=self._hotbar_batch,
            )
            self._hotbar_slot_shapes.append(bg)

            # Texture icon — sampled from the atlas TextureGrid
            cell = config.TEXTURE_ATLAS_CELL.get(tuple(tex))
            if cell is not None:
                col, row = cell
                region = self._atlas_grid[row, col]
                sprite = pyglet.sprite.Sprite(
                    region,
                    x=sx + icon_pad,
                    y=sy + icon_pad,
                    batch=self._hotbar_batch,
                )
                sprite.width  = icon_size
                sprite.height = icon_size
                self._hotbar_sprites.append(sprite)
            else:
                self._hotbar_sprites.append(None)

    def draw_hotbar(self):
        """Draw the hotbar batch then overlay a highlight on the selected slot."""
        if self._hotbar_batch is None:
            return

        n         = len(self.inventory)
        slot_size = config.HOTBAR_SLOT_SIZE
        padding   = config.HOTBAR_PADDING
        total_w   = n * slot_size + (n - 1) * padding
        start_x   = (self.width - total_w) // 2
        base_y    = config.HOTBAR_Y

        self._hotbar_batch.draw()

        # Highlight the selected slot with a bright border
        try:
            sel = self.inventory.index(self.block)
        except ValueError:
            sel = 0
        sx = start_x + sel * (slot_size + padding)
        t  = 2   # border thickness
        border_shapes = []
        border_batch  = pyglet.graphics.Batch()
        # Top bar
        border_shapes.append(pyglet.shapes.Rectangle(sx, base_y + slot_size - t, slot_size, t,     color=(255, 255, 255, 230), batch=border_batch))
        # Bottom bar
        border_shapes.append(pyglet.shapes.Rectangle(sx, base_y,                  slot_size, t,     color=(255, 255, 255, 230), batch=border_batch))
        # Left bar
        border_shapes.append(pyglet.shapes.Rectangle(sx, base_y,                  t, slot_size,     color=(255, 255, 255, 230), batch=border_batch))
        # Right bar
        border_shapes.append(pyglet.shapes.Rectangle(sx + slot_size - t, base_y,  t, slot_size,     color=(255, 255, 255, 230), batch=border_batch))
        border_batch.draw()

    def _rebuild_reticle(self):
        cx, cy = self.width // 2, self.height // 2
        arm   = 10   # length of each arm from gap edge
        gap   = 4    # empty space around the centre dot
        thick = 2    # white line thickness

        self.reticle_batch  = pyglet.graphics.Batch()
        self._reticle_lines = []   # keep refs — shapes GC themselves if not held

        arms = [
            (cx - arm - gap, cy, cx - gap,        cy),  # left
            (cx + gap,        cy, cx + arm + gap,  cy),  # right
            (cx, cy - arm - gap, cx, cy - gap      ),    # down
            (cx, cy + gap,       cx, cy + arm + gap),    # up
        ]
        # Dark outline drawn first, white fill drawn on top
        for x1, y1, x2, y2 in arms:
            self._reticle_lines.append(pyglet.shapes.Line(
                x1, y1, x2, y2,
                thickness=thick + 2,
                color=(0, 0, 0, 180),
                batch=self.reticle_batch,
            ))
        for x1, y1, x2, y2 in arms:
            self._reticle_lines.append(pyglet.shapes.Line(
                x1, y1, x2, y2,
                thickness=thick,
                color=(255, 255, 255, 230),
                batch=self.reticle_batch,
            ))

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
        r, g, b = config.sky_colour(self.model.game_time)
        gl.glClearColor(r, g, b, 1.0)
        self.clear()
        self.set_3d()
        # Update frustum planes from this frame's view/projection before any sector ops
        self.model.set_frustum(self.projection @ self.view)
        # Push view/projection matrices and sun brightness into both shaders
        self.model.set_shader_uniforms(self.view, self.projection)
        self.model.batch.draw()
        self.set_2d()
        self.draw_particles()
        self.draw_hotbar()
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

    def draw_particles(self):
        """Project living particles into screen space and draw them as quads.

        We're already in the 2-D orthographic pass (set_2d called beforehand).
        Each particle's world position is multiplied by the last 3-D view and
        projection matrices to get a clip-space coordinate, which we then
        convert to window pixels.  Particles behind the camera are skipped.
        """
        if not self.model.particles:
            return

        width, height = self.get_size()
        half_s = config.PARTICLE_SIZE / 2

        view = self.view
        proj = self.projection

        batch = pyglet.graphics.Batch()
        quads = []   # keep references alive until batch.draw()

        for p in self.model.particles:
            # World → view space
            vp = view @ (p.x, p.y, p.z, 1.0)
            if vp.z >= 0:            # behind camera plane
                continue

            # View → clip space
            cp = proj @ (vp.x, vp.y, vp.z, vp.w)
            if cp.w == 0:
                continue

            # Clip → NDC → window pixels
            sx = (cp.x / cp.w + 1.0) * 0.5 * width
            sy = (cp.y / cp.w + 1.0) * 0.5 * height

            # Fade alpha out over lifetime
            r, g, b, a = p.colour
            faded_a = int(a * (1.0 - p.alpha_fraction))
            colour = (r, g, b, faded_a)

            quads.append(pyglet.shapes.Rectangle(
                sx - half_s, sy - half_s,
                config.PARTICLE_SIZE, config.PARTICLE_SIZE,
                color=colour, batch=batch,
            ))

        batch.draw()

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
