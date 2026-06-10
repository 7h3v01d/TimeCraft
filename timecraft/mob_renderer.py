# mob_renderer.py
#
# Renders mobs as textured billboard sprites using mob_atlas.png.
# Each mob type occupies a horizontal strip of the atlas.
# Walk animation oscillates the leg quads using game_time.

import math
import os

import pyglet
import pyglet.graphics.shader
import pyglet.image
from pyglet.gl import gl


# ---------------------------------------------------------------------------
# Shader — textured billboard with per-vertex UV
# ---------------------------------------------------------------------------

_MOB_VERT = """
#version 330 core
in vec3  position;
in vec2  tex_coord;
out vec2 v_uv;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    v_uv = tex_coord;
}
"""

_MOB_FRAG = """
#version 330 core
in  vec2 v_uv;
out vec4 out_color;
uniform sampler2D mob_tex;
uniform float     brightness;
void main() {
    vec4 c = texture(mob_tex, v_uv);
    if (c.a < 0.05) discard;
    out_color = vec4(c.rgb * brightness, c.a);
}
"""

# UV atlas layout: each mob occupies a horizontal slice
# u0, u1 for each mob type (atlas is 2 mobs wide)
MOB_UV = {
    'chicken': (0.0, 0.5),   # left half
    'sheep':   (0.5, 1.0),   # right half
}

# Leg region: bottom 30% of sprite height
LEG_V_TOP = 0.30   # v above which = body, below = legs

# Walk animation
WALK_FREQ       = 3.5    # cycles per second per unit of speed
WALK_AMPLITUDE  = 0.10   # world units leg offset


# ---------------------------------------------------------------------------
# MobRenderer
# ---------------------------------------------------------------------------

class MobRenderer:
    """Loads mob_atlas.png and renders mob sprites each frame."""

    def __init__(self):
        self._shader  = None
        self._texture = None

    def build(self, atlas_path):
        """Compile shader and load atlas texture.  Call once after GL init."""
        self._shader = pyglet.graphics.shader.ShaderProgram(
            pyglet.graphics.shader.Shader(_MOB_VERT, 'vertex'),
            pyglet.graphics.shader.Shader(_MOB_FRAG, 'fragment'),
        )
        img = pyglet.image.load(atlas_path)
        self._texture = img.get_texture()
        # Nearest-neighbour filtering — keeps pixels crisp
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture.id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_NEAREST)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def draw(self, mobs, view, projection, game_time, brightness):
        """Draw all mobs.  Call in the 3-D pass after world batch."""
        if not mobs or self._shader is None or self._texture is None:
            return

        # Camera right vector from view matrix column 0 (column-major)
        vm = list(view)
        cam_rx = vm[0]
        cam_rz = vm[2]

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_TRUE)

        # Bind mob atlas
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture.id)

        self._shader.use()
        self._shader['view']       = vm
        self._shader['projection'] = list(projection)
        self._shader['mob_tex']    = 0
        self._shader['brightness'] = float(brightness)

        for mob in mobs:
            self._draw_mob(mob, cam_rx, cam_rz, game_time)

        self._shader.stop()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    # ------------------------------------------------------------------
    # Per-mob draw
    # ------------------------------------------------------------------

    def _draw_mob(self, mob, cam_rx, cam_rz, game_time):
        mob_type = mob.mob_type
        u0, u1   = MOB_UV.get(mob_type, (0.0, 1.0))

        w   = mob.width
        h   = mob.height
        hw  = w / 2.0

        # Collision pad offset — mob.y is 0.25 inside block surface
        PAD = 0.25
        mx  = mob.x
        my  = mob.y + PAD
        mz  = mob.z

        # Split height into body and leg regions
        leg_h  = h * LEG_V_TOP          # world units of leg portion
        body_h = h - leg_h              # world units of body portion
        body_y = my + leg_h             # y where body starts

        walking = (mob._state == 'walk')

        # Walk animation: left and right legs swing in opposite phase
        if walking:
            freq  = mob.speed * WALK_FREQ
            phase = game_time * freq
            l_off = math.sin(phase)  * WALK_AMPLITUDE
            r_off = -math.sin(phase) * WALK_AMPLITUDE
        else:
            l_off = r_off = 0.0

        # --- Body quad (upper sprite region, static) ---
        bverts, buvs = self._quad(
            mx, body_y, mz, hw, body_h, cam_rx, cam_rz,
            u0, u1, LEG_V_TOP, 1.0,
        )
        self._emit(bverts, buvs)

        # --- Left leg quad (lower-left sprite, animated) ---
        lhw = hw * 0.5
        lx  = mx - cam_rx * hw * 0.25
        lz  = mz - cam_rz * hw * 0.25
        lverts, luvs = self._quad(
            lx, my + l_off, lz, lhw, leg_h, cam_rx, cam_rz,
            u0, u0 + (u1-u0)*0.5, 0.0, LEG_V_TOP,
        )
        self._emit(lverts, luvs)

        # --- Right leg quad (lower-right sprite, animated) ---
        rx_ = mx + cam_rx * hw * 0.25
        rz  = mz + cam_rz * hw * 0.25
        rverts, ruvs = self._quad(
            rx_, my + r_off, rz, lhw, leg_h, cam_rx, cam_rz,
            u0 + (u1-u0)*0.5, u1, 0.0, LEG_V_TOP,
        )
        self._emit(rverts, ruvs)

    def _quad(self, cx, cy, cz, hw, h, rx, rz,
              u0, u1, v0, v1):
        """Return (positions, uvs) for a billboard quad (2 triangles)."""
        # 4 corners: BL, BR, TR, TL
        # OpenGL v=0 at bottom of texture → v0=0 at feet, v1=1 at head
        bl = (cx - rx*hw, cy,     cz - rz*hw)
        br = (cx + rx*hw, cy,     cz + rz*hw)
        tr = (cx + rx*hw, cy + h, cz + rz*hw)
        tl = (cx - rx*hw, cy + h, cz - rz*hw)

        # Atlas v: image y=0 is top, OpenGL v=0 is bottom
        # Image top = head (v=1 in OpenGL), image bottom = feet (v=0)
        verts = [
            *bl, *br, *tr,
            *bl, *tr, *tl,
        ]
        uvs = [
            u0, v0,  u1, v0,  u1, v1,
            u0, v0,  u1, v1,  u0, v1,
        ]
        return verts, uvs

    def _emit(self, verts, uvs):
        n = len(verts) // 3
        vl = self._shader.vertex_list(
            n, gl.GL_TRIANGLES,
            position=('f', verts),
            tex_coord=('f', uvs),
        )
        vl.draw(gl.GL_TRIANGLES)
        vl.delete()
