# sky.py
#
# Renders sun, moon, stars, and clouds in the 3-D pass before the world
# batch.  All sky objects are drawn with depth writes disabled so they
# always sit behind terrain without affecting the depth buffer.

import math
import random

import pyglet
import pyglet.graphics.shader
from pyglet.gl import gl
from pyglet.math import Vec3, Mat4

import config


# ---------------------------------------------------------------------------
# Minimal sky shader — position only, outputs a flat colour + alpha
# ---------------------------------------------------------------------------

_SKY_VERT = """
#version 330 core
in vec3 position;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * vec4(position, 1.0);
}
"""

_SKY_FRAG = """
#version 330 core
out vec4 out_color;
uniform vec4 sky_obj_colour;
void main() {
    out_color = sky_obj_colour;
}
"""


def _make_sky_shader():
    return pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(_SKY_VERT, 'vertex'),
        pyglet.graphics.shader.Shader(_SKY_FRAG, 'fragment'),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quad_verts(cx, cy, cz, hw, hh, axis='y'):
    """Return 6 vertex positions (2 triangles) for a flat quad.

    axis='y'  → horizontal quad (clouds, flat on XZ plane)
    axis='billboard' is handled separately via camera-facing logic
    """
    if axis == 'y':
        return [
            cx-hw, cy, cz-hh,
            cx+hw, cy, cz-hh,
            cx+hw, cy, cz+hh,
            cx-hw, cy, cz-hh,
            cx+hw, cy, cz+hh,
            cx-hw, cy, cz+hh,
        ]
    return []


def _circle_verts(cx, cy, cz, radius, nx, ny, nz, segments=16):
    """Return vertices for a disc (triangle fan) facing normal (nx,ny,nz).

    Returns flat list of (x,y,z) per vertex — one triangle per segment.
    """
    # Build two perpendicular axes in the plane of the disc
    n = Vec3(nx, ny, nz)
    up = Vec3(0, 1, 0)
    if abs(nx) < 0.9:
        right = up.cross(n).normalize()
    else:
        right = Vec3(0, 0, 1).cross(n).normalize()
    up2 = n.cross(right).normalize()

    verts = []
    for i in range(segments):
        a0 = (i / segments) * 2 * math.pi
        a1 = ((i + 1) / segments) * 2 * math.pi
        x0 = cx + (right.x * math.cos(a0) + up2.x * math.sin(a0)) * radius
        y0 = cy + (right.y * math.cos(a0) + up2.y * math.sin(a0)) * radius
        z0 = cz + (right.z * math.cos(a0) + up2.z * math.sin(a0)) * radius
        x1 = cx + (right.x * math.cos(a1) + up2.x * math.sin(a1)) * radius
        y1 = cy + (right.y * math.cos(a1) + up2.y * math.sin(a1)) * radius
        z1 = cz + (right.z * math.cos(a1) + up2.z * math.sin(a1)) * radius
        # Triangle: centre, v0, v1
        verts.extend([cx, cy, cz, x0, y0, z0, x1, y1, z1])
    return verts


# ---------------------------------------------------------------------------
# SkyRenderer
# ---------------------------------------------------------------------------

class SkyRenderer:
    """Draws sun, moon, stars, and clouds each frame.

    Call build() once after GL is initialised, then draw() each frame
    inside the 3-D pass before the world batch.
    """

    def __init__(self):
        self._shader = None
        self._star_positions = []   # list of (x,y,z) unit vectors
        self._cloud_offsets = []    # list of (ox, oz, w, d) relative offsets

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def build(self):
        """Compile shader and pre-generate star/cloud data."""
        self._shader = _make_sky_shader()
        self._build_stars()
        self._build_clouds()

    def _build_stars(self):
        rng = random.Random(config.STAR_SEED)
        self._star_positions = []
        while len(self._star_positions) < config.STAR_COUNT:
            cos_theta = rng.uniform(0.05, 1.0)
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            psi = rng.uniform(0, 2 * math.pi)
            self._star_positions.append((
                sin_theta * math.cos(psi),
                cos_theta,
                sin_theta * math.sin(psi),
            ))

    def _build_clouds(self):
        """Pre-generate cloud shapes as clusters of overlapping soft puffs.

        Each cloud is a list of (ox, oz, oy_offset, radius) puffs scattered
        around a centre point.  The overlap between puffs with alpha blending
        creates the illusion of a fluffy, volumetric mass.
        """
        rng = random.Random(config.CLOUD_SEED)
        s   = config.CLOUD_SPREAD
        self._cloud_offsets = []   # list of lists of puffs per cloud

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
                py_off = rng.uniform(0.0, 2.0)       # height variation
                r      = rng.uniform(scale * 0.4, scale * 0.95)
                puffs.append((px, pz, py_off, r))
            self._cloud_offsets.append(puffs)

    # ------------------------------------------------------------------
    # Per-frame draw
    # ------------------------------------------------------------------

    def draw(self, game_time, player_pos, view, proj,
             fog_density=0.0, weather_type='clear'):
        """Draw all sky objects.  Must be called in the 3-D pass."""
        if self._shader is None:
            return

        px, py, pz = player_pos
        R = config.SKY_SPHERE_RADIUS
        angle = (game_time / config.DAY_LENGTH) * 2.0 * math.pi
        brightness = config.sun_brightness(game_time)

        # Disable depth writes — sky is always behind terrain
        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self._shader.use()
        self._shader['view']       = list(view)
        self._shader['projection'] = list(proj)

        # Sun position: rises in +Z, sets in -Z, zenith at noon
        sun_x = px
        sun_y = py + math.sin(angle) * R
        sun_z = pz - math.cos(angle) * R

        # Moon is opposite
        moon_x = px
        moon_y = py + math.sin(angle + math.pi) * R
        moon_z = pz - math.cos(angle + math.pi) * R

        # Direction vectors from player to sun/moon (for disc orientation)
        sdx, sdy, sdz = sun_x-px, sun_y-py, sun_z-pz
        sl = math.sqrt(sdx*sdx + sdy*sdy + sdz*sdz)
        if sl > 0:
            sdx, sdy, sdz = sdx/sl, sdy/sl, sdz/sl

        mdx, mdy, mdz = moon_x-px, moon_y-py, moon_z-pz
        ml = math.sqrt(mdx*mdx + mdy*mdy + mdz*mdz)
        if ml > 0:
            mdx, mdy, mdz = mdx/ml, mdy/ml, mdz/ml

        # --- Sun (only when above horizon) ---
        if sun_y > py - 2:
            sun_alpha = min(255, int(255 * min(1.0, (sun_y - py + 2) / 6)))
            # Halo (outer glow)
            halo_verts = _circle_verts(sun_x, sun_y, sun_z,
                                       config.SUN_HALO_SIZE,
                                       sdx, sdy, sdz, segments=20)
            r, g, b, _ = config.SUN_COLOUR
            self._draw_verts(halo_verts, (r, g, b, sun_alpha // 3))
            # Core disc
            sun_verts = _circle_verts(sun_x, sun_y, sun_z,
                                      config.SUN_SIZE,
                                      sdx, sdy, sdz, segments=20)
            self._draw_verts(sun_verts, (r, g, b, sun_alpha))

        # --- Moon (only when above horizon) ---
        if moon_y > py - 2:
            moon_alpha = min(220, int(220 * min(1.0, (moon_y - py + 2) / 6)))
            # Dim moon during day
            moon_alpha = int(moon_alpha * max(0.1, 1.0 - brightness * 1.5))
            if moon_alpha > 5:
                moon_verts = _circle_verts(moon_x, moon_y, moon_z,
                                           config.MOON_SIZE,
                                           mdx, mdy, mdz, segments=20)
                r, g, b, _ = config.MOON_COLOUR
                self._draw_verts(moon_verts, (r, g, b, moon_alpha))

        # --- Stars ---
        if brightness < config.STAR_FADE_START:
            t = (config.STAR_FADE_START - brightness) / \
                (config.STAR_FADE_START - config.STAR_FADE_END)
            star_alpha = int(200 * min(1.0, t))
            if star_alpha > 5:
                self._draw_stars(px, py, pz, R, star_alpha)

        # --- Clouds ---
        cloud_alpha = config.CLOUD_ALPHA_BASE
        # Fade clouds in rain/snow weather
        if weather_type == 'clear':
            cloud_alpha = int(cloud_alpha * (1.0 - fog_density * 0.3))
        else:
            cloud_alpha = int(cloud_alpha * (0.6 + fog_density * 0.4))
        # Fade near dawn/dusk to match sky brightness
        cloud_alpha = int(cloud_alpha * (0.3 + 0.7 * brightness))
        if cloud_alpha > 8:
            self._draw_clouds(game_time, px, py, pz, cloud_alpha)

        self._shader.stop()
        gl.glDepthMask(gl.GL_TRUE)

    # ------------------------------------------------------------------
    # Internal draw helpers
    # ------------------------------------------------------------------

    def _draw_verts(self, verts, colour):
        """Draw a flat list of triangle verts with a solid colour."""
        if not verts:
            return
        n = len(verts) // 3
        r, g, b, a = colour
        self._shader['sky_obj_colour'] = [r/255, g/255, b/255, a/255]
        vl = self._shader.vertex_list(n, gl.GL_TRIANGLES,
                                      position=('f', verts))
        vl.draw(gl.GL_TRIANGLES)
        vl.delete()

    def _draw_stars(self, px, py, pz, R, alpha):
        """Draw all stars as tiny quads on the sky sphere."""
        s = config.STAR_SIZE
        all_verts = []
        for (ux, uy, uz) in self._star_positions:
            cx = px + ux * R
            cy = py + uy * R
            cz = pz + uz * R
            # Tiny billboard quad — just a flat XZ aligned square
            # (small enough that orientation barely matters)
            all_verts.extend([
                cx-s, cy-s, cz,
                cx+s, cy-s, cz,
                cx+s, cy+s, cz,
                cx-s, cy-s, cz,
                cx+s, cy+s, cz,
                cx-s, cy+s, cz,
            ])
        # Vary star brightness slightly with a deterministic flicker
        self._draw_verts(all_verts, (255, 255, 240, alpha))

    def _draw_clouds(self, game_time, px, py, pz, alpha):
        """Draw cloud clusters as overlapping soft horizontal octagons.

        Each cloud is a group of puffs at slightly different heights.
        Individual puffs are semi-transparent so they stack convincingly
        at overlap points, creating a fluffy volumetric look.
        """
        base_y    = py + config.CLOUD_HEIGHT
        drift_x   = (game_time * config.CLOUD_SPEED) % (config.CLOUD_SPREAD * 2)
        r_colour, g_colour, b_colour = config.CLOUD_COLOUR
        # Per-puff alpha: low enough that overlap builds naturally
        puff_alpha = max(4, int(alpha * 0.55))
        segments   = 8    # octagon — cheap and soft-looking at distance

        all_verts = []

        for cloud_puffs in self._cloud_offsets:
            for (ox, oz, oy_off, radius) in cloud_puffs:
                # Apply drift and wrap
                cx = px + ((ox + drift_x) % (config.CLOUD_SPREAD * 2)) \
                         - config.CLOUD_SPREAD
                cy = base_y + oy_off
                cz = pz + oz

                # Triangle fan octagon (flat horizontal disc)
                for i in range(segments):
                    a0 = (i / segments) * 2 * math.pi
                    a1 = ((i + 1) / segments) * 2 * math.pi
                    x0 = cx + math.cos(a0) * radius
                    z0 = cz + math.sin(a0) * radius
                    x1 = cx + math.cos(a1) * radius
                    z1 = cz + math.sin(a1) * radius
                    all_verts.extend([cx, cy, cz, x0, cy, z0, x1, cy, z1])

        if all_verts:
            self._draw_verts(all_verts,
                             (r_colour, g_colour, b_colour, puff_alpha))
