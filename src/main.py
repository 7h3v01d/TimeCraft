# main.py

import pyglet
from pyglet.gl import gl
from window import Window

def setup_fog():
    """ Configure the OpenGL fog properties. """
    gl.glEnable(gl.GL_FOG)
    gl.glFogfv(gl.GL_FOG_COLOR, (gl.GLfloat * 4)(0.5, 0.69, 1.0, 1.0))
    gl.glHint(gl.GL_FOG_HINT, gl.GL_DONT_CARE)
    gl.glFogi(gl.GL_FOG_MODE, gl.GL_LINEAR)
    gl.glFogf(gl.GL_FOG_START, 40.0)
    gl.glFogf(gl.GL_FOG_END, 60.0)

def setup():
    """ Basic OpenGL configuration. """
    gl.glClearColor(0.5, 0.69, 1.0, 1)
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    setup_fog()

def main():
    """ Main entry point for the application. """
    window = Window(width=1280, height=720, caption='Time Craft', resizable=True)
    window.set_exclusive_mouse(True)
    setup()
    pyglet.app.run()

if __name__ == '__main__':
    main()