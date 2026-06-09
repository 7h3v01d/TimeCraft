# main.py

import pyglet
from pyglet.gl import gl
from window import Window


def setup():
    """ Basic OpenGL configuration. """
    gl.glClearColor(0.5, 0.69, 1.0, 1)
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)


def main():
    """ Main entry point for the application. """

    # Open a plain window first just to show the loading screen.
    # We need a GL context before we can draw anything, but we don't want
    # to start world-gen yet — so we use a temporary pyglet.window.Window.
    splash = pyglet.window.Window(width=1280, height=720,
                                  caption='Time Craft - Loading...')

    loading_label = pyglet.text.Label(
        'Loading world, please wait...',
        font_name='Arial', font_size=28,
        x=640, y=380, anchor_x='center', anchor_y='center',
        color=(255, 255, 255, 255)
    )
    hint_label = pyglet.text.Label(
        'Generating terrain...',
        font_name='Arial', font_size=16,
        x=640, y=330, anchor_x='center', anchor_y='center',
        color=(180, 210, 255, 210)
    )

    gl.glClearColor(0.08, 0.08, 0.18, 1)
    splash.clear()
    loading_label.draw()
    hint_label.draw()
    splash.flip()

    # Pump the OS event queue once so the window actually paints on Windows
    pyglet.clock.tick()
    splash.dispatch_events()

    # Now do the heavy work: world gen lives inside Window.__init__ → Model()
    window = Window(width=1280, height=720, caption='Time Craft', resizable=True)
    setup()

    # Close the splash and hand control to the game window
    splash.close()
    window.set_exclusive_mouse(True)

    pyglet.app.run()


if __name__ == '__main__':
    main()
