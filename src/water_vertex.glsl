#version 120
attribute vec3 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
uniform mat4 modelview;
uniform mat4 projection;
uniform float time;
void main() {
    vec3 pos = position;
    pos.y += sin(time + position.x + position.z) * 0.1; // Wave effect
    gl_Position = projection * modelview * vec4(pos, 1.0);
    v_texcoord = texcoord;
}