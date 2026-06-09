#version 330 core
in vec3 position;
in vec2 tex_coords;
out vec2 v_texcoord;
uniform mat4 view;
uniform mat4 projection;
uniform float time;
void main() {
    vec3 pos = position;
    pos.y += sin(time + position.x + position.z) * 0.1;
    gl_Position = projection * view * vec4(pos, 1.0);
    v_texcoord = tex_coords;
}
