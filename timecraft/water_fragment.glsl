#version 330 core
in vec2 v_texcoord;
out vec4 out_color;
uniform sampler2D our_texture;
uniform float time;
uniform int is_magic_water;
void main() {
    vec4 color = texture(our_texture, v_texcoord);
    float shimmer = sin(time * 2.0 + v_texcoord.x * 10.0) * 0.1;
    color.rgb += vec3(0.1, 0.1, 0.2) * shimmer;
    if (is_magic_water == 1) {
        color.rgb += vec3(0.2, 0.2, 0.4) * (0.5 + 0.5 * sin(time));
    }
    out_color = color;
}
