#version 120
varying vec2 v_texcoord;
uniform sampler2D texture;
uniform float time;
uniform int is_magic_water;
void main() {
    vec4 color = texture2D(texture, v_texcoord);
    float shimmer = sin(time * 2.0 + v_texcoord.x * 10.0) * 0.1;
    color.rgb += vec3(0.1, 0.1, 0.2) * shimmer; // Blue shimmer
    if (is_magic_water == 1) {
        color.rgb += vec3(0.2, 0.2, 0.4) * (0.5 + 0.5 * sin(time)); // Glowing pulse
    }
    gl_FragColor = color;
}