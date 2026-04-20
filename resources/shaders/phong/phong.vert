#version 330 core

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_mat;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_world_pos;
out vec3 v_world_norm;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_world_norm = normalize(u_normal_mat * in_normal);
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
