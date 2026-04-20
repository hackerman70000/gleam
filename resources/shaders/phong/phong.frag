#version 330 core

// Static Phong coefficients from the project specification.
// 76 / 255 = 0.29803922, 25 / 255 = 0.09803922.
const vec3 Ka = vec3(0.29803922);
const vec3 Ia = vec3(0.09803922);
const vec3 Ks = vec3(1.0);
const vec3 Id = vec3(1.0);
const vec3 Is = vec3(1.0);

uniform vec3  u_kd;
uniform float u_n;
uniform vec3  u_light_pos;
uniform vec3  u_cam_pos;

in  vec3 v_world_pos;
in  vec3 v_world_norm;
out vec4 f_color;

void main() {
    vec3 N = normalize(v_world_norm);
    vec3 L = normalize(u_light_pos - v_world_pos);
    vec3 V = normalize(u_cam_pos   - v_world_pos);
    vec3 R = reflect(-L, N);

    float NdotL = max(dot(N, L), 0.0);
    float RdotV = max(dot(R, V), 0.0);

    vec3 ambient  = Ka * Ia;
    vec3 diffuse  = u_kd * NdotL * Id;
    vec3 specular = (NdotL > 0.0) ? Ks * pow(RdotV, u_n) * Is : vec3(0.0);

    f_color = vec4(clamp(ambient + diffuse + specular, 0.0, 1.0), 1.0);
}
