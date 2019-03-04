---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec3  v_normal;
attribute vec4  v_color;
attribute vec2  v_tc0;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform float Tr;

varying vec4 frag_color;
varying vec2 uv_vec;
varying vec4 normal_vec;
varying vec4 vertex_pos;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos,1.0);
    frag_color = v_color;
    uv_vec = v_tc0;
    vertex_pos = pos;
    normal_vec = vec4(v_normal,0.0);
    gl_Position = projection_mat * pos;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec4 frag_color;
varying vec2 uv_vec;

uniform mat4 normal_mat;
uniform vec3 Kd; // diffuse color
uniform vec3 Ka; // color
uniform vec3 Ks; // specular color
uniform float Tr; // transparency
uniform float Ns; // shininess


uniform sampler2D tex;

void main (void){
    vec4 v_normal = normalize( normal_mat * normal_vec );
    vec4 v_light = normalize( vec4(0,0,0,1) - vertex_pos );

    vec3 Ia = Ka;
    vec3 Id = Kd * max(dot(v_light, v_normal), 0.0);
    vec3 Is = Ks * pow(max(dot(v_light, v_normal), 0.0), Ns);

    vec4 tex_color = texture2D(tex, uv_vec);
    gl_FragColor = vec4(Ia + Id + Is, Tr);
    gl_FragColor = gl_FragColor * tex_color;
}
