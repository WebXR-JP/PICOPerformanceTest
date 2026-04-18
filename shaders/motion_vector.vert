#version 450
#extension GL_EXT_multiview : enable

layout(location=0) out vec2 vUV;

void main() {
    vUV = vec2(float(gl_VertexIndex & 1) * 2.0, float(gl_VertexIndex >> 1) * 2.0);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
