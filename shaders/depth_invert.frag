#version 450
#extension GL_EXT_multiview : enable

layout(location=0) in vec2 vUV;
layout(set=0, binding=0) uniform sampler2DArray depthTex;

void main() {
    // Runtime supports reverse-Z; pass through.
    float d = texture(depthTex, vec3(vUV, float(gl_ViewIndex))).r;
    gl_FragDepth = d;
}
