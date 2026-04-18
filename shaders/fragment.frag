#version 450

layout(location = 0) in vec3 vColor;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outPrevDepth;
layout(location = 2) out vec4 outMotionVec;

void main() {
    outColor = vec4(vColor, 1.0);
    outPrevDepth = gl_FragCoord.z;
    outMotionVec = vec4(0.0);
}
