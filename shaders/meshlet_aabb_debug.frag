#version 450

layout(location = 0) in vec4 inColor;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outPrevDepth;
layout(location = 2) out vec4 outMotionVec;

void main() {
    outColor = inColor;
    outPrevDepth = 1.0;
    outMotionVec = vec4(0.0);
}
