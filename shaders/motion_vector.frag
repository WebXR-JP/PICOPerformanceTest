#version 450
#extension GL_EXT_multiview : enable

layout(location=0) in vec2 vUV;
layout(location=0) out vec4 outMV;

layout(set=0, binding=0) uniform sampler2DArray depthTex;
layout(set=0, binding=1) uniform MvUBO {
    mat4 invCurMvp[2];
    mat4 prevMvp[2];
} ubo;

void main() {
    float d = texture(depthTex, vec3(vUV, float(gl_ViewIndex))).r;
    vec4 clipPos = vec4(vUV * 2.0 - 1.0, d, 1.0);
    vec4 worldPos = ubo.invCurMvp[gl_ViewIndex] * clipPos;
    worldPos /= worldPos.w;
    vec4 prevClip = ubo.prevMvp[gl_ViewIndex] * worldPos;
    prevClip /= prevClip.w;
    // Motion in NDC space: where this pixel was in the previous frame.
    vec2 motion = 0.5 * (prevClip.xy - clipPos.xy);
    outMV = vec4(motion, 0.0, 0.0);
}
