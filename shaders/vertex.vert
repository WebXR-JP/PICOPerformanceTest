#version 450
#extension GL_EXT_multiview : enable

// skinnedPosBuffer（F16 packed）を gl_VertexIndex で参照する pulling 方式。
// pre-skin CS が毎フレーム書き込む。

layout(std430, binding = 0) readonly buffer SkinnedPos { uvec2 posF16[]; };

layout(push_constant) uniform PushConstants {
    mat4 mvp[2];   // [0]=左目, [1]=右目
    int  aluIters;
} pc;

void main() {
    uvec2 packed = posF16[gl_VertexIndex];
    vec2  xy     = unpackHalf2x16(packed.x);
    float z      = unpackHalf2x16(packed.y).x;
    vec3  pos    = vec3(xy.x, xy.y, z);

    float dummy = 0.0;
    for (int i = 0; i < pc.aluIters; i++) {
        dummy += sin(float(i) * 0.001 + pos.x);
    }
    gl_Position = pc.mvp[gl_ViewIndex] * vec4(pos, 1.0) + vec4(dummy * 0.000001, 0.0, 0.0, 0.0);
}
