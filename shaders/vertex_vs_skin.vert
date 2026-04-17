#version 450
#extension GL_EXT_multiview : enable

// mode=5: VS が FatVertex + BoneMatrix を直接読んで LBS を実行。
// pre-skin CS の posF16 書き出しなし → 帯域 vs ALU の切り分けテスト用。

struct FatVertex {
    vec4  pos;      // xyz = bind-pose, w unused
    vec4  weights;  // 4 skin weights
    uvec4 boneIdx;  // 4 global bone indices
};

layout(std430, binding = 0) readonly buffer FatVertBuf { FatVertex verts[]; };
layout(std430, binding = 1) readonly buffer BoneBuf    { mat4      bones[]; };

layout(push_constant) uniform PushConstants {
    mat4 mvp[2];
    int  aluIters;
} pc;

void main() {
    FatVertex v = verts[gl_VertexIndex];
    vec4 p = vec4(v.pos.xyz, 1.0);

    vec3 skinned = vec3(0.0);
    for (int b = 0; b < 4; b++) {
        float w = v.weights[b];
        if (w > 0.0) {
            skinned += w * (bones[v.boneIdx[b]] * p).xyz;
        }
    }

    float dummy = 0.0;
    for (int i = 0; i < pc.aluIters; i++) {
        dummy += sin(float(i) * 0.001 + skinned.x);
    }
    gl_Position = pc.mvp[gl_ViewIndex] * vec4(skinned, 1.0) + vec4(dummy * 0.000001, 0.0, 0.0, 0.0);
}
