#version 450
#extension GL_EXT_multiview : enable
#extension GL_EXT_shader_16bit_storage : enable

// mode=5,6,7: VS が FatVertex + BoneData(mat4 fp16, 16要素/骨) を読んで LBS を実行。

struct FatVertex {
    vec4  pos;
    vec4  weights;
    uvec4 boneIdx;
};

layout(std430, binding = 0) readonly buffer FatVertBuf { FatVertex  verts[];    };
layout(std430, binding = 1) readonly buffer BoneBuf    { float16_t  boneData[]; };

mat4 getBone(uint idx) {
    uint b = idx * 16u;
    return mat4(
        float(boneData[b+ 0]), float(boneData[b+ 1]), float(boneData[b+ 2]), float(boneData[b+ 3]),
        float(boneData[b+ 4]), float(boneData[b+ 5]), float(boneData[b+ 6]), float(boneData[b+ 7]),
        float(boneData[b+ 8]), float(boneData[b+ 9]), float(boneData[b+10]), float(boneData[b+11]),
        float(boneData[b+12]), float(boneData[b+13]), float(boneData[b+14]), float(boneData[b+15])
    );
}

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
        if (w > 0.0) skinned += w * (getBone(v.boneIdx[b]) * p).xyz;
    }

    float dummy = 0.0;
    for (int i = 0; i < pc.aluIters; i++) {
        dummy += sin(float(i) * 0.001 + skinned.x);
    }
    gl_Position = pc.mvp[gl_ViewIndex] * vec4(skinned, 1.0) + vec4(dummy * 0.000001, 0.0, 0.0, 0.0);
}
