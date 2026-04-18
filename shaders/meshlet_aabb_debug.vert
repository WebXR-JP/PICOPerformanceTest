#version 450
#extension GL_EXT_multiview : enable

struct Meshlet {
    vec3  aabbMin;   uint indexOffset;
    vec3  aabbMax;   uint indexCount;
    vec3  normal;    uint boneBase;
    uint  boneCount; uint _pad0; uint _pad1; uint _pad2;
};

layout(std430, binding = 0) readonly buffer MeshletBuf { Meshlet meshlets[]; };
layout(binding = 1) uniform sampler2DArray hiZTex;

layout(push_constant) uniform PC {
    mat4 mvp[2];
    vec4 viewportAndBias; // x=width, y=height, z=bias, w=hiZ mip count
} pc;

layout(location = 0) out vec4 outColor;

vec3 cornerOf(Meshlet m, int idx) {
    return vec3((idx & 1) != 0 ? m.aabbMax.x : m.aabbMin.x,
                (idx & 2) != 0 ? m.aabbMax.y : m.aabbMin.y,
                (idx & 4) != 0 ? m.aabbMax.z : m.aabbMin.z);
}

ivec2 edgeVerts(int edgeIdx) {
    switch (edgeIdx) {
    case 0:  return ivec2(0, 1);
    case 1:  return ivec2(1, 3);
    case 2:  return ivec2(3, 2);
    case 3:  return ivec2(2, 0);
    case 4:  return ivec2(4, 5);
    case 5:  return ivec2(5, 7);
    case 6:  return ivec2(7, 6);
    case 7:  return ivec2(6, 4);
    case 8:  return ivec2(0, 4);
    case 9:  return ivec2(1, 5);
    case 10: return ivec2(2, 6);
    default: return ivec2(3, 7);
    }
}

float sampleHiZ(uint eye, vec2 uv, float lod) {
    return textureLod(hiZTex, vec3(clamp(uv, vec2(0.0), vec2(1.0)), float(eye)), lod).r;
}

float chooseHiZMip(vec2 rectSizePx) {
    float maxExtentHiZ = max(max(rectSizePx.x, rectSizePx.y) * 0.5, 1.0);
    float maxMip = min(max(pc.viewportAndBias.w - 1.0, 0.0), 3.0);
    return clamp(max(ceil(log2(maxExtentHiZ)) - 1.0, 0.0), 0.0, maxMip);
}

bool occludedForEye(Meshlet m, uint eye) {
    vec2 minUv = vec2(1e9);
    vec2 maxUv = vec2(-1e9);
    float nearestDepth = 0.0;
    bool anyValid = false;
    for (int i = 0; i < 8; ++i) {
        vec4 clip = pc.mvp[eye] * vec4(cornerOf(m, i), 1.0);
        if (clip.w <= 1e-5) continue;
        vec3 ndc = clip.xyz / clip.w;
        vec2 uv = ndc.xy * 0.5 + 0.5;
        minUv = min(minUv, uv);
        maxUv = max(maxUv, uv);
        nearestDepth = max(nearestDepth, ndc.z);
        anyValid = true;
    }
    if (!anyValid) return false;
    vec2 rectMin = clamp(minUv, vec2(0.0), vec2(1.0));
    vec2 rectMax = clamp(maxUv, vec2(0.0), vec2(1.0));
    vec2 rectSizePx = max((rectMax - rectMin) * pc.viewportAndBias.xy, vec2(1.0));
    float lod = chooseHiZMip(rectSizePx);
    float texelSpanPx = exp2(lod + 1.0);
    vec2 hiZSize = max(ceil(pc.viewportAndBias.xy / texelSpanPx), vec2(1.0));
    vec2 texelMin = clamp(floor(rectMin * hiZSize) - vec2(1.0), vec2(0.0), hiZSize - 1.0);
    vec2 texelMax = clamp(ceil(rectMax * hiZSize) + vec2(1.0), vec2(0.0), hiZSize - 1.0);
    float depthLimit = max(nearestDepth + pc.viewportAndBias.z, 0.0);
    bool hasRendered = false;
    for (int ty = int(texelMin.y); ty <= int(texelMax.y); ++ty) {
        for (int tx = int(texelMin.x); tx <= int(texelMax.x); ++tx) {
            vec2 uv = (vec2(float(tx), float(ty)) + vec2(0.5)) / hiZSize;
            float hRaw = sampleHiZ(eye, uv, lod);
            bool valid = (hRaw >= 1e-4 && hRaw < 1.0);
            float h = valid ? hRaw : 1.0;
            if (valid) hasRendered = true;
            if (h <= depthLimit) {
                return false;
            }
        }
    }
    return hasRendered;  // 実レンダ値が 1 件もなければ遮蔽と判定しない
}

void main() {
    Meshlet m = meshlets[gl_InstanceIndex];
    bool occluded = occludedForEye(m, uint(gl_ViewIndex));

    ivec2 edge = edgeVerts(gl_VertexIndex / 2);
    int cornerIdx = ((gl_VertexIndex & 1) == 0) ? edge.x : edge.y;
    gl_Position = pc.mvp[gl_ViewIndex] * vec4(cornerOf(m, cornerIdx), 1.0);
    outColor = occluded ? vec4(1.0, 0.15, 0.1, 1.0)    // 赤 = reject
                        : vec4(0.1, 1.0, 0.3, 1.0);    // 緑 = visible
}
