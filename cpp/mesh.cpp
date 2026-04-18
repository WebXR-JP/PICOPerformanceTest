#include "mesh.hpp"

#include "vk_resources.hpp"

uint16_t FloatToHalf(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1u;
    int      exp  = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man  = x & 0x7fffffu;
    if (exp <= 0)  return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | 0x7c00u);
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (man >> 13));
}

void UpdateBones(VulkanCtx& vk, float time) {
    const uint32_t N = vk.totalBones;
    if (N == 0) return;
    const float amp = glm::radians(8.0f);
    std::vector<uint16_t> boneData(16u * N);
    for (uint32_t i = 0; i < N; i++) {
        int bInCube = (int)(i % (uint32_t)BONES_PER_CUBE);
        float phase = (float)bInCube * (glm::pi<float>() * 0.25f);
        float ang   = amp * std::sin(time + phase);
        const glm::vec3& piv = vk.bonePivots[i];
        glm::mat4 T1 = glm::translate(glm::mat4(1.0f),  piv);
        glm::mat4 R  = glm::rotate(glm::mat4(1.0f), ang, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 T0 = glm::translate(glm::mat4(1.0f), -piv);
        glm::mat4 m  = T1 * R * T0;
        const float* src = glm::value_ptr(m);
        uint16_t* dst = boneData.data() + i * 16u;
        for (int k = 0; k < 16; k++) dst[k] = FloatToHalf(src[k]);
    }
    void* p = vk.boneBuffer.getAllocation().map();
    VkDeviceSize bSize = sizeof(uint16_t) * 16u * N;
    std::memcpy(p, boneData.data(), (size_t)bSize);
    vk.boneBuffer.getAllocation().unmap();
}

void GenerateMultiCubeMesh(VulkanCtx& vk, int N, int cubeCount) {
    struct Face { glm::vec3 origin, du, dv; };
    const Face faces[6] = {
        { glm::vec3( 1,-1,-1), glm::vec3( 0, 0, 2), glm::vec3( 0, 2, 0) },
        { glm::vec3(-1,-1, 1), glm::vec3( 0, 0,-2), glm::vec3( 0, 2, 0) },
        { glm::vec3(-1, 1,-1), glm::vec3( 2, 0, 0), glm::vec3( 0, 0, 2) },
        { glm::vec3(-1,-1, 1), glm::vec3( 2, 0, 0), glm::vec3( 0, 0,-2) },
        { glm::vec3( 1,-1, 1), glm::vec3(-2, 0, 0), glm::vec3( 0, 2, 0) },
        { glm::vec3(-1,-1,-1), glm::vec3( 2, 0, 0), glm::vec3( 0, 2, 0) },
    };

    const int tiles        = (N - 1) / MESHLET_TILE;
    const int Nfit         = tiles * MESHLET_TILE + 1;
    const int vPerFace     = Nfit * Nfit;
    const int iPerMeshlet  = MESHLET_TILE * MESHLET_TILE * 6;
    const int mPerFace     = tiles * tiles;
    const int vPerCube     = vPerFace * 6;
    const int iPerCube     = iPerMeshlet * mPerFace * 6;
    const int mPerCube     = mPerFace * 6;
    const int totalVerts    = vPerCube * cubeCount;
    const int totalIdx      = iPerCube * cubeCount;
    const int totalMeshlets = mPerCube * cubeCount;

    std::vector<FatVertex> verts(totalVerts);
    std::vector<uint32_t>  indices(totalIdx);
    std::vector<Meshlet>   meshlets(totalMeshlets);

    int GX = cubeCount, GY = 1, GZ = 1;
    int bestDiff = cubeCount;
    for (int gz = 1; gz * gz * gz <= cubeCount; gz++) {
        for (int gy = gz; gy * gz <= cubeCount; gy++) {
            int gx = (cubeCount + gy * gz - 1) / (gy * gz);
            if (gx >= gy && gx * gy * gz >= cubeCount) {
                int diff = gx - gz;
                if (diff < bestDiff) {
                    bestDiff = diff;
                    GX = gx; GY = gy; GZ = gz;
                }
            }
        }
    }
    const float spacing  = 0.6f;
    const float cubeSize = 0.25f;

    vk.bonePivots.resize((size_t)cubeCount * BONES_PER_CUBE);

    for (int c = 0; c < cubeCount; c++) {
        int gx = c % GX;
        int gy = (c / GX) % GY;
        int gz = c / (GX * GY);

        glm::vec3 center(
            (gx - (GX - 1) * 0.5f) * spacing,
            (gy - (GY - 1) * 0.5f) * spacing + 1.5f,
            -3.0f - (gz - (GZ - 1) * 0.5f) * spacing);

        for (int b = 0; b < BONES_PER_CUBE; b++) {
            float ly = -1.0f + 2.0f * (float)b / 7.0f;
            vk.bonePivots[c * BONES_PER_CUBE + b] = center + glm::vec3(0.0f, ly, 0.0f) * cubeSize;
        }

        const int vBaseC = c * vPerCube;
        const int iBaseC = c * iPerCube;
        const int mBaseC = c * mPerCube;

        for (int f = 0; f < 6; f++) {
            const Face& face   = faces[f];
            const glm::vec3 fN = glm::normalize(glm::cross(face.dv, face.du));
            const int   vBase = vBaseC + f * vPerFace;
            const int   iBase = iBaseC + f * iPerMeshlet * mPerFace;
            const int   mBase = mBaseC + f * mPerFace;

            for (int y = 0; y < Nfit; y++) {
                for (int x = 0; x < Nfit; x++) {
                    float u = (float)x / (Nfit - 1);
                    float v = (float)y / (Nfit - 1);
                    glm::vec3 localPos = face.origin + u * face.du + v * face.dv;
                    glm::vec3 worldPos = center + localPos * cubeSize;
                    FatVertex& vv = verts[vBase + y * Nfit + x];
                    vv.pos = glm::vec4(worldPos, 0.0f);

                    float ly = localPos.y;
                    float t  = (ly + 1.0f) * 0.5f * 7.0f;
                    int   b0 = (int)std::floor(t);
                    if (b0 < 0) b0 = 0;
                    if (b0 > 6) b0 = 6;
                    int   b1 = b0 + 1;
                    float w1 = t - (float)b0;
                    float w0 = 1.0f - w1;

                    uint32_t globalB0 = (uint32_t)(c * BONES_PER_CUBE + b0);
                    uint32_t globalB1 = (uint32_t)(c * BONES_PER_CUBE + b1);
                    vv.boneIdx = glm::uvec4(globalB0, globalB1, 0u, 0u);
                    vv.weights = glm::vec4(w0, w1, 0.0f, 0.0f);
                }
            }

            int iOff = iBase;
            for (int my = 0; my < tiles; my++) {
                for (int mx = 0; mx < tiles; mx++) {
                    uint32_t offsetStart = (uint32_t)iOff;
                    for (int ty = 0; ty < MESHLET_TILE; ty++) {
                        for (int tx = 0; tx < MESHLET_TILE; tx++) {
                            int xi = mx * MESHLET_TILE + tx;
                            int yi = my * MESHLET_TILE + ty;
                            uint32_t tl = (uint32_t)(vBase + yi * Nfit + xi);
                            uint32_t tr = tl + 1;
                            uint32_t bl = tl + Nfit;
                            uint32_t br = bl + 1;
                            indices[iOff++] = tl; indices[iOff++] = bl; indices[iOff++] = tr;
                            indices[iOff++] = tr; indices[iOff++] = bl; indices[iOff++] = br;
                        }
                    }
                    glm::vec3 pMin( 1e9f);
                    glm::vec3 pMax(-1e9f);
                    for (int ty = 0; ty <= MESHLET_TILE; ty++) {
                        for (int tx = 0; tx <= MESHLET_TILE; tx++) {
                            int xi = mx * MESHLET_TILE + tx;
                            int yi = my * MESHLET_TILE + ty;
                            float u = (float)xi / (Nfit - 1);
                            float v = (float)yi / (Nfit - 1);
                            glm::vec3 localPos = face.origin + u * face.du + v * face.dv;
                            glm::vec3 p = center + localPos * cubeSize;
                            pMin = glm::min(pMin, p);
                            pMax = glm::max(pMax, p);
                        }
                    }
                    for (int i = 0; i < 3; i++) {
                        if (pMax[i] - pMin[i] < 1e-4f) {
                            pMin[i] -= 0.001f;
                            pMax[i] += 0.001f;
                        }
                    }
                    {
                        glm::vec3 cc = 0.5f * (pMin + pMax);
                        glm::vec3 ext = pMax - pMin;
                        int thinAxis = 0;
                        if (ext[1] < ext[thinAxis]) thinAxis = 1;
                        if (ext[2] < ext[thinAxis]) thinAxis = 2;
                        glm::vec3 scale(1.1f);
                        scale[thinAxis] = 1.5f;
                        glm::vec3 hh = 0.5f * ext * scale;
                        pMin = cc - hh;
                        pMax = cc + hh;
                    }
                    Meshlet& m = meshlets[mBase + my * tiles + mx];
                    m.aabbMin     = pMin;
                    m.aabbMax     = pMax;
                    m.indexOffset = offsetStart;
                    m.indexCount  = (uint32_t)iPerMeshlet;
                    m.normal      = fN;
                    m.boneBase    = (uint32_t)(c * BONES_PER_CUBE);
                    m.boneCount   = (uint32_t)BONES_PER_CUBE;
                }
            }
        }
    }
    vk.meshletCount = (uint32_t)totalMeshlets;
    vk.vPerFace = (uint32_t)vPerFace;
    vk.vPerCube = (uint32_t)vPerCube;

    vk.vertexCount = (uint32_t)totalVerts;
    vk.indexCount  = (uint32_t)totalIdx;

    int polyCount = totalMeshlets * MESHLET_TILE * MESHLET_TILE * 2;
    LOGI("Multi-cube mesh: %d cubes (%dx%dx%d), %d verts, %d idx, %d polys, %d meshlets",
         cubeCount, GX, GY, GZ, totalVerts, totalIdx, polyCount, totalMeshlets);

    VkDeviceSize vSize = sizeof(FatVertex) * totalVerts;
    CreateBuffer(vk, vSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk.vertexBuffer);
    UploadBufferData(vk, vk.vertexBuffer, verts.data(), vSize);

    VkDeviceSize iSize = sizeof(uint32_t) * totalIdx;
    CreateBuffer(vk, iSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk.indexBuffer);
    UploadBufferData(vk, vk.indexBuffer, indices.data(), iSize);

    VkDeviceSize mSize = sizeof(Meshlet) * totalMeshlets;
    CreateBuffer(vk, mSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk.meshletBuffer);
    UploadBufferData(vk, vk.meshletBuffer, meshlets.data(), mSize);

    CreateBuffer(vk, iSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk.compactIndexBuffer);

    VkDeviceSize drawCmdSize = 5 * sizeof(uint32_t);
    CreateBuffer(vk, drawCmdSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vk.bfDrawCmdBuffer);

    CreateBuffer(vk, sizeof(uint32_t) * 40u,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vk.cullStatsBuffer);

    CreateBuffer(vk, sizeof(glm::mat4) * 2u,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vk.prevFrameBuffer);

    vk.totalBones = (uint32_t)(cubeCount * BONES_PER_CUBE);
    VkDeviceSize bSize = sizeof(uint16_t) * 16u * vk.totalBones;
    CreateBuffer(vk, bSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vk.boneBuffer);
    {
        std::vector<uint16_t> identData(16u * vk.totalBones, 0u);
        static const uint16_t one = 0x3C00u;
        for (uint32_t i = 0; i < vk.totalBones; i++) {
            uint16_t* m = identData.data() + i * 16u;
            m[0] = m[5] = m[10] = m[15] = one;
        }
        void* p = vk.boneBuffer.getAllocation().map();
        std::memcpy(p, identData.data(), (size_t)bSize);
        vk.boneBuffer.getAllocation().unmap();
    }
}
