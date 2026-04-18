/**
 * PICOPerformanceTest - OpenXR + Vulkan 頂点負荷テスト
 *
 * 目的: PICO4 上で大量ポリゴン（100万〜1000万）を描画し、
 *       フレームタイムを計測してGPU頂点処理性能を評価する。
 *
 * グリッドメッシュを GRID_N×GRID_N の格子で生成し、
 * ステレオレンダリング（左右目）で毎フレーム描画する。
 * フレームタイム（ms）と頂点数を毎秒 logcat に出力する。
 */

#include <android/log.h>
#include <android_native_app_glue.h>
#include <jni.h>

// Vulkan は openxr_platform.h より前にインクルードする必要がある
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

// 生成したシェーダーヘッダー（ビルド時に cmake/embed_spirv.cmake が生成）
#include "vertex_vs_skin_spv.h"
#include "fragment_spv.h"
#include "skin_cull_lds_spv.h"
#include "hiz_spd_spv.h"

// ---- ログマクロ ----
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "PICOPerfTest", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  "PICOPerfTest", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PICOPerfTest", __VA_ARGS__)

// ---- 負荷設定 ----
// GRID_N を変えて負荷を調整する
// ポリゴン数 ≈ 2 * (GRID_N - 1)^2
//   GRID_N=708  → 約 1,000,000 ポリゴン
//   GRID_N=1584 → 約 5,000,000 ポリゴン
//   GRID_N=2237 → 約 10,000,000 ポリゴン
#ifndef GRID_N
#define GRID_N 708
#endif

// ---- CHECK マクロ ----
#define CHECK_XR(expr)                                                     \
    do {                                                                   \
        XrResult _r = (expr);                                              \
        if (XR_FAILED(_r)) {                                               \
            LOGE("XR error %d at %s:%d", (int)_r, __FILE__, __LINE__);    \
            assert(false);                                                 \
        }                                                                  \
    } while (0)

#define CHECK_VK(expr)                                                     \
    do {                                                                   \
        VkResult _r = (expr);                                              \
        if (_r != VK_SUCCESS) {                                            \
            LOGE("VK error %d at %s:%d", (int)_r, __FILE__, __LINE__);    \
            assert(false);                                                 \
        }                                                                  \
    } while (0)

// ============================================================
// Vulkan コンテキスト
// ============================================================
struct SwapchainImage {
    VkImage     image;
    VkImageView view;
    VkFramebuffer framebuffers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
};

struct EyeSwapchain {
    XrSwapchain handle = XR_NULL_HANDLE;
    uint32_t    width  = 0;
    uint32_t    height = 0;
    std::vector<SwapchainImage> images;
    // デプスバッファ（全画像で共有）
    VkImage        depthImage  = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView    depthView   = VK_NULL_HANDLE;
    VkImage        prevDepthImages[2]  = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory prevDepthMemories[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView    prevDepthViews[2]   = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImage        motionImages[2]     = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory motionMemories[2]   = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView    motionViews[2]      = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImage        hiZImages[2]        = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory hiZMemories[2]      = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView    hiZViews[2]         = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::vector<VkImageView> hiZMipViews[2];
    uint32_t       hiZW                = 0;
    uint32_t       hiZH                = 0;
    uint32_t       hiZMipCount         = 0;
};

struct VulkanCtx {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice     = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    uint32_t         queueFamily    = 0;
    VkQueue          queue          = VK_NULL_HANDLE;
    VkCommandPool    cmdPool        = VK_NULL_HANDLE;
    VkRenderPass     renderPass     = VK_NULL_HANDLE;
    VkRenderPass     captureRenderPass = VK_NULL_HANDLE;
    VkBuffer         vertexBuffer   = VK_NULL_HANDLE;
    VkDeviceMemory   vertexMemory   = VK_NULL_HANDLE;
    VkBuffer         indexBuffer    = VK_NULL_HANDLE;
    VkDeviceMemory   indexMemory    = VK_NULL_HANDLE;
    uint32_t         indexCount     = 0;
    uint32_t         vertexCount    = 0;
    VkFence          fence          = VK_NULL_HANDLE;
    // コマンドバッファ（スワップチェーン画像数、multiview で両目を1回で描画）
    std::vector<VkCommandBuffer> cmdBuffers; // [imageIdx]

    // ---- Meshlet Buffer ----
    VkBuffer              meshletBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory        meshletMemory        = VK_NULL_HANDLE;
    uint32_t              meshletCount         = 0;

    // ---- Timestamp Query ----
    VkQueryPool           queryPool              = VK_NULL_HANDLE;
    float                 timestampPeriod        = 1.f; // ns/tick

    // ---- Per-triangle Backface Culling (Compute) ----
    VkBuffer              compactIndexBuffer     = VK_NULL_HANDLE;
    VkDeviceMemory        compactIndexMemory     = VK_NULL_HANDLE;
    VkBuffer              bfDrawCmdBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory        bfDrawCmdMemory        = VK_NULL_HANDLE;

    // ---- Skinning ----
    VkBuffer              boneBuffer             = VK_NULL_HANDLE;
    VkDeviceMemory        boneMemory             = VK_NULL_HANDLE;
    uint32_t              totalBones             = 0;
    std::vector<glm::vec3> bonePivots;            // world 空間のボーン原点（回転軸）

    // ---- mode=5: VS-inline skinning（CS posF16 書き出しなし）----
    VkDescriptorSetLayout vsSkinDescLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      vsSkinDescPool        = VK_NULL_HANDLE;
    VkDescriptorSet       vsSkinDescSet         = VK_NULL_HANDLE;
    VkPipelineLayout      vsSkinPipelineLayout  = VK_NULL_HANDLE;
    VkPipeline            vsSkinPipeline        = VK_NULL_HANDLE;
    VkPipeline            vsSkinCapturePipeline = VK_NULL_HANDLE;

    // ---- mode=7: CS LDS skin+cull（1 workgroup = 1 meshlet）----
    VkDescriptorSetLayout skinCullLdsDescLayout    = VK_NULL_HANDLE;
    VkDescriptorPool      skinCullLdsDescPool      = VK_NULL_HANDLE;
    VkDescriptorSet       skinCullLdsDescSets[2]   = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout      skinCullLdsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            skinCullLdsPipeline      = VK_NULL_HANDLE;
    VkSampler             prevDepthSampler         = VK_NULL_HANDLE;
    VkSampler             hiZSampler               = VK_NULL_HANDLE;
    VkBuffer              hiZSpdAtomicBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory        hiZSpdAtomicMemory       = VK_NULL_HANDLE;
    VkDescriptorSetLayout hiZSpdDescLayout         = VK_NULL_HANDLE;
    VkDescriptorPool      hiZSpdDescPool           = VK_NULL_HANDLE;
    VkDescriptorSet       hiZSpdDescSets[2]        = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout      hiZSpdPipelineLayout     = VK_NULL_HANDLE;
    VkPipeline            hiZSpdPipeline           = VK_NULL_HANDLE;
};

// timestamp index 定数
// 0: CS開始, 1: CS終了, 2: Graphics開始, 3: Graphics終了
static constexpr uint32_t TS_CS_BEGIN         = 0;
static constexpr uint32_t TS_CS_END           = 1;
static constexpr uint32_t TS_GFX_BEGIN        = 2;
static constexpr uint32_t TS_GFX_END          = 3;
static constexpr uint32_t TS_HIZ_BEGIN        = 4;
static constexpr uint32_t TS_HIZ_END          = 5;
static constexpr uint32_t TS_COUNT            = 6;
static constexpr uint32_t HIZ_SPD_MAX_MIPS    = 12;

// ============================================================
// OpenXR コンテキスト
// ============================================================
struct XrCtx {
    XrInstance    instance    = XR_NULL_HANDLE;
    XrSystemId    systemId    = XR_NULL_SYSTEM_ID;
    XrSession     session     = XR_NULL_HANDLE;
    XrSpace       appSpace    = XR_NULL_HANDLE;
    XrSessionState sessionState = XR_SESSION_STATE_UNKNOWN;
    bool          sessionRunning = false;
    bool          exitRequested  = false;
    std::vector<EyeSwapchain> swapchains; // [0]=左目, [1]=右目
};

// ============================================================
// OpenXR 拡張関数ポインタ
// ============================================================
static PFN_xrGetVulkanGraphicsRequirements2KHR  pfn_xrGetVulkanGraphicsRequirements2KHR = nullptr;
static PFN_xrCreateVulkanInstanceKHR            pfn_xrCreateVulkanInstanceKHR           = nullptr;
static PFN_xrGetVulkanGraphicsDevice2KHR        pfn_xrGetVulkanGraphicsDevice2KHR       = nullptr;
static PFN_xrCreateVulkanDeviceKHR              pfn_xrCreateVulkanDeviceKHR             = nullptr;

static void LoadXrExtFunctions(XrInstance instance) {
    CHECK_XR(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirements2KHR",
        (PFN_xrVoidFunction*)&pfn_xrGetVulkanGraphicsRequirements2KHR));
    CHECK_XR(xrGetInstanceProcAddr(instance, "xrCreateVulkanInstanceKHR",
        (PFN_xrVoidFunction*)&pfn_xrCreateVulkanInstanceKHR));
    CHECK_XR(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDevice2KHR",
        (PFN_xrVoidFunction*)&pfn_xrGetVulkanGraphicsDevice2KHR));
    CHECK_XR(xrGetInstanceProcAddr(instance, "xrCreateVulkanDeviceKHR",
        (PFN_xrVoidFunction*)&pfn_xrCreateVulkanDeviceKHR));
}

// ============================================================
// Vulkan ユーティリティ
// ============================================================
static uint32_t FindMemoryType(VkPhysicalDevice physDevice,
                               uint32_t typeBits,
                               VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    LOGE("FindMemoryType: no suitable memory type found");
    assert(false);
    return 0;
}

static void CreateBuffer(VulkanCtx& vk, VkDeviceSize size,
                         VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags memProps,
                         VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size        = size;
    bi.usage       = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VK(vkCreateBuffer(vk.device, &bi, nullptr, &buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk.device, buffer, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = FindMemoryType(vk.physDevice, req.memoryTypeBits, memProps);
    CHECK_VK(vkAllocateMemory(vk.device, &ai, nullptr, &memory));
    CHECK_VK(vkBindBufferMemory(vk.device, buffer, memory, 0));
}

static void UploadBufferData(VulkanCtx& vk, VkBuffer dstBuffer, const void* srcData, VkDeviceSize size) {
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    CreateBuffer(vk, size,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingMemory);

    void* mapped = nullptr;
    CHECK_VK(vkMapMemory(vk.device, stagingMemory, 0, size, 0, &mapped));
    memcpy(mapped, srcData, (size_t)size);
    vkUnmapMemory(vk.device, stagingMemory);

    VkCommandBufferAllocateInfo cbAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool = vk.cmdPool;
    cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateCommandBuffers(vk.device, &cbAI, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, stagingBuffer, dstBuffer, 1, &region);
    CHECK_VK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    CHECK_VK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(vk.queue));

    vkFreeCommandBuffers(vk.device, vk.cmdPool, 1, &cmd);
    vkDestroyBuffer(vk.device, stagingBuffer, nullptr);
    vkFreeMemory(vk.device, stagingMemory, nullptr);
}

static void CreateImage(VulkanCtx& vk, uint32_t w, uint32_t h,
                        VkFormat format, VkImageUsageFlags usage,
                        VkImage& image, VkDeviceMemory& memory,
                        uint32_t arrayLayers = 1,
                        uint32_t mipLevels = 1) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType   = VK_IMAGE_TYPE_2D;
    ci.format      = format;
    ci.extent      = {w, h, 1};
    ci.mipLevels   = mipLevels;
    ci.arrayLayers = arrayLayers;
    ci.samples     = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    CHECK_VK(vkCreateImage(vk.device, &ci, nullptr, &image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(vk.device, image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = FindMemoryType(vk.physDevice, req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK_VK(vkAllocateMemory(vk.device, &ai, nullptr, &memory));
    CHECK_VK(vkBindImageMemory(vk.device, image, memory, 0));
}

static VkImageView CreateImageView(VulkanCtx& vk,
                                   VkImage image,
                                   VkFormat format,
                                   VkImageAspectFlags aspectMask,
                                   VkImageViewType viewType,
                                   uint32_t baseMipLevel,
                                   uint32_t levelCount,
                                   uint32_t baseArrayLayer,
                                   uint32_t layerCount) {
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = image;
    ci.viewType = viewType;
    ci.format = format;
    ci.subresourceRange = {aspectMask, baseMipLevel, levelCount, baseArrayLayer, layerCount};
    VkImageView view = VK_NULL_HANDLE;
    CHECK_VK(vkCreateImageView(vk.device, &ci, nullptr, &view));
    return view;
}

static void TransitionImageLayoutNow(VulkanCtx& vk,
                                     VkImage image,
                                     VkImageAspectFlags aspectMask,
                                     VkImageLayout oldLayout,
                                     VkImageLayout newLayout,
                                     uint32_t layerCount = 1,
                                     uint32_t mipCount = 1) {
    VkCommandBufferAllocateInfo cbAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool = vk.cmdPool;
    cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateCommandBuffers(vk.device, &cbAI, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(cmd, &bi));

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspectMask, 0, mipCount, 0, layerCount};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = (newLayout == VK_IMAGE_LAYOUT_GENERAL)
        ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
        : 0;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    CHECK_VK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    CHECK_VK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(vk.queue));
    vkFreeCommandBuffers(vk.device, vk.cmdPool, 1, &cmd);
}

// ============================================================
// 頂点フォーマット
// fatVertex=false: vec3 のみ（12 bytes）
// FatVertex: vec4 pos(xyz) + vec4 weights(4) + uvec4 boneIdx(4) = 48B, std430 整列
// ============================================================
struct FatVertex {
    glm::vec4  pos;      // xyz = bind-pose 位置、w 未使用
    glm::vec4  weights;  // 4 本分のスキンウェイト
    glm::uvec4 boneIdx;  // 4 本分のボーンインデックス（グローバル）
};

static constexpr int BONES_PER_CUBE = 8;

// ============================================================
// Meshlet 構造体（Compute shader と共有、std430 レイアウト）
// ============================================================
struct Meshlet {
    glm::vec3 aabbMin;
    uint32_t  indexOffset;
    glm::vec3 aabbMax;
    uint32_t  indexCount;
    glm::vec3 normal;
    uint32_t  boneBase;   // このmeshletが使う最初のグローバル骨インデックス
    uint32_t  boneCount;  // ユニーク骨数
    uint32_t  _pad[3];
};
static_assert(sizeof(Meshlet) == 64, "Meshlet struct must be 64 bytes (std430)");

// ============================================================
// 複数立方体メッシュ生成（world 空間に 3D 格子配置、meshlet 分割）
// ============================================================
static constexpr int MESHLET_TILE = 16; // 16×16 quads/meshlet = 512 tris

// float → fp16 変換（CPU 側）
static uint16_t FloatToHalf(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1u;
    int      exp  = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man  = x & 0x7fffffu;
    if (exp <= 0)  return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | 0x7c00u);
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (man >> 13));
}
// vec3 を uvec2 に pack（x,y → .x の上下16bit、z → .y 下16bit）
// CPU 側でボーン行列を更新して host-visible boneBuffer へ書き込む。
// 各ボーンを世界空間のピボット回りで Y 軸回転させ、位相をずらした正弦波で振動させる。
static void UpdateBones(VulkanCtx& vk, float time) {
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
    void* p = nullptr;
    VkDeviceSize bSize = sizeof(uint16_t) * 16u * N;
    vkMapMemory(vk.device, vk.boneMemory, 0, bSize, 0, &p);
    memcpy(p, boneData.data(), bSize);
    vkUnmapMemory(vk.device, vk.boneMemory);
}

static void GenerateMultiCubeMesh(VulkanCtx& vk, int N, int cubeCount) {
    // 立方体の6面（origin=左下頂点、du=横方向、dv=縦方向、反時計回りで外向き法線）
    struct Face { glm::vec3 origin, du, dv; };
    const Face faces[6] = {
        { glm::vec3( 1,-1,-1), glm::vec3( 0, 0, 2), glm::vec3( 0, 2, 0) }, // +X
        { glm::vec3(-1,-1, 1), glm::vec3( 0, 0,-2), glm::vec3( 0, 2, 0) }, // -X
        { glm::vec3(-1, 1,-1), glm::vec3( 2, 0, 0), glm::vec3( 0, 0, 2) }, // +Y
        { glm::vec3(-1,-1, 1), glm::vec3( 2, 0, 0), glm::vec3( 0, 0,-2) }, // -Y
        { glm::vec3( 1,-1, 1), glm::vec3(-2, 0, 0), glm::vec3( 0, 2, 0) }, // +Z
        { glm::vec3(-1,-1,-1), glm::vec3( 2, 0, 0), glm::vec3( 0, 2, 0) }, // -Z
    };

    // 端数は切り捨て（TILEの倍数になる範囲のみメッシュ化）
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

    // 3D 格子配置（できるだけ cubeCount に近い分解を探索）
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
            (gy - (GY - 1) * 0.5f) * spacing,
            -3.0f - (gz - (GZ - 1) * 0.5f) * spacing);

        // 8 ボーンを Y 軸沿いに配置（ly ∈ [-1, 1]）
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

                    // ---- 8 ボーンを Y 軸沿い（localPos.y ∈ [-1, 1]）に配置し、
                    //      上下 2 本のボーンに距離比で weight を振る（4 枠中 2 本のみ使用）----
                    //   y = -1 + 2 * (b / 7.0), b ∈ [0, 7]
                    float ly = localPos.y; // ∈ [-1, 1]
                    float t  = (ly + 1.0f) * 0.5f * 7.0f; // ∈ [0, 7]
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
                    // meshlet AABB（world 空間）
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
                    // Skinned mesh 想定: AABB を 1.3 倍に膨張（ボーン回転マージン）
                    {
                        glm::vec3 cc = 0.5f * (pMin + pMax);
                        glm::vec3 hh = 0.5f * (pMax - pMin) * 1.3f;
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

    vk.vertexCount = (uint32_t)totalVerts;
    vk.indexCount  = (uint32_t)totalIdx;

    int polyCount = totalMeshlets * MESHLET_TILE * MESHLET_TILE * 2;
    LOGI("Multi-cube mesh: %d cubes (%dx%dx%d), %d verts, %d idx, %d polys, %d meshlets",
         cubeCount, GX, GY, GZ, totalVerts, totalIdx, polyCount, totalMeshlets);

    // 頂点バッファ（初期化時に staging upload して DEVICE_LOCAL 化）
    VkDeviceSize vSize = sizeof(FatVertex) * totalVerts;
    CreateBuffer(vk, vSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk.vertexBuffer, vk.vertexMemory);
    UploadBufferData(vk, vk.vertexBuffer, verts.data(), vSize);

    // インデックスバッファ（backface CS が読む、staging upload 後は DEVICE_LOCAL）
    VkDeviceSize iSize = sizeof(uint32_t) * totalIdx;
    CreateBuffer(vk, iSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk.indexBuffer, vk.indexMemory);
    UploadBufferData(vk, vk.indexBuffer, indices.data(), iSize);

    // ---- Meshlet SSBO（初期化時に staging upload して DEVICE_LOCAL 化）----
    VkDeviceSize mSize = sizeof(Meshlet) * totalMeshlets;
    CreateBuffer(vk, mSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk.meshletBuffer, vk.meshletMemory);
    UploadBufferData(vk, vk.meshletBuffer, meshlets.data(), mSize);

    // ---- Per-triangle backface culling 用バッファ（mode=7 で使用）----
    // 生存インデックス（最悪ケース = 元 index と同サイズ）
    CreateBuffer(vk, iSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk.compactIndexBuffer, vk.compactIndexMemory);

    // 単一の DrawCmd（indexCount は CS が atomicAdd で書く、他は固定）
    VkDeviceSize drawCmdSize = 5 * sizeof(uint32_t);
    CreateBuffer(vk, drawCmdSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vk.bfDrawCmdBuffer, vk.bfDrawCmdMemory);

    // ---- Bone buffer (60 cubes × 8 bones = 480 bones)、mat4 fp16（16要素/骨、列優先）、CPU が毎フレーム更新 ----
    vk.totalBones = (uint32_t)(cubeCount * BONES_PER_CUBE);
    VkDeviceSize bSize = sizeof(uint16_t) * 16u * vk.totalBones;
    CreateBuffer(vk, bSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vk.boneBuffer, vk.boneMemory);
    // 初期値は identity（fp16 対角1.0=0x3C00）
    {
        std::vector<uint16_t> identData(16u * vk.totalBones, 0u);
        static const uint16_t one = 0x3C00u;
        for (uint32_t i = 0; i < vk.totalBones; i++) {
            uint16_t* m = identData.data() + i * 16u;
            m[0] = m[5] = m[10] = m[15] = one;
        }
        void* p;
        vkMapMemory(vk.device, vk.boneMemory, 0, bSize, 0, &p);
        memcpy(p, identData.data(), bSize);
        vkUnmapMemory(vk.device, vk.boneMemory);
    }
}

// ============================================================
// アプリケーション本体
// ============================================================
struct App {
    android_app*  androidApp = nullptr;
    XrCtx         xr;
    VulkanCtx     vk;
    bool          initialized = false;
    int           gridN       = GRID_N; // Intent の "grid_n" で上書き可能
    int           instCount   = 1;      // Intent の "inst_count" で上書き可能
    int           aluIters    = 0;      // Intent の "alu_iters" で上書き可能
    int           cubeCount   = 60;     // Intent の "num_cubes" で上書き可能
    int           mode7Frustum = 1;     // Intent の "mode7_frustum" で上書き可能
    float         resScale    = 1.0f;   // Intent の "res_scale" で上書き可能（0.25-1.0）
    uint32_t      lastBfIndexCount = 0; // 前フレームの backface 生存インデックス数
    double        lastCsMs         = 0.0; // 前フレームの CS 時間（ms）
    double        lastGfxMs        = 0.0; // 前フレームの Graphics pass 時間（ms）
    double        lastHiZMs        = 0.0; // 前フレームの Hi-Z SPD 時間（ms）
    double        lastDownsampleMs = 0.0; // 前フレームの Hi-Z 時間（ms）
    uint32_t      frameParity      = 0;
    bool          prevDepthValid   = false;
    bool          frozenViewsValid = false;
    XrView        frozenViews[2]   = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};

    // フレーム計測
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point lastLogTime;
    Clock::time_point startTime;
    int    frameCount     = 0;
    double frameMsAccum   = 0.0;
};

// ---- OpenXR ローダー初期化（Android では xrCreateInstance の前に必須）----
static void InitializeOpenXRLoader(App& app) {
    PFN_xrInitializeLoaderKHR initLoader = nullptr;
    CHECK_XR(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
        (PFN_xrVoidFunction*)&initLoader));

    XrLoaderInitInfoAndroidKHR loaderInfo{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
    loaderInfo.applicationVM      = app.androidApp->activity->vm;
    loaderInfo.applicationContext = app.androidApp->activity->clazz;
    CHECK_XR(initLoader((const XrLoaderInitInfoBaseHeaderKHR*)&loaderInfo));
    LOGI("OpenXR loader initialized");
}

// ---- OpenXR インスタンス作成 ----
static void CreateXrInstance(App& app) {
    // 拡張一覧
    const char* extensions[] = {
        "XR_KHR_android_create_instance",
        "XR_KHR_vulkan_enable2",
    };

    XrInstanceCreateInfoAndroidKHR androidInfo{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
    androidInfo.applicationVM       = app.androidApp->activity->vm;
    androidInfo.applicationActivity = app.androidApp->activity->clazz;

    XrApplicationInfo appInfo{};
    strncpy(appInfo.applicationName, "PICOPerfTest", XR_MAX_APPLICATION_NAME_SIZE - 1);
    strncpy(appInfo.engineName,      "None",          XR_MAX_ENGINE_NAME_SIZE - 1);
    appInfo.apiVersion = XR_MAKE_VERSION(1, 0, 34); // PICOランタイムは OpenXR 1.0 のみ対応

    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
    ci.next                    = &androidInfo;
    ci.applicationInfo         = appInfo;
    ci.enabledExtensionCount   = (uint32_t)(sizeof(extensions) / sizeof(extensions[0]));
    ci.enabledExtensionNames   = extensions;

    CHECK_XR(xrCreateInstance(&ci, &app.xr.instance));
    LOGI("XrInstance created");

    LoadXrExtFunctions(app.xr.instance);
}

// ---- Vulkan インスタンス・デバイス作成（OpenXR 経由）----
static void CreateVulkanDevice(App& app) {
    // Vulkan バージョン要件を確認
    XrGraphicsRequirementsVulkan2KHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
    CHECK_XR(pfn_xrGetVulkanGraphicsRequirements2KHR(
        app.xr.instance, app.xr.systemId, &req));
    LOGI("Vulkan version requirement: min=%d.%d, max=%d.%d",
        XR_VERSION_MAJOR(req.minApiVersionSupported),
        XR_VERSION_MINOR(req.minApiVersionSupported),
        XR_VERSION_MAJOR(req.maxApiVersionSupported),
        XR_VERSION_MINOR(req.maxApiVersionSupported));

    // VkInstance 作成（xrCreateVulkanInstanceKHR 経由）
    // 注意: OpenXR 統合では Surface 拡張は不要（OpenXR が直接 swapchain を管理する）
    VkApplicationInfo vkApp{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    vkApp.pApplicationName = "PICOPerfTest";
    vkApp.apiVersion       = VK_API_VERSION_1_0;

    VkInstanceCreateInfo vkInstCI{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    vkInstCI.pApplicationInfo        = &vkApp;
    vkInstCI.enabledExtensionCount   = 0;
    vkInstCI.ppEnabledExtensionNames = nullptr;

    XrVulkanInstanceCreateInfoKHR xrInstCI{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
    xrInstCI.systemId                = app.xr.systemId;
    xrInstCI.pfnGetInstanceProcAddr  = vkGetInstanceProcAddr;
    xrInstCI.vulkanCreateInfo        = &vkInstCI;

    VkResult vkResult;
    CHECK_XR(pfn_xrCreateVulkanInstanceKHR(
        app.xr.instance, &xrInstCI, &app.vk.instance, &vkResult));
    CHECK_VK(vkResult);
    LOGI("VkInstance created");

    // Physical Device を取得
    XrVulkanGraphicsDeviceGetInfoKHR devGetInfo{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
    devGetInfo.systemId     = app.xr.systemId;
    devGetInfo.vulkanInstance = app.vk.instance;
    CHECK_XR(pfn_xrGetVulkanGraphicsDevice2KHR(
        app.xr.instance, &devGetInfo, &app.vk.physDevice));
    LOGI("VkPhysicalDevice selected");

    // キューファミリー選択（グラフィックスキュー）
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(app.vk.physDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(app.vk.physDevice, &qfCount, qfProps.data());
    app.vk.queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            app.vk.queueFamily = i;
            break;
        }
    }
    assert(app.vk.queueFamily != UINT32_MAX);

    // maxPushConstantsSize を確認（multiview では mat4[2]+int = 132 バイト必要）
    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(app.vk.physDevice, &devProps);
    LOGI("maxPushConstantsSize=%u", devProps.limits.maxPushConstantsSize);

    // VkDevice 作成（xrCreateVulkanDeviceKHR 経由）
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = app.vk.queueFamily;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;

    // VK_KHR_draw_indirect_count サポート確認
    {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(app.vk.physDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(app.vk.physDevice, nullptr, &extCount, exts.data());
        bool found = false;
        for (auto& e : exts) {
            if (strcmp(e.extensionName, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME) == 0) {
                found = true; break;
            }
        }
        LOGI("VK_KHR_draw_indirect_count: %s", found ? "YES" : "NO");
        if (!found) { LOGE("VK_KHR_draw_indirect_count not supported"); assert(false); }
    }

    // multiview / subgroup size control を有効化
    VkPhysicalDeviceMultiviewFeatures multiviewFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES};
    multiviewFeatures.multiview = VK_TRUE;

    VkPhysicalDeviceVulkan13Features vulkan13Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    vulkan13Features.subgroupSizeControl = VK_TRUE;
    vulkan13Features.computeFullSubgroups = VK_TRUE;
    multiviewFeatures.pNext = &vulkan13Features;

    // multiDrawIndirect feature（drawCount > 1 の indirect draw に必須）
    VkPhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.multiDrawIndirect = VK_TRUE;

    const char* deviceExtensions[] = {
        VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
    };

    VkDeviceCreateInfo devCI{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devCI.pNext                    = &multiviewFeatures;
    devCI.queueCreateInfoCount     = 1;
    devCI.pQueueCreateInfos        = &qci;
    devCI.pEnabledFeatures         = &enabledFeatures;
    devCI.enabledExtensionCount    = (uint32_t)(sizeof(deviceExtensions) / sizeof(deviceExtensions[0]));
    devCI.ppEnabledExtensionNames  = deviceExtensions;

    XrVulkanDeviceCreateInfoKHR xrDevCI{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
    xrDevCI.systemId               = app.xr.systemId;
    xrDevCI.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    xrDevCI.vulkanPhysicalDevice   = app.vk.physDevice;
    xrDevCI.vulkanCreateInfo       = &devCI;

    CHECK_XR(pfn_xrCreateVulkanDeviceKHR(
        app.xr.instance, &xrDevCI, &app.vk.device, &vkResult));
    CHECK_VK(vkResult);
    LOGI("VkDevice created");

    vkGetDeviceQueue(app.vk.device, app.vk.queueFamily, 0, &app.vk.queue);
}

// ---- XrSession 作成 ----
static void CreateXrSession(App& app) {
    XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    binding.instance         = app.vk.instance;
    binding.physicalDevice   = app.vk.physDevice;
    binding.device           = app.vk.device;
    binding.queueFamilyIndex = app.vk.queueFamily;
    binding.queueIndex       = 0;

    XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
    sci.next     = &binding;
    sci.systemId = app.xr.systemId;
    CHECK_XR(xrCreateSession(app.xr.instance, &sci, &app.xr.session));
    LOGI("XrSession created");

    // 参照空間（ローカル）
    XrReferenceSpaceCreateInfo rci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    rci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    rci.poseInReferenceSpace.orientation = {0, 0, 0, 1};
    rci.poseInReferenceSpace.position    = {0, 0, 0};
    CHECK_XR(xrCreateReferenceSpace(app.xr.session, &rci, &app.xr.appSpace));
}

// ---- スワップチェーン作成 ----
static void CreateSwapchains(App& app) {
    // ビュー設定の取得
    uint32_t viewCount = 0;
    CHECK_XR(xrEnumerateViewConfigurationViews(
        app.xr.instance, app.xr.systemId,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr));
    std::vector<XrViewConfigurationView> vcViews(viewCount,
        {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    CHECK_XR(xrEnumerateViewConfigurationViews(
        app.xr.instance, app.xr.systemId,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, vcViews.data()));

    // サポートされているフォーマットから RGBA8 SRGB を選択
    uint32_t fmtCount = 0;
    CHECK_XR(xrEnumerateSwapchainFormats(app.xr.session, 0, &fmtCount, nullptr));
    std::vector<int64_t> fmts(fmtCount);
    CHECK_XR(xrEnumerateSwapchainFormats(app.xr.session, fmtCount, &fmtCount, fmts.data()));

    int64_t colorFmt = (int64_t)VK_FORMAT_R8G8B8A8_SRGB;
    bool found = false;
    for (int64_t f : fmts) {
        if (f == (int64_t)VK_FORMAT_R8G8B8A8_SRGB) { found = true; break; }
    }
    if (!found) {
        colorFmt = fmts[0]; // フォールバック
        LOGW("VK_FORMAT_R8G8B8A8_SRGB not supported, using format %lld", (long long)colorFmt);
    }
    LOGI("Swapchain color format: %lld", (long long)colorFmt);

    // multiview: 1つの swapchain（arraySize=2）で両目を扱う
    app.xr.swapchains.resize(1);
    EyeSwapchain& sc = app.xr.swapchains[0];
    sc.width  = (uint32_t)(vcViews[0].recommendedImageRectWidth  * app.resScale);
    sc.height = (uint32_t)(vcViews[0].recommendedImageRectHeight * app.resScale);

    XrSwapchainCreateInfo sci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    sci.usageFlags  = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                      XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    sci.format      = colorFmt;
    sci.sampleCount = 1;
    sci.width       = sc.width;
    sci.height      = sc.height;
    sci.faceCount   = 1;
    sci.arraySize   = 2;  // 両目分
    sci.mipCount    = 1;
    CHECK_XR(xrCreateSwapchain(app.xr.session, &sci, &sc.handle));

    uint32_t imgCount = 0;
    CHECK_XR(xrEnumerateSwapchainImages(sc.handle, 0, &imgCount, nullptr));
    std::vector<XrSwapchainImageVulkanKHR> xrImages(imgCount,
        {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
    CHECK_XR(xrEnumerateSwapchainImages(sc.handle, imgCount, &imgCount,
        (XrSwapchainImageBaseHeader*)xrImages.data()));

    sc.images.resize(imgCount);
    for (uint32_t i = 0; i < imgCount; i++) {
        sc.images[i].image = xrImages[i].image;
    }
    LOGI("Stereo swapchain: %dx%d arraySize=2, %d images", sc.width, sc.height, imgCount);
}

// ---- RenderPass 作成 ----
static void CreateRenderPass(App& app, VkFormat colorFormat) {
    VkAttachmentDescription attachments[4]{};

    // カラー
    attachments[0].format         = colorFormat;
    attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // デプス
    attachments[1].format         = VK_FORMAT_D24_UNORM_S8_UINT;
    attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // 前フレーム可視性用の深度キャプチャ
    attachments[2].format         = VK_FORMAT_R32_SFLOAT;
    attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // MotionVector 想定のキャプチャ
    attachments[3].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[3].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[3].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkAttachmentReference prevDepthRef{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference motionRef{3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference colorRefs[3] = {colorRef, prevDepthRef, motionRef};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 3;
    subpass.pColorAttachments       = colorRefs;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // multiview: viewMask=0b11 で左右目を同時描画
    uint32_t viewMask        = 0b11u;
    uint32_t correlationMask = 0b11u;
    VkRenderPassMultiviewCreateInfo multiviewCI{
        VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO};
    multiviewCI.subpassCount         = 1;
    multiviewCI.pViewMasks           = &viewMask;
    multiviewCI.correlationMaskCount = 1;
    multiviewCI.pCorrelationMasks    = &correlationMask;

    VkRenderPassCreateInfo rpCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpCI.pNext           = &multiviewCI;
    rpCI.attachmentCount = 4;
    rpCI.pAttachments    = attachments;
    rpCI.subpassCount    = 1;
    rpCI.pSubpasses      = &subpass;
    rpCI.dependencyCount = 1;
    rpCI.pDependencies   = &dep;
    CHECK_VK(vkCreateRenderPass(app.vk.device, &rpCI, nullptr, &app.vk.renderPass));

    VkAttachmentDescription captureAttachments[4] = {};
    captureAttachments[0] = attachments[0];
    captureAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    captureAttachments[1] = attachments[1];
    captureAttachments[2] = attachments[2];
    captureAttachments[3] = attachments[3];

    VkRenderPassCreateInfo captureRpCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    captureRpCI.pNext           = &multiviewCI;
    captureRpCI.attachmentCount = 4;
    captureRpCI.pAttachments    = captureAttachments;
    captureRpCI.subpassCount    = 1;
    captureRpCI.pSubpasses      = &subpass;
    captureRpCI.dependencyCount = 1;
    captureRpCI.pDependencies   = &dep;
    CHECK_VK(vkCreateRenderPass(app.vk.device, &captureRpCI, nullptr, &app.vk.captureRenderPass));
}

static void CreateQueryPool(App& app) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(app.vk.physDevice, &props);
    app.vk.timestampPeriod = props.limits.timestampPeriod;

    // timestampValidBits が 0 のキューはタイムスタンプ非対応
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(app.vk.physDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(app.vk.physDevice, &qfCount, qfProps.data());
    uint32_t validBits = qfProps[app.vk.queueFamily].timestampValidBits;
    LOGI("timestampPeriod=%.2f ns  timestampValidBits=%u", app.vk.timestampPeriod, validBits);
    if (validBits == 0) {
        LOGW("Timestamp queries not supported on this queue family");
        return;
    }

    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = TS_COUNT;
    CHECK_VK(vkCreateQueryPool(app.vk.device, &ci, nullptr, &app.vk.queryPool));
    LOGI("QueryPool (timestamp x%u) created", TS_COUNT);
}

// ---- VS-inline skinning パイプライン（mode=5）----
// bindings: 0=FatVertex SSBO, 1=bone mat4 SSBO（両方 vertex stage）
static void CreateVsSkinPipeline(App& app, uint32_t width, uint32_t height) {
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t i = 0; i < 2; i++) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.bindingCount = 2;
    dslCI.pBindings    = bindings;
    CHECK_VK(vkCreateDescriptorSetLayout(app.vk.device, &dslCI, nullptr, &app.vk.vsSkinDescLayout));

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCI.maxSets = 1; dpCI.poolSizeCount = 1; dpCI.pPoolSizes = &poolSize;
    CHECK_VK(vkCreateDescriptorPool(app.vk.device, &dpCI, nullptr, &app.vk.vsSkinDescPool));

    VkDescriptorSetAllocateInfo dsAI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAI.descriptorPool = app.vk.vsSkinDescPool;
    dsAI.descriptorSetCount = 1; dsAI.pSetLayouts = &app.vk.vsSkinDescLayout;
    CHECK_VK(vkAllocateDescriptorSets(app.vk.device, &dsAI, &app.vk.vsSkinDescSet));

    VkDescriptorBufferInfo bufInfos[2] = {
        { app.vk.vertexBuffer, 0, VK_WHOLE_SIZE },
        { app.vk.boneBuffer,   0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; i++) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = app.vk.vsSkinDescSet;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufInfos[i];
    }
    vkUpdateDescriptorSets(app.vk.device, 2, writes, 0, nullptr);

    VkPushConstantRange pcRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) * 2 + sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount         = 1;
    plCI.pSetLayouts            = &app.vk.vsSkinDescLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges    = &pcRange;
    CHECK_VK(vkCreatePipelineLayout(app.vk.device, &plCI, nullptr, &app.vk.vsSkinPipelineLayout));

    auto makeShader = [&](const uint32_t* spv, uint32_t size) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = size; ci.pCode = spv;
        VkShaderModule mod;
        CHECK_VK(vkCreateShaderModule(app.vk.device, &ci, nullptr, &mod));
        return mod;
    };
    VkShaderModule vertMod = makeShader(vertex_vs_skin_spv, vertex_vs_skin_spv_size);
    VkShaderModule fragMod = makeShader(fragment_spv,       fragment_spv_size);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo viCI{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo iaCI{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    iaCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport viewport{0.f, 0.f, (float)width, (float)height, 0.f, 1.f};
    VkRect2D   scissor{{0, 0}, {width, height}};
    VkPipelineViewportStateCreateInfo vpCI{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpCI.viewportCount = 1; vpCI.pViewports = &viewport;
    vpCI.scissorCount  = 1; vpCI.pScissors  = &scissor;
    VkPipelineRasterizationStateCreateInfo rsCI{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rsCI.polygonMode = VK_POLYGON_MODE_FILL;
    rsCI.cullMode    = VK_CULL_MODE_NONE;
    rsCI.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsCI.lineWidth   = 1.0f;
    VkPipelineMultisampleStateCreateInfo msCI{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo dsCI{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dsCI.depthTestEnable  = VK_TRUE;
    dsCI.depthWriteEnable = VK_TRUE;
    dsCI.depthCompareOp   = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState cbAtt[3]{};
    cbAtt[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAtt[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    cbAtt[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cbCI{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbCI.attachmentCount = 3; cbCI.pAttachments = cbAtt;

    VkGraphicsPipelineCreateInfo gpCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpCI.stageCount          = 2;
    gpCI.pStages             = stages;
    gpCI.pVertexInputState   = &viCI;
    gpCI.pInputAssemblyState = &iaCI;
    gpCI.pViewportState      = &vpCI;
    gpCI.pRasterizationState = &rsCI;
    gpCI.pMultisampleState   = &msCI;
    gpCI.pDepthStencilState  = &dsCI;
    gpCI.pColorBlendState    = &cbCI;
    gpCI.layout              = app.vk.vsSkinPipelineLayout;
    gpCI.renderPass          = app.vk.renderPass;
    gpCI.subpass             = 0;
    CHECK_VK(vkCreateGraphicsPipelines(
        app.vk.device, VK_NULL_HANDLE, 1, &gpCI, nullptr, &app.vk.vsSkinPipeline));

    cbAtt[0].colorWriteMask = 0;
    cbAtt[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    cbAtt[2].colorWriteMask = 0;
    gpCI.renderPass         = app.vk.captureRenderPass;
    CHECK_VK(vkCreateGraphicsPipelines(
        app.vk.device, VK_NULL_HANDLE, 1, &gpCI, nullptr, &app.vk.vsSkinCapturePipeline));

    vkDestroyShaderModule(app.vk.device, vertMod, nullptr);
    vkDestroyShaderModule(app.vk.device, fragMod, nullptr);
    LOGI("GraphicsPipeline (VS-skin mode=5) created");
}

// ---- mode=6: CS fused skin+cull pipeline ----
// ---- mode=7: CS LDS skin+cull pipeline（1 workgroup = 1 meshlet）----
// bindings: 0=meshletBuf, 1=FatVertex, 2=BoneMatrix, 3=srcIndex, 4=dstIndex, 5=DrawCmd
static void CreateSkinCullLdsPipeline(App& app) {
    VkDescriptorSetLayoutBinding bindings[7]{};
    for (uint32_t i = 0; i < 6; i++) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[6].binding         = 6;
    bindings[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.bindingCount = 7;
    dslCI.pBindings    = bindings;
    CHECK_VK(vkCreateDescriptorSetLayout(app.vk.device, &dslCI, nullptr, &app.vk.skinCullLdsDescLayout));

    VkDescriptorPoolSize poolSizes[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
    };
    VkDescriptorPoolCreateInfo dpCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCI.maxSets = 2;
    dpCI.poolSizeCount = 2;
    dpCI.pPoolSizes = poolSizes;
    CHECK_VK(vkCreateDescriptorPool(app.vk.device, &dpCI, nullptr, &app.vk.skinCullLdsDescPool));

    VkSamplerCreateInfo samplerCI{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCI.magFilter = VK_FILTER_NEAREST;
    samplerCI.minFilter = VK_FILTER_NEAREST;
    samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = 0.0f;
    CHECK_VK(vkCreateSampler(app.vk.device, &samplerCI, nullptr, &app.vk.prevDepthSampler));

    samplerCI.maxLod = (float)std::max<int32_t>((int32_t)app.xr.swapchains[0].hiZMipCount - 1, 0);
    CHECK_VK(vkCreateSampler(app.vk.device, &samplerCI, nullptr, &app.vk.hiZSampler));

    VkDescriptorSetLayout setLayouts[2] = {
        app.vk.skinCullLdsDescLayout,
        app.vk.skinCullLdsDescLayout,
    };
    VkDescriptorSetAllocateInfo dsAI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAI.descriptorPool = app.vk.skinCullLdsDescPool;
    dsAI.descriptorSetCount = 2;
    dsAI.pSetLayouts = setLayouts;
    CHECK_VK(vkAllocateDescriptorSets(app.vk.device, &dsAI, app.vk.skinCullLdsDescSets));

    EyeSwapchain& sc = app.xr.swapchains[0];
    for (uint32_t setIdx = 0; setIdx < 2; ++setIdx) {
        VkDescriptorBufferInfo bufInfos[6] = {
        { app.vk.meshletBuffer,      0, VK_WHOLE_SIZE },
        { app.vk.vertexBuffer,       0, VK_WHOLE_SIZE },
        { app.vk.boneBuffer,         0, VK_WHOLE_SIZE },
        { app.vk.indexBuffer,        0, VK_WHOLE_SIZE },
        { app.vk.compactIndexBuffer, 0, VK_WHOLE_SIZE },
        { app.vk.bfDrawCmdBuffer,    0, VK_WHOLE_SIZE },
        };
        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = app.vk.hiZSampler;
        imageInfo.imageView = sc.hiZViews[setIdx];
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[7]{};
        for (uint32_t i = 0; i < 6; i++) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = app.vk.skinCullLdsDescSets[setIdx];
            writes[i].dstBinding      = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &bufInfos[i];
        }
        writes[6].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet          = app.vk.skinCullLdsDescSets[setIdx];
        writes[6].dstBinding      = 6;
        writes[6].descriptorCount = 1;
        writes[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[6].pImageInfo      = &imageInfo;
        vkUpdateDescriptorSets(app.vk.device, 7, writes, 0, nullptr);
    }

    // push constant: mat4[2] + vec4[2] + uvec4 + vec4 = 192 bytes
    VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(glm::mat4) * 2 + sizeof(glm::vec4) * 2 + sizeof(uint32_t) * 4 + sizeof(glm::vec4)};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1; plCI.pSetLayouts = &app.vk.skinCullLdsDescLayout;
    plCI.pushConstantRangeCount = 1; plCI.pPushConstantRanges = &pcRange;
    CHECK_VK(vkCreatePipelineLayout(app.vk.device, &plCI, nullptr, &app.vk.skinCullLdsPipelineLayout));

    VkShaderModuleCreateInfo smCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smCI.codeSize = skin_cull_lds_spv_size;
    smCI.pCode    = skin_cull_lds_spv;
    VkShaderModule mod;
    CHECK_VK(vkCreateShaderModule(app.vk.device, &smCI, nullptr, &mod));

    VkComputePipelineCreateInfo cpCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroupSizeCI{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
    subgroupSizeCI.requiredSubgroupSize = 64;
    cpCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpCI.stage.module = mod;
    cpCI.stage.pName  = "main";
    cpCI.stage.flags  = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    cpCI.stage.pNext  = &subgroupSizeCI;
    cpCI.layout       = app.vk.skinCullLdsPipelineLayout;
    CHECK_VK(vkCreateComputePipelines(app.vk.device, VK_NULL_HANDLE, 1, &cpCI, nullptr, &app.vk.skinCullLdsPipeline));
    vkDestroyShaderModule(app.vk.device, mod, nullptr);
    LOGI("ComputePipeline (skin+cull LDS mode=7) created, meshlets=%u", app.vk.meshletCount);
}

static void CreateHiZSpdPipeline(App& app) {
    EyeSwapchain& sc = app.xr.swapchains[0];
    if (sc.hiZMipCount > HIZ_SPD_MAX_MIPS) {
        LOGE("Hi-Z mip count %u exceeds SPD limit %u", sc.hiZMipCount, HIZ_SPD_MAX_MIPS);
        assert(false);
    }

    CreateBuffer(app.vk, sizeof(uint32_t) * 2u,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app.vk.hiZSpdAtomicBuffer, app.vk.hiZSpdAtomicMemory);
    {
        void* p = nullptr;
        vkMapMemory(app.vk.device, app.vk.hiZSpdAtomicMemory, 0, sizeof(uint32_t) * 2u, 0, &p);
        std::memset(p, 0, sizeof(uint32_t) * 2u);
        vkUnmapMemory(app.vk.device, app.vk.hiZSpdAtomicMemory);
    }

    VkDescriptorSetLayoutBinding bindings[15]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 2; i < 15; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.bindingCount = 15;
    dslCI.pBindings = bindings;
    CHECK_VK(vkCreateDescriptorSetLayout(app.vk.device, &dslCI, nullptr, &app.vk.hiZSpdDescLayout));

    VkDescriptorPoolSize poolSizes[3] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 13u * 2u},
    };
    VkDescriptorPoolCreateInfo dpCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCI.maxSets = 2;
    dpCI.poolSizeCount = 3;
    dpCI.pPoolSizes = poolSizes;
    CHECK_VK(vkCreateDescriptorPool(app.vk.device, &dpCI, nullptr, &app.vk.hiZSpdDescPool));

    std::vector<VkDescriptorSetLayout> layouts(2, app.vk.hiZSpdDescLayout);
    VkDescriptorSetAllocateInfo dsAI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAI.descriptorPool = app.vk.hiZSpdDescPool;
    dsAI.descriptorSetCount = 2;
    dsAI.pSetLayouts = layouts.data();
    CHECK_VK(vkAllocateDescriptorSets(app.vk.device, &dsAI, app.vk.hiZSpdDescSets));

    for (uint32_t ping = 0; ping < 2; ++ping) {
        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler = app.vk.prevDepthSampler;
        srcInfo.imageView = sc.prevDepthViews[ping];
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorBufferInfo counterInfo{};
        counterInfo.buffer = app.vk.hiZSpdAtomicBuffer;
        counterInfo.offset = 0;
        counterInfo.range = sizeof(uint32_t) * 2u;

        std::vector<VkDescriptorImageInfo> mipInfos(13);
        for (uint32_t i = 0; i < mipInfos.size(); ++i) {
            const uint32_t level = std::min<uint32_t>(i == 0 ? 5u : (i - 1u), sc.hiZMipCount - 1u);
            mipInfos[i].imageView = sc.hiZMipViews[ping][level];
            mipInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        std::vector<VkWriteDescriptorSet> writes(15);
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = app.vk.hiZSpdDescSets[ping];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &srcInfo;

        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = app.vk.hiZSpdDescSets[ping];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &counterInfo;

        for (uint32_t i = 0; i < mipInfos.size(); ++i) {
            writes[i + 2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i + 2].dstSet = app.vk.hiZSpdDescSets[ping];
            writes[i + 2].dstBinding = i + 2;
            writes[i + 2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i + 2].descriptorCount = 1;
            writes[i + 2].pImageInfo = &mipInfos[i];
        }
        vkUpdateDescriptorSets(app.vk.device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &app.vk.hiZSpdDescLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pcRange;
    CHECK_VK(vkCreatePipelineLayout(app.vk.device, &plCI, nullptr, &app.vk.hiZSpdPipelineLayout));

    VkShaderModuleCreateInfo smCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smCI.codeSize = hiz_spd_spv_size;
    smCI.pCode = hiz_spd_spv;
    VkShaderModule mod = VK_NULL_HANDLE;
    CHECK_VK(vkCreateShaderModule(app.vk.device, &smCI, nullptr, &mod));

    VkComputePipelineCreateInfo cpCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpCI.stage.module = mod;
    cpCI.stage.pName = "main";
    cpCI.layout = app.vk.hiZSpdPipelineLayout;
    CHECK_VK(vkCreateComputePipelines(app.vk.device, VK_NULL_HANDLE, 1, &cpCI, nullptr, &app.vk.hiZSpdPipeline));
    vkDestroyShaderModule(app.vk.device, mod, nullptr);
}

// ---- ImageView / Framebuffer / CommandBuffer を作成（multiview: 1 swapchain） ----
static void CreateFrameResources(App& app, VkFormat colorFormat) {
    // コマンドプール
    VkCommandPoolCreateInfo cpCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpCI.queueFamilyIndex = app.vk.queueFamily;
    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK_VK(vkCreateCommandPool(app.vk.device, &cpCI, nullptr, &app.vk.cmdPool));

    // multiview では swapchain は1つ（arraySize=2）
    EyeSwapchain& sc = app.xr.swapchains[0];
    uint32_t imgCount = (uint32_t)sc.images.size();
    sc.hiZW = (sc.width + 1u) / 2u;
    sc.hiZH = (sc.height + 1u) / 2u;
    sc.hiZMipCount = 0;
    for (uint32_t w = sc.hiZW, h = sc.hiZH; w > 0 && h > 0; w = (w + 1u) / 2u, h = (h + 1u) / 2u) {
        sc.hiZMipCount++;
        if (w == 1u && h == 1u) break;
    }

    // デプスバッファ（arrayLayers=2 で両目分）
    CreateImage(app.vk, sc.width, sc.height,
                VK_FORMAT_D24_UNORM_S8_UINT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                sc.depthImage, sc.depthMemory,
                2 /* arrayLayers=2 */);

    VkImageViewCreateInfo dvCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dvCI.image    = sc.depthImage;
    dvCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    dvCI.format   = VK_FORMAT_D24_UNORM_S8_UINT;
    dvCI.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                             0, 1, 0, 2 /* layerCount=2 */};
    CHECK_VK(vkCreateImageView(app.vk.device, &dvCI, nullptr, &sc.depthView));

    for (uint32_t ping = 0; ping < 2; ++ping) {
        CreateImage(app.vk, sc.width, sc.height,
                    VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    sc.prevDepthImages[ping], sc.prevDepthMemories[ping],
                    2 /* arrayLayers=2 */);

        VkImageViewCreateInfo pvCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        pvCI.image    = sc.prevDepthImages[ping];
        pvCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        pvCI.format   = VK_FORMAT_R32_SFLOAT;
        pvCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
        CHECK_VK(vkCreateImageView(app.vk.device, &pvCI, nullptr, &sc.prevDepthViews[ping]));

        CreateImage(app.vk, sc.width, sc.height,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    sc.motionImages[ping], sc.motionMemories[ping],
                    2 /* arrayLayers=2 */);
        pvCI.image  = sc.motionImages[ping];
        pvCI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        CHECK_VK(vkCreateImageView(app.vk.device, &pvCI, nullptr, &sc.motionViews[ping]));

        CreateImage(app.vk, sc.hiZW, sc.hiZH,
                    VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    sc.hiZImages[ping], sc.hiZMemories[ping],
                    2 /* arrayLayers=2 */, sc.hiZMipCount);
        sc.hiZViews[ping] = CreateImageView(
            app.vk, sc.hiZImages[ping], VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, sc.hiZMipCount, 0, 2);
        sc.hiZMipViews[ping].resize(sc.hiZMipCount);
        for (uint32_t level = 0; level < sc.hiZMipCount; ++level) {
            sc.hiZMipViews[ping][level] = CreateImageView(
                app.vk, sc.hiZImages[ping], VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_VIEW_TYPE_2D_ARRAY, level, 1, 0, 2);
        }

    }

    // 各スワップチェーン画像の ImageView + Framebuffer
    for (uint32_t i = 0; i < imgCount; i++) {
        SwapchainImage& si = sc.images[i];

        // color: 2D_ARRAY（arraySize=2 の swapchain image）
        VkImageViewCreateInfo civCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        civCI.image    = si.image;
        civCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        civCI.format   = colorFormat;
        civCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2 /* layerCount=2 */};
        CHECK_VK(vkCreateImageView(app.vk.device, &civCI, nullptr, &si.view));

        for (uint32_t ping = 0; ping < 2; ++ping) {
            VkImageView fbAttachments[] = {si.view, sc.depthView, sc.prevDepthViews[ping], sc.motionViews[ping]};
            VkFramebufferCreateInfo fbCI{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fbCI.renderPass      = app.vk.renderPass;
            fbCI.attachmentCount = 4;
            fbCI.pAttachments    = fbAttachments;
            fbCI.width           = sc.width;
            fbCI.height          = sc.height;
            fbCI.layers          = 1; // multiview では layers=1
            CHECK_VK(vkCreateFramebuffer(app.vk.device, &fbCI, nullptr, &si.framebuffers[ping]));
        }
    }

    // コマンドバッファ（imageIdx の1次元）
    app.vk.cmdBuffers.resize(imgCount);
    VkCommandBufferAllocateInfo cbAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool        = app.vk.cmdPool;
    cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = imgCount;
    CHECK_VK(vkAllocateCommandBuffers(app.vk.device, &cbAI, app.vk.cmdBuffers.data()));

    // フェンス
    VkFenceCreateInfo fCI{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    CHECK_VK(vkCreateFence(app.vk.device, &fCI, nullptr, &app.vk.fence));

    for (uint32_t ping = 0; ping < 2; ++ping) {
        TransitionImageLayoutNow(app.vk, sc.hiZImages[ping], VK_IMAGE_ASPECT_COLOR_BIT,
                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 2, sc.hiZMipCount);
    }
}

// ---- 全体初期化 ----
static void Initialize(App& app) {
    // OpenXR ローダー初期化（Android では必須 / xrCreateInstance より前）
    InitializeOpenXRLoader(app);

    // OpenXR インスタンス
    CreateXrInstance(app);

    // システム取得
    XrSystemGetInfo sgi{XR_TYPE_SYSTEM_GET_INFO};
    sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    CHECK_XR(xrGetSystem(app.xr.instance, &sgi, &app.xr.systemId));

    // Vulkan デバイス（OpenXR 経由）
    CreateVulkanDevice(app);

    // XrSession
    CreateXrSession(app);

    // スワップチェーン
    CreateSwapchains(app);

    // スワップチェーンのカラーフォーマット（最初の目から取得）
    // ※ xrEnumerateSwapchainFormats で選んだ値を保存するため、再取得は不要
    // ここでは RGBA8_SRGB を前提とする（上記 CreateSwapchains で選択済み）
    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_SRGB;

    // Vulkan リソース
    CreateRenderPass(app, colorFormat);
    uint32_t w = app.xr.swapchains[0].width;
    uint32_t h = app.xr.swapchains[0].height;
    CreateFrameResources(app, colorFormat);
    GenerateMultiCubeMesh(app.vk, app.gridN, app.cubeCount);
    CreateVsSkinPipeline(app, w, h);
    CreateSkinCullLdsPipeline(app);
    CreateHiZSpdPipeline(app);
    CreateQueryPool(app);

    app.initialized = true;
    app.lastLogTime = App::Clock::now();
    app.startTime   = app.lastLogTime;
    uint32_t computeGroups = (app.vk.meshletCount + 63) / 64;
    LOGI("=== Initialization complete. GRID_N=%d cubes=%d meshlets=%u computeGroups=%u ===",
         app.gridN, app.cubeCount, app.vk.meshletCount, computeGroups);
}

// モデル行列（頂点は既に world 空間に配置済みなので identity）
static glm::mat4 CubeModelMatrix() {
    return glm::mat4(1.f);
}

// ---- XrView から MVP 行列を計算 ----
static glm::mat4 ComputeMVP(const XrView& view, const glm::mat4& model) {
    // Projection（非対称フラスタム）
    const XrFovf& fov = view.fov;
    float l = tanf(fov.angleLeft);
    float r = tanf(fov.angleRight);
    float d = tanf(fov.angleDown);
    float u = tanf(fov.angleUp);
    float zNear = 0.05f, zFar = 100.f;
    glm::mat4 proj(0.f);
    proj[0][0] = 2.f / (r - l);
    proj[1][1] = 2.f / (u - d);
    proj[2][0] = (r + l) / (r - l);
    proj[2][1] = (u + d) / (u - d);
    proj[2][2] = -(zFar + zNear) / (zFar - zNear);
    proj[2][3] = -1.f;
    proj[3][2] = -(2.f * zFar * zNear) / (zFar - zNear);

    // View（頭のpose）
    const XrQuaternionf& q = view.pose.orientation;
    const XrVector3f&    p = view.pose.position;
    glm::mat4 rot   = glm::mat4_cast(glm::quat(q.w, q.x, q.y, q.z));
    glm::mat4 trans = glm::translate(glm::mat4(1.f), glm::vec3(p.x, p.y, p.z));
    glm::mat4 view_mat = glm::inverse(trans * rot);

    return proj * view_mat * model;
}

// ---- 両目をまとめて描画（multiview: 1 draw call で左右同時） ----
static void RenderStereo(App& app, uint32_t imageIdx,
                         const std::vector<XrView>& views,
                         uint32_t swapW, uint32_t swapH) {
    const uint32_t prevDepthReadIdx  = app.frameParity & 1u;
    const uint32_t prevDepthWriteIdx = prevDepthReadIdx ^ 1u;
    VkCommandBuffer cmd = app.vk.cmdBuffers[imageIdx];
    VkFramebuffer   fb  = app.xr.swapchains[0].images[imageIdx].framebuffers[prevDepthWriteIdx];

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(cmd, &bi));

    glm::mat4 model = CubeModelMatrix();
    glm::mat4 mvp0  = ComputeMVP(views[0], model);
    glm::mat4 mvp1  = ComputeMVP(views[1], model);

    // タイムスタンプリセット（queryPool が有効なときのみ）
    if (app.vk.queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, app.vk.queryPool, 0, TS_COUNT);
    }

    // t0: CS 開始直前
    if (app.vk.queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            app.vk.queryPool, TS_CS_BEGIN);
    }

    // ---- Compute pass: mode=7 CS LDS skin+cull ----
    // drawCmd 全体を毎フレーム初期化（indexCount=0 / instanceCount は動的）
    uint32_t drawCmdInit[5] = {
        0,
        (uint32_t)app.instCount,
        0,
        0,
        0,
    };
    vkCmdUpdateBuffer(cmd, app.vk.bfDrawCmdBuffer, 0, sizeof(drawCmdInit), drawCmdInit);

    VkMemoryBarrier fillToCs{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    fillToCs.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fillToCs.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &fillToCs, 0, nullptr, 0, nullptr);

    // ---- mode=7: LBS + backface cull、LDS で頂点共有（1 wg = 1 meshlet）----
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, app.vk.skinCullLdsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            app.vk.skinCullLdsPipelineLayout, 0, 1,
                            &app.vk.skinCullLdsDescSets[prevDepthReadIdx], 0, nullptr);

    struct SkinCullLdsPC {
        glm::mat4 mvp[2];
        glm::vec4 camPos[2];
        uint32_t  meshletCount;
        uint32_t  cullEnabled;
        uint32_t  frustumEnabled;
        uint32_t  prevDepthEnabled;
        glm::vec4 prevDepthParams;
    };
    SkinCullLdsPC lpc{};
    lpc.mvp[0]       = mvp0;
    lpc.mvp[1]       = mvp1;
    lpc.camPos[0]    = glm::vec4(views[0].pose.position.x,
                                 views[0].pose.position.y,
                                 views[0].pose.position.z, 0.0f);
    lpc.camPos[1]    = glm::vec4(views[1].pose.position.x,
                                 views[1].pose.position.y,
                                 views[1].pose.position.z, 0.0f);
    lpc.meshletCount = app.vk.meshletCount;
    lpc.cullEnabled  = 1u;
    lpc.frustumEnabled = (uint32_t)app.mode7Frustum;
    lpc.prevDepthEnabled = app.prevDepthValid ? 1u : 0u;
    lpc.prevDepthParams = glm::vec4((float)swapW, (float)swapH, 0.02f,
                                    (float)app.xr.swapchains[0].hiZMipCount);
    vkCmdPushConstants(cmd, app.vk.skinCullLdsPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lpc), &lpc);
    if (app.prevDepthValid) {
        VkImageMemoryBarrier prevDepthToRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        prevDepthToRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        prevDepthToRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        prevDepthToRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        prevDepthToRead.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        prevDepthToRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prevDepthToRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prevDepthToRead.image = app.xr.swapchains[0].hiZImages[prevDepthReadIdx];
        prevDepthToRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                            app.xr.swapchains[0].hiZMipCount, 0, 2};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &prevDepthToRead);
    }
    vkCmdDispatch(cmd, app.vk.meshletCount, 1, 1);

    // t1: CS 終了（dispatch 完了後）
    if (app.vk.queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            app.vk.queryPool, TS_CS_END);
    }

    // compute → indirect + index read
    VkMemoryBarrier csToDraw{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    csToDraw.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    csToDraw.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        0, 1, &csToDraw, 0, nullptr, 0, nullptr);

    // t2: Graphics pass 開始直前
    if (app.vk.queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            app.vk.queryPool, TS_GFX_BEGIN);
    }

    // ---- Graphics pass ----
    VkClearValue clears[2];
    clears[0].color        = {0.1f, 0.1f, 0.15f, 1.0f};
    clears[1].depthStencil = {1.0f, 0};
    VkClearValue prevDepthClear{};
    prevDepthClear.color.float32[0] = 1.0f;
    VkClearValue motionClear{};
    motionClear.color.float32[0] = 0.0f;
    motionClear.color.float32[1] = 0.0f;
    motionClear.color.float32[2] = 0.0f;
    motionClear.color.float32[3] = 0.0f;
    VkClearValue clears4[4] = {clears[0], clears[1], prevDepthClear, motionClear};

    VkRenderPassBeginInfo rpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBI.renderPass        = app.vk.renderPass;
    rpBI.framebuffer       = fb;
    rpBI.renderArea.offset = {0, 0};
    rpBI.renderArea.extent = {swapW, swapH};
    rpBI.clearValueCount   = 4;
    rpBI.pClearValues      = clears4;

    vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

    struct GfxPushConst { glm::mat4 mvp[2]; int32_t aluIters; };
    GfxPushConst pc;
    pc.mvp[0]   = mvp0;
    pc.mvp[1]   = mvp1;
    pc.aluIters = (int32_t)app.aluIters;

    // mode=7: FatVertex+Bone SSBO を VS で読んで LBS
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, app.vk.vsSkinPipeline);
    vkCmdPushConstants(cmd, app.vk.vsSkinPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            app.vk.vsSkinPipelineLayout, 0, 1,
                            &app.vk.vsSkinDescSet, 0, nullptr);

    // compactIndex + bfDrawCmd による indirect draw (mode=7)
    vkCmdBindIndexBuffer(cmd, app.vk.compactIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(cmd, app.vk.bfDrawCmdBuffer, 0, 1, 0);

    vkCmdEndRenderPass(cmd);

    VkMemoryBarrier gfxToCapture{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    gfxToCapture.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    gfxToCapture.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0, 1, &gfxToCapture, 0, nullptr, 0, nullptr);

    VkRenderPassBeginInfo captureRpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    captureRpBI.renderPass        = app.vk.captureRenderPass;
    captureRpBI.framebuffer       = fb;
    captureRpBI.renderArea.offset = {0, 0};
    captureRpBI.renderArea.extent = {swapW, swapH};
    captureRpBI.clearValueCount   = 4;
    captureRpBI.pClearValues      = clears4;

    vkCmdBeginRenderPass(cmd, &captureRpBI, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, app.vk.vsSkinCapturePipeline);
    vkCmdPushConstants(cmd, app.vk.vsSkinPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            app.vk.vsSkinPipelineLayout, 0, 1,
                            &app.vk.vsSkinDescSet, 0, nullptr);
    vkCmdBindIndexBuffer(cmd, app.vk.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, app.vk.indexCount, (uint32_t)app.instCount, 0, 0, 0);
    vkCmdEndRenderPass(cmd);

    if (app.vk.queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            app.vk.queryPool, TS_GFX_END);
    }

    VkImageMemoryBarrier downsampleSrcBarriers[1]{};
    for (uint32_t i = 0; i < 1; ++i) {
        downsampleSrcBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        downsampleSrcBarriers[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        downsampleSrcBarriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        downsampleSrcBarriers[i].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        downsampleSrcBarriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        downsampleSrcBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        downsampleSrcBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        downsampleSrcBarriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
    }
    downsampleSrcBarriers[0].image = app.xr.swapchains[0].prevDepthImages[prevDepthWriteIdx];
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, downsampleSrcBarriers);

    struct HiZSpdPC {
        uint32_t mips;
        uint32_t numWorkGroups;
        uint32_t srcWidth;
        uint32_t srcHeight;
    } spdPc{};
    const uint32_t spdDispatchX = (app.xr.swapchains[0].width + 63u) / 64u;
    const uint32_t spdDispatchY = (app.xr.swapchains[0].height + 63u) / 64u;
    spdPc.mips = app.xr.swapchains[0].hiZMipCount;
    spdPc.numWorkGroups = spdDispatchX * spdDispatchY;
    spdPc.srcWidth = app.xr.swapchains[0].width;
    spdPc.srcHeight = app.xr.swapchains[0].height;

    if (app.vk.queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            app.vk.queryPool, TS_HIZ_BEGIN);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, app.vk.hiZSpdPipeline);
    VkDescriptorSet hiZSpdDs = app.vk.hiZSpdDescSets[prevDepthWriteIdx];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            app.vk.hiZSpdPipelineLayout, 0, 1, &hiZSpdDs, 0, nullptr);
    vkCmdPushConstants(cmd, app.vk.hiZSpdPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(spdPc), &spdPc);
    vkCmdDispatch(cmd, spdDispatchX, spdDispatchY, 2u);

    VkImageMemoryBarrier hiZBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    hiZBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hiZBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    hiZBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    hiZBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    hiZBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hiZBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hiZBarrier.image = app.xr.swapchains[0].hiZImages[prevDepthWriteIdx];
    hiZBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, app.xr.swapchains[0].hiZMipCount, 0, 2};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &hiZBarrier);

    if (app.vk.queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            app.vk.queryPool, TS_HIZ_END);
    }

    CHECK_VK(vkEndCommandBuffer(cmd));
}

// ---- 1 フレームをレンダリング ----
static void RenderFrame(App& app) {
    using Clock = App::Clock;
    auto t0 = Clock::now();

    XrFrameWaitInfo fwi{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState    fst{XR_TYPE_FRAME_STATE};
    CHECK_XR(xrWaitFrame(app.xr.session, &fwi, &fst));
    CHECK_XR(xrBeginFrame(app.xr.session, nullptr));

    std::vector<XrCompositionLayerBaseHeader*> layers;
    XrCompositionLayerProjection projLayer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    std::vector<XrCompositionLayerProjectionView> projViews;

    if (fst.shouldRender) {
        // ビューのロケーション取得（PRIMARY_STEREO は常に2ビュー）
        uint32_t viewCount = 2;
        std::vector<XrView> views(viewCount, {XR_TYPE_VIEW});
        XrViewLocateInfo vli{XR_TYPE_VIEW_LOCATE_INFO};
        vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        vli.displayTime           = fst.predictedDisplayTime;
        vli.space                 = app.xr.appSpace;
        XrViewState vs{XR_TYPE_VIEW_STATE};
        CHECK_XR(xrLocateViews(app.xr.session, &vli, &vs,
                               viewCount, &viewCount, views.data()));
        if (!app.frozenViewsValid && viewCount == 2) {
            app.frozenViews[0] = views[0];
            app.frozenViews[1] = views[1];
            app.frozenViewsValid = true;
            LOGI("Frozen render views captured");
        }
        const XrView* renderViews = app.frozenViewsValid ? app.frozenViews : views.data();

        projViews.resize(viewCount);

        // multiview: 1つの swapchain から画像を取得して両目を1 draw call で描画
        EyeSwapchain& sc = app.xr.swapchains[0];

        XrSwapchainImageAcquireInfo acqInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        uint32_t imgIdx = 0;
        CHECK_XR(xrAcquireSwapchainImage(sc.handle, &acqInfo, &imgIdx));

        XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        waitInfo.timeout = XR_INFINITE_DURATION;
        CHECK_XR(xrWaitSwapchainImage(sc.handle, &waitInfo));

        // 両目まとめて描画
        std::vector<XrView> renderViewsVec(viewCount, {XR_TYPE_VIEW});
        for (uint32_t eye = 0; eye < viewCount; ++eye) {
            renderViewsVec[eye] = renderViews[eye];
        }
        RenderStereo(app, imgIdx, renderViewsVec, sc.width, sc.height);

        // ProjectionView を設定（左右眼それぞれ、arrayIndex で layer を指定）
        for (uint32_t eye = 0; eye < viewCount; eye++) {
            projViews[eye] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            projViews[eye].pose = renderViews[eye].pose;
            projViews[eye].fov  = renderViews[eye].fov;
            projViews[eye].subImage.swapchain             = sc.handle;
            projViews[eye].subImage.imageRect.offset      = {0, 0};
            projViews[eye].subImage.imageRect.extent      = {(int32_t)sc.width, (int32_t)sc.height};
            projViews[eye].subImage.imageArrayIndex       = eye; // 0=左, 1=右
        }

        // フェンス待機 & リセット
        vkWaitForFences(app.vk.device, 1, &app.vk.fence, VK_TRUE, UINT64_MAX);

        // ---- CPU ボーン行列更新（前フレームの GPU 使用が完了した後）----
        {
            double elapsed = std::chrono::duration<double>(
                App::Clock::now() - app.startTime).count();
            UpdateBones(app.vk, (float)elapsed);
        }

        // タイムスタンプ読み出し
        if (app.vk.queryPool != VK_NULL_HANDLE) {
            uint64_t ts[TS_COUNT] = {};
            VkResult qr = vkGetQueryPoolResults(
                app.vk.device, app.vk.queryPool, 0, TS_COUNT,
                sizeof(ts), ts, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT);
            if (qr == VK_SUCCESS) {
                double nsPerTick = (double)app.vk.timestampPeriod;
                app.lastCsMs  = (ts[TS_CS_END]  - ts[TS_CS_BEGIN])  * nsPerTick * 1e-6;
                app.lastGfxMs = (ts[TS_GFX_END] - ts[TS_GFX_BEGIN]) * nsPerTick * 1e-6;
                app.lastHiZMs = (ts[TS_HIZ_END] - ts[TS_HIZ_BEGIN]) * nsPerTick * 1e-6;
                app.lastDownsampleMs = app.lastHiZMs;
            }
        }

        // backface 生存 index 数を CPU に読み出す（直前フレームの結果）
        {
            void* p = nullptr;
            vkMapMemory(app.vk.device, app.vk.bfDrawCmdMemory, 0,
                        sizeof(uint32_t), 0, &p);
            app.lastBfIndexCount = *reinterpret_cast<uint32_t*>(p);
            vkUnmapMemory(app.vk.device, app.vk.bfDrawCmdMemory);
        }
        vkResetFences(app.vk.device, 1, &app.vk.fence);

        // キューサブミット（コマンドバッファ1つ）
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &app.vk.cmdBuffers[imgIdx];
        CHECK_VK(vkQueueSubmit(app.vk.queue, 1, &si, app.vk.fence));
        app.prevDepthValid = true;
        app.frameParity ^= 1u;

        // スワップチェーン画像を解放
        XrSwapchainImageReleaseInfo relInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        CHECK_XR(xrReleaseSwapchainImage(sc.handle, &relInfo));

        projLayer.space                = app.xr.appSpace;
        projLayer.viewCount            = viewCount;
        projLayer.views                = projViews.data();
        layers.push_back((XrCompositionLayerBaseHeader*)&projLayer);
    }

    XrFrameEndInfo fei{XR_TYPE_FRAME_END_INFO};
    fei.displayTime          = fst.predictedDisplayTime;
    fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    fei.layerCount           = (uint32_t)layers.size();
    fei.layers               = layers.data();
    CHECK_XR(xrEndFrame(app.xr.session, &fei));

    // フレーム時間計測
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    app.frameMsAccum += ms;
    app.frameCount++;

    // 毎秒ログ出力
    double elapsed = std::chrono::duration<double>(t1 - app.lastLogTime).count();
    if (elapsed >= 1.0) {
        double avgMs    = app.frameMsAccum / app.frameCount;
        double fps      = app.frameCount / elapsed;
        int    polysTotal = (int)app.vk.meshletCount * MESHLET_TILE * MESHLET_TILE * 2 * app.instCount;
        uint32_t liveTris = app.lastBfIndexCount / 3;
        LOGI("[PERF] FPS=%.1f  FrameTime=%.2fms  CS=%.3fms  GFX=%.3fms  DS=%.3fms  HiZ=%.3fms  Polys=%d  LiveTris=%u  Meshlets=%u",
             fps, avgMs, app.lastCsMs, app.lastGfxMs, app.lastDownsampleMs, app.lastHiZMs,
             polysTotal, liveTris, app.vk.meshletCount);
        app.frameMsAccum = 0.0;
        app.frameCount   = 0;
        app.lastLogTime  = t1;
    }
}

// ---- OpenXR イベント処理 ----
static void HandleXrEvents(App& app) {
    XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(app.xr.instance, &ev) == XR_SUCCESS) {
        switch (ev.type) {
        case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
            auto* sce = (XrEventDataSessionStateChanged*)&ev;
            app.xr.sessionState = sce->state;
            LOGI("SessionState -> %d", (int)sce->state);

            if (sce->state == XR_SESSION_STATE_READY) {
                XrSessionBeginInfo sbi{XR_TYPE_SESSION_BEGIN_INFO};
                sbi.primaryViewConfigurationType =
                    XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                CHECK_XR(xrBeginSession(app.xr.session, &sbi));
                app.xr.sessionRunning = true;
                LOGI("Session RUNNING");
            } else if (sce->state == XR_SESSION_STATE_STOPPING) {
                CHECK_XR(xrEndSession(app.xr.session));
                app.xr.sessionRunning = false;
            } else if (sce->state == XR_SESSION_STATE_EXITING ||
                       sce->state == XR_SESSION_STATE_LOSS_PENDING) {
                app.xr.exitRequested = true;
            }
            break;
        }
        case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
            app.xr.exitRequested = true;
            break;
        default:
            break;
        }
        ev = {XR_TYPE_EVENT_DATA_BUFFER};
    }
}

// ---- Android イベント処理コールバック ----
static void HandleAndroidCmd(android_app* aApp, int32_t cmd) {
    switch (cmd) {
    case APP_CMD_INIT_WINDOW:
        LOGI("APP_CMD_INIT_WINDOW");
        break;
    case APP_CMD_TERM_WINDOW:
        LOGI("APP_CMD_TERM_WINDOW");
        break;
    default:
        break;
    }
}

// ---- クリーンアップ ----
static void Cleanup(App& app) {
    if (app.vk.device) {
        vkDeviceWaitIdle(app.vk.device);

        if (app.vk.fence)      vkDestroyFence(app.vk.device, app.vk.fence, nullptr);
        if (app.vk.captureRenderPass) vkDestroyRenderPass(app.vk.device, app.vk.captureRenderPass, nullptr);
        if (app.vk.renderPass) vkDestroyRenderPass(app.vk.device, app.vk.renderPass, nullptr);

        if (app.vk.vertexBuffer)   vkDestroyBuffer(app.vk.device, app.vk.vertexBuffer, nullptr);
        if (app.vk.vertexMemory)   vkFreeMemory(app.vk.device, app.vk.vertexMemory, nullptr);
        if (app.vk.indexBuffer)    vkDestroyBuffer(app.vk.device, app.vk.indexBuffer, nullptr);
        if (app.vk.indexMemory)    vkFreeMemory(app.vk.device, app.vk.indexMemory, nullptr);

        if (app.vk.meshletBuffer)  vkDestroyBuffer(app.vk.device, app.vk.meshletBuffer, nullptr);
        if (app.vk.meshletMemory)  vkFreeMemory(app.vk.device, app.vk.meshletMemory, nullptr);

        if (app.vk.queryPool)          vkDestroyQueryPool(app.vk.device, app.vk.queryPool, nullptr);

        if (app.vk.compactIndexBuffer) vkDestroyBuffer(app.vk.device, app.vk.compactIndexBuffer, nullptr);
        if (app.vk.compactIndexMemory) vkFreeMemory(app.vk.device, app.vk.compactIndexMemory, nullptr);
        if (app.vk.bfDrawCmdBuffer)    vkDestroyBuffer(app.vk.device, app.vk.bfDrawCmdBuffer, nullptr);
        if (app.vk.bfDrawCmdMemory)    vkFreeMemory(app.vk.device, app.vk.bfDrawCmdMemory, nullptr);

        if (app.vk.boneBuffer)             vkDestroyBuffer(app.vk.device, app.vk.boneBuffer, nullptr);
        if (app.vk.boneMemory)             vkFreeMemory(app.vk.device, app.vk.boneMemory, nullptr);

        if (app.vk.vsSkinPipeline)        vkDestroyPipeline(app.vk.device, app.vk.vsSkinPipeline, nullptr);
        if (app.vk.vsSkinCapturePipeline) vkDestroyPipeline(app.vk.device, app.vk.vsSkinCapturePipeline, nullptr);
        if (app.vk.vsSkinPipelineLayout)  vkDestroyPipelineLayout(app.vk.device, app.vk.vsSkinPipelineLayout, nullptr);
        if (app.vk.vsSkinDescPool)        vkDestroyDescriptorPool(app.vk.device, app.vk.vsSkinDescPool, nullptr);
        if (app.vk.vsSkinDescLayout)      vkDestroyDescriptorSetLayout(app.vk.device, app.vk.vsSkinDescLayout, nullptr);

        if (app.vk.skinCullLdsPipeline)        vkDestroyPipeline(app.vk.device, app.vk.skinCullLdsPipeline, nullptr);
        if (app.vk.skinCullLdsPipelineLayout)  vkDestroyPipelineLayout(app.vk.device, app.vk.skinCullLdsPipelineLayout, nullptr);
        if (app.vk.skinCullLdsDescPool)        vkDestroyDescriptorPool(app.vk.device, app.vk.skinCullLdsDescPool, nullptr);
        if (app.vk.skinCullLdsDescLayout)      vkDestroyDescriptorSetLayout(app.vk.device, app.vk.skinCullLdsDescLayout, nullptr);
        if (app.vk.hiZSampler)                 vkDestroySampler(app.vk.device, app.vk.hiZSampler, nullptr);
        if (app.vk.prevDepthSampler)           vkDestroySampler(app.vk.device, app.vk.prevDepthSampler, nullptr);
        if (app.vk.hiZSpdPipeline)             vkDestroyPipeline(app.vk.device, app.vk.hiZSpdPipeline, nullptr);
        if (app.vk.hiZSpdPipelineLayout)       vkDestroyPipelineLayout(app.vk.device, app.vk.hiZSpdPipelineLayout, nullptr);
        if (app.vk.hiZSpdDescPool)             vkDestroyDescriptorPool(app.vk.device, app.vk.hiZSpdDescPool, nullptr);
        if (app.vk.hiZSpdDescLayout)           vkDestroyDescriptorSetLayout(app.vk.device, app.vk.hiZSpdDescLayout, nullptr);
        if (app.vk.hiZSpdAtomicBuffer)         vkDestroyBuffer(app.vk.device, app.vk.hiZSpdAtomicBuffer, nullptr);
        if (app.vk.hiZSpdAtomicMemory)         vkFreeMemory(app.vk.device, app.vk.hiZSpdAtomicMemory, nullptr);

        for (auto& sc : app.xr.swapchains) {
            for (auto& si : sc.images) {
                for (VkFramebuffer fb : si.framebuffers) {
                    if (fb) vkDestroyFramebuffer(app.vk.device, fb, nullptr);
                }
                if (si.view)        vkDestroyImageView(app.vk.device, si.view, nullptr);
            }
            for (uint32_t ping = 0; ping < 2; ++ping) {
                if (sc.prevDepthViews[ping])   vkDestroyImageView(app.vk.device, sc.prevDepthViews[ping], nullptr);
                if (sc.prevDepthImages[ping])  vkDestroyImage(app.vk.device, sc.prevDepthImages[ping], nullptr);
                if (sc.prevDepthMemories[ping]) vkFreeMemory(app.vk.device, sc.prevDepthMemories[ping], nullptr);
                if (sc.motionViews[ping])      vkDestroyImageView(app.vk.device, sc.motionViews[ping], nullptr);
                if (sc.motionImages[ping])     vkDestroyImage(app.vk.device, sc.motionImages[ping], nullptr);
                if (sc.motionMemories[ping])   vkFreeMemory(app.vk.device, sc.motionMemories[ping], nullptr);
                for (VkImageView v : sc.hiZMipViews[ping]) {
                    if (v) vkDestroyImageView(app.vk.device, v, nullptr);
                }
                if (sc.hiZViews[ping])         vkDestroyImageView(app.vk.device, sc.hiZViews[ping], nullptr);
                if (sc.hiZImages[ping])        vkDestroyImage(app.vk.device, sc.hiZImages[ping], nullptr);
                if (sc.hiZMemories[ping])      vkFreeMemory(app.vk.device, sc.hiZMemories[ping], nullptr);
            }
            if (sc.depthView)   vkDestroyImageView(app.vk.device, sc.depthView, nullptr);
            if (sc.depthImage)  vkDestroyImage(app.vk.device, sc.depthImage, nullptr);
            if (sc.depthMemory) vkFreeMemory(app.vk.device, sc.depthMemory, nullptr);
        }

        if (app.vk.cmdPool) vkDestroyCommandPool(app.vk.device, app.vk.cmdPool, nullptr);
        vkDestroyDevice(app.vk.device, nullptr);
    }
    if (app.vk.instance) vkDestroyInstance(app.vk.instance, nullptr);

    if (app.xr.appSpace) xrDestroySpace(app.xr.appSpace);
    for (auto& sc : app.xr.swapchains) {
        if (sc.handle) xrDestroySwapchain(sc.handle);
    }
    if (app.xr.session)  xrDestroySession(app.xr.session);
    if (app.xr.instance) xrDestroyInstance(app.xr.instance);
}

// ============================================================
// Intent エクストラからパラメータを取得する
// 使い方:
//   adb shell am start -n com.example.picoperftest/android.app.NativeActivity \
//       --ei grid_n 1584
//   adb shell am start -n com.example.picoperftest/android.app.NativeActivity \
//       --ei grid_n 17 --ei inst_count 10000
// ============================================================
static void GetIntentParams(android_app* aApp, int& outGridN, int& outInstCount, int& outAluIters, int& outCubeCount, int& outMode7Frustum, float& outResScale) {
    JNIEnv* env = nullptr;
    aApp->activity->vm->AttachCurrentThread(&env, nullptr);

    jobject activity   = aApp->activity->clazz;
    jclass  actClass   = env->GetObjectClass(activity);

    jmethodID getIntent   = env->GetMethodID(actClass, "getIntent", "()Landroid/content/Intent;");
    jobject   intent      = env->CallObjectMethod(activity, getIntent);
    jclass    intentClass = env->GetObjectClass(intent);
    jmethodID getIntExtra   = env->GetMethodID(intentClass, "getIntExtra",
                                               "(Ljava/lang/String;I)I");
    jmethodID getFloatExtra = env->GetMethodID(intentClass, "getFloatExtra",
                                               "(Ljava/lang/String;F)F");

    jstring keyGridN = env->NewStringUTF("grid_n");
    jint    gridN    = env->CallIntMethod(intent, getIntExtra, keyGridN, (jint)outGridN);
    env->DeleteLocalRef(keyGridN);

    jstring keyInst  = env->NewStringUTF("inst_count");
    jint    instCount = env->CallIntMethod(intent, getIntExtra, keyInst, (jint)outInstCount);
    env->DeleteLocalRef(keyInst);

    jstring keyAlu   = env->NewStringUTF("alu_iters");
    jint    aluIters = env->CallIntMethod(intent, getIntExtra, keyAlu, (jint)outAluIters);
    env->DeleteLocalRef(keyAlu);

    jstring keyCubes = env->NewStringUTF("num_cubes");
    jint    numCubes = env->CallIntMethod(intent, getIntExtra, keyCubes, (jint)outCubeCount);
    env->DeleteLocalRef(keyCubes);

    jstring keyMode7Frustum = env->NewStringUTF("mode7_frustum");
    jint    mode7Frustum    = env->CallIntMethod(intent, getIntExtra, keyMode7Frustum, (jint)outMode7Frustum);
    env->DeleteLocalRef(keyMode7Frustum);

    jstring keyResScale = env->NewStringUTF("res_scale");
    jfloat  resScale    = env->CallFloatMethod(intent, getFloatExtra, keyResScale, (jfloat)outResScale);
    env->DeleteLocalRef(keyResScale);

    env->DeleteLocalRef(intent);
    env->DeleteLocalRef(actClass);
    aApp->activity->vm->DetachCurrentThread();

    if (gridN <= 1) {
        LOGW("grid_n=%d is too small, using default %d", (int)gridN, outGridN);
    } else {
        outGridN = (int)gridN;
    }
    if (instCount < 1) {
        LOGW("inst_count=%d is invalid, using default %d", (int)instCount, outInstCount);
    } else {
        outInstCount = (int)instCount;
    }
    if (aluIters >= 0) outAluIters = (int)aluIters;
    if (numCubes >= 1) outCubeCount = (int)numCubes;
    if (mode7Frustum == 0 || mode7Frustum == 1) outMode7Frustum = (int)mode7Frustum;
    if (resScale >= 0.25f && resScale <= 1.0f) outResScale = (float)resScale;
}

// ============================================================
// エントリポイント
// ============================================================
void android_main(android_app* aApp) {
    LOGI("android_main: start");
    aApp->onAppCmd = HandleAndroidCmd;

    App app;
    app.androidApp = aApp;

    // Android イベントを一通り処理してから初期化に進む
    // 注意: OpenXR VR アプリはウィンドウサーフェスを使わないので、
    //       aApp->window の待機は不要（かつデバイスが headset モードでは window が来ない場合がある）
    {
        int events;
        android_poll_source* source;
        // 既存のイベントをすべてフラッシュ
        while (ALooper_pollAll(0, nullptr, &events, (void**)&source) >= 0) {
            if (source) source->process(aApp, source);
        }
    }

    if (aApp->destroyRequested) return;

    // Intent エクストラを読んでパラメータを設定
    // num_cubes>1 の検証用途ではデフォルト GRID_N を小さくする（未指定時の保険）
    app.gridN = 97;
    GetIntentParams(aApp, app.gridN, app.instCount, app.aluIters, app.cubeCount, app.mode7Frustum, app.resScale);
    LOGI("grid_n=%d  num_cubes=%d  inst_count=%d  alu_iters=%d  mode7_frustum=%d  res_scale=%.2f",
         app.gridN, app.cubeCount, app.instCount, app.aluIters, app.mode7Frustum, app.resScale);

    // OpenXR ローダーおよびランタイムへの接続に JNI スレッドのアタッチが必要。
    // PICO SDK フレームワーク (BasicOpenXrWrapper) は xrInitializeLoaderKHR の前に
    // AttachCurrentThread を呼び出しており、その後デタッチしない。
    // GetGridNFromIntent では Attach→Detach を行ったので、ここで再アタッチする。
    JNIEnv* jniEnv = nullptr;
    aApp->activity->vm->AttachCurrentThread(&jniEnv, nullptr);

    Initialize(app);

    aApp->activity->vm->DetachCurrentThread();

    LOGI("=== Entering main loop ===");
    while (!aApp->destroyRequested && !app.xr.exitRequested) {
        // Android イベント
        int events;
        android_poll_source* source;
        while (ALooper_pollAll(0, nullptr, &events, (void**)&source) >= 0) {
            if (source) source->process(aApp, source);
            if (aApp->destroyRequested) break;
        }

        // OpenXR イベント
        HandleXrEvents(app);

        // レンダリング
        if (app.xr.sessionRunning) {
            RenderFrame(app);
        }
    }

    LOGI("=== Exiting, cleaning up ===");
    Cleanup(app);
}
