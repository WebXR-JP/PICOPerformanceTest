#pragma once

#ifndef XR_USE_PLATFORM_ANDROID
#define XR_USE_PLATFORM_ANDROID
#endif

#ifndef XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_GRAPHICS_API_VULKAN
#endif

#include <android/log.h>
#include <android_native_app_glue.h>
#include <jni.h>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_VULKAN_VERSION 1001000
#include "vk_mem_alloc.hpp"
#include "vk_mem_alloc_raii.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "PICOPerfTest", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  "PICOPerfTest", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PICOPerfTest", __VA_ARGS__)

#ifndef GRID_N
#define GRID_N 708
#endif

#define CHECK_XR(expr)                                                         \
    do {                                                                       \
        XrResult _r = (expr);                                                  \
        if (XR_FAILED(_r)) {                                                   \
            LOGE("XR error %d at %s:%d", (int)_r, __FILE__, __LINE__);         \
            assert(false);                                                     \
        }                                                                      \
    } while (0)

inline void CheckVkResult(VkResult result) {
    if (result != VK_SUCCESS) {
        LOGE("VK error %d", static_cast<int>(result));
        assert(false);
    }
}

template <typename Handle>
inline auto Raw(const Handle& handle) -> typename Handle::CType {
    return static_cast<typename Handle::CType>(*handle);
}

template <typename Handle>
inline bool IsValid(const Handle& handle) {
    return Raw(handle) != typename Handle::CType{};
}

struct SwapchainImage {
    VkImage           image = VK_NULL_HANDLE;
    vk::raii::ImageView view{nullptr};
};

struct EyeSwapchain {
    XrSwapchain handle = XR_NULL_HANDLE;
    uint32_t    width  = 0;
    uint32_t    height = 0;
    std::vector<SwapchainImage> images;
    vma::raii::Image                     depthImage{nullptr};
    vk::raii::ImageView                  depthView{nullptr};
    vk::raii::ImageView                  depthSampleView{nullptr};
    std::array<vma::raii::Image, 2>      prevDepthImages{nullptr, nullptr};
    std::array<vk::raii::ImageView, 2>   prevDepthViews{nullptr, nullptr};
    std::array<vma::raii::Image, 2>      motionImages{nullptr, nullptr};
    std::array<vk::raii::ImageView, 2>   motionViews{nullptr, nullptr};
    std::array<vma::raii::Image, 2>      hiZImages{nullptr, nullptr};
    std::array<vk::raii::ImageView, 2>   hiZViews{nullptr, nullptr};
    std::array<std::vector<vk::raii::ImageView>, 2> hiZMipViews;
    uint32_t                             hiZW        = 0;
    uint32_t                             hiZH        = 0;
    uint32_t                             hiZMipCount = 0;
};

struct VulkanCtx {
    vk::raii::Context                       context;
    vk::raii::Instance                      instance{nullptr};
    vk::raii::PhysicalDevice                physDevice{nullptr};
    vk::raii::Device                        device{nullptr};
    uint32_t                                queueFamily = 0;
    vk::raii::Queue                         queue{nullptr};
    vk::raii::CommandPool                   cmdPool{nullptr};
    vma::raii::Allocator                    allocator{nullptr};
    VkFormat                                colorFormat = VK_FORMAT_UNDEFINED;
    vma::raii::Buffer                       vertexBuffer{nullptr};
    vma::raii::Buffer                       indexBuffer{nullptr};
    uint32_t                                indexCount  = 0;
    uint32_t                                vertexCount = 0;
    vk::raii::Fence                         fence{nullptr};
    vk::raii::CommandBuffers                cmdBuffers{nullptr};
    vma::raii::Buffer                       meshletBuffer{nullptr};
    uint32_t                                meshletCount = 0;
    vk::raii::QueryPool                     queryPool{nullptr};
    float                                   timestampPeriod = 1.f;
    vma::raii::Buffer                       compactIndexBuffer{nullptr};
    vma::raii::Buffer                       bfDrawCmdBuffer{nullptr};
    vma::raii::Buffer                       cullStatsBuffer{nullptr};
    vma::raii::Buffer                       boneBuffer{nullptr};
    uint32_t                                totalBones = 0;
    uint32_t                                vPerFace = 0;
    uint32_t                                vPerCube = 0;
    std::vector<glm::vec3> bonePivots;
    vk::raii::DescriptorSetLayout           vsSkinDescLayout{nullptr};
    vk::raii::PipelineLayout                vsSkinPipelineLayout{nullptr};
    vk::raii::Pipeline                      vsSkinPipeline{nullptr};
    vk::raii::DescriptorSetLayout           skinCullLdsDescLayout{nullptr};
    vk::raii::PipelineLayout                skinCullLdsPipelineLayout{nullptr};
    vk::raii::Pipeline                      skinCullLdsPipeline{nullptr};
    vk::raii::Sampler                       prevDepthSampler{nullptr};
    vk::raii::Sampler                       hiZSampler{nullptr};
    vma::raii::Buffer                       hiZSpdAtomicBuffer{nullptr};
    vk::raii::DescriptorSetLayout           hiZSpdDescLayout{nullptr};
    vk::raii::PipelineLayout                hiZSpdPipelineLayout{nullptr};
    vk::raii::Pipeline                      hiZSpdPipeline{nullptr};
    vk::raii::DescriptorSetLayout           meshletDebugDescLayout{nullptr};
    vk::raii::PipelineLayout                meshletDebugPipelineLayout{nullptr};
    vk::raii::Pipeline                      meshletDebugPipeline{nullptr};
};

inline constexpr uint32_t TS_CS_BEGIN      = 0;
inline constexpr uint32_t TS_CS_END        = 1;
inline constexpr uint32_t TS_GFX_BEGIN     = 2;
inline constexpr uint32_t TS_GFX_END       = 3;
inline constexpr uint32_t TS_HIZ_BEGIN     = 4;
inline constexpr uint32_t TS_HIZ_END       = 5;
inline constexpr uint32_t TS_COUNT         = 6;
inline constexpr uint32_t HIZ_SPD_MAX_MIPS = 12;

struct XrCtx {
    XrInstance     instance       = XR_NULL_HANDLE;
    XrSystemId     systemId       = XR_NULL_SYSTEM_ID;
    XrSession      session        = XR_NULL_HANDLE;
    XrSpace        appSpace       = XR_NULL_HANDLE;
    XrSessionState sessionState   = XR_SESSION_STATE_UNKNOWN;
    bool           sessionRunning = false;
    bool           exitRequested  = false;
    std::vector<EyeSwapchain> swapchains;

    XrActionSet    actionSet      = XR_NULL_HANDLE;
    XrAction       moveAction     = XR_NULL_HANDLE; // vec2f: x=strafe, y=forward
    XrAction       turnAction     = XR_NULL_HANDLE; // vec2f: x=yaw
};

extern PFN_xrGetVulkanGraphicsRequirements2KHR pfn_xrGetVulkanGraphicsRequirements2KHR;
extern PFN_xrCreateVulkanInstanceKHR           pfn_xrCreateVulkanInstanceKHR;
extern PFN_xrGetVulkanGraphicsDevice2KHR       pfn_xrGetVulkanGraphicsDevice2KHR;
extern PFN_xrCreateVulkanDeviceKHR             pfn_xrCreateVulkanDeviceKHR;

extern PFN_vkCmdBeginRenderingKHR    pfn_vkCmdBeginRenderingKHR;
extern PFN_vkCmdEndRenderingKHR      pfn_vkCmdEndRenderingKHR;
extern PFN_vkCmdPipelineBarrier2KHR  pfn_vkCmdPipelineBarrier2KHR;
extern PFN_vkCmdWriteTimestamp2KHR   pfn_vkCmdWriteTimestamp2KHR;
extern PFN_vkCmdPushDescriptorSetKHR pfn_vkCmdPushDescriptorSetKHR;

#define vkCmdBeginRenderingKHR    pfn_vkCmdBeginRenderingKHR
#define vkCmdEndRenderingKHR      pfn_vkCmdEndRenderingKHR
#define vkCmdPipelineBarrier2KHR  pfn_vkCmdPipelineBarrier2KHR
#define vkCmdWriteTimestamp2KHR   pfn_vkCmdWriteTimestamp2KHR
#define vkCmdPushDescriptorSetKHR pfn_vkCmdPushDescriptorSetKHR

struct FatVertex {
    glm::vec4  pos;
    glm::vec4  weights;
    glm::uvec4 boneIdx;
};

inline constexpr int BONES_PER_CUBE = 8;

struct Meshlet {
    glm::vec3 aabbMin;
    uint32_t  indexOffset;
    glm::vec3 aabbMax;
    uint32_t  indexCount;
    glm::vec3 normal;
    uint32_t  boneBase;
    uint32_t  boneCount;
    uint32_t  _pad[3];
};
static_assert(sizeof(Meshlet) == 64, "Meshlet struct must be 64 bytes (std430)");

inline constexpr int MESHLET_TILE = 16;

struct App {
    android_app*  androidApp = nullptr;
    XrCtx         xr;
    VulkanCtx     vk;
    bool          initialized = false;
    int           gridN       = GRID_N;
    int           instCount   = 1;
    int           aluIters    = 0;
    int           cubeCount   = 60;
    int           mode7Frustum = 1;
    float         resScale    = 1.0f;
    uint32_t      lastBfIndexCount = 0;
    uint32_t      lastFrustumMeshlets = 0;
    uint32_t      lastDepthRejectedMeshlets = 0;
    uint32_t      lastVisibleMeshlets = 0;
    uint32_t      lastDeltaHist[8]    = {0};
    uint32_t      lastDeltaHistTotal  = 0;
    uint32_t      lastMinHHist[6]     = {0};
    uint32_t      lastDepthLimitHist[6] = {0};
    float         lastHiZProbe[6]     = {0};
    uint32_t      lastMinHHist2[6]    = {0};
    double        lastCsMs         = 0.0;
    double        lastGfxMs        = 0.0;
    double        lastHiZMs        = 0.0;
    double        lastDownsampleMs = 0.0;
    uint32_t      frameParity      = 0;
    bool          prevDepthValid   = false;
    glm::vec3     playerPos        = glm::vec3(0.f);
    float         playerYaw        = 0.f;
    XrTime        lastFrameTime    = 0;

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point lastLogTime;
    Clock::time_point startTime;
    int    frameCount   = 0;
    double frameMsAccum = 0.0;
};
