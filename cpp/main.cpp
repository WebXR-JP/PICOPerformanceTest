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
#include <glm/gtc/type_ptr.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// 生成したシェーダーヘッダー（ビルド時に cmake/embed_spirv.cmake が生成）
#include "vertex_spv.h"
#include "fragment_spv.h"

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
    VkFramebuffer framebuffer;
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
};

struct VulkanCtx {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice     = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    uint32_t         queueFamily    = 0;
    VkQueue          queue          = VK_NULL_HANDLE;
    VkCommandPool    cmdPool        = VK_NULL_HANDLE;
    VkRenderPass     renderPass     = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       pipeline       = VK_NULL_HANDLE;
    VkBuffer         vertexBuffer   = VK_NULL_HANDLE;
    VkDeviceMemory   vertexMemory   = VK_NULL_HANDLE;
    VkBuffer         indexBuffer    = VK_NULL_HANDLE;
    VkDeviceMemory   indexMemory    = VK_NULL_HANDLE;
    uint32_t         indexCount     = 0;
    uint32_t         vertexCount    = 0;
    VkFence          fence          = VK_NULL_HANDLE;
    // コマンドバッファ（スワップチェーン画像数、multiview で両目を1回で描画）
    std::vector<VkCommandBuffer> cmdBuffers; // [imageIdx]
};

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

static void CreateImage(VulkanCtx& vk, uint32_t w, uint32_t h,
                        VkFormat format, VkImageUsageFlags usage,
                        VkImage& image, VkDeviceMemory& memory,
                        uint32_t arrayLayers = 1) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType   = VK_IMAGE_TYPE_2D;
    ci.format      = format;
    ci.extent      = {w, h, 1};
    ci.mipLevels   = 1;
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

// ============================================================
// 頂点フォーマット
// fatVertex=false: vec3 のみ（12 bytes）
// fatVertex=true : vec3 pos + vec3 normal + vec2 uv（32 bytes）
// ============================================================
struct FatVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

// ============================================================
// グリッドメッシュ生成
// ============================================================
static void GenerateGridMesh(VulkanCtx& vk, int N) {
    // N×N 格子頂点、2*(N-1)^2 三角形
    const int   vCount = N * N;
    const int   iCount = (N - 1) * (N - 1) * 6;
    const float step   = 2.0f / (N - 1); // -1.0 〜 +1.0

    std::vector<FatVertex> verts(vCount);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            FatVertex& v = verts[y * N + x];
            v.pos    = { -1.0f + x * step, -1.0f + y * step, 0.0f };
            v.normal = { 0.0f, 0.0f, 1.0f };
            v.uv     = { (float)x / (N - 1), (float)y / (N - 1) };
        }
    }

    std::vector<uint32_t> indices(iCount);
    int idx = 0;
    for (int y = 0; y < N - 1; y++) {
        for (int x = 0; x < N - 1; x++) {
            uint32_t tl = y * N + x;
            uint32_t tr = tl + 1;
            uint32_t bl = tl + N;
            uint32_t br = bl + 1;
            indices[idx++] = tl; indices[idx++] = bl; indices[idx++] = tr;
            indices[idx++] = tr; indices[idx++] = bl; indices[idx++] = br;
        }
    }

    vk.vertexCount = (uint32_t)vCount;
    vk.indexCount  = (uint32_t)iCount;

    int polyCount = (N - 1) * (N - 1) * 2;
    LOGI("Mesh: %d x %d grid, %d vertices, %d indices (%d polygons)",
         N, N, vCount, iCount, polyCount);

    // 頂点バッファ（HOST_VISIBLE で直接書き込み）
    VkDeviceSize vSize = sizeof(FatVertex) * vCount;
    CreateBuffer(vk, vSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vk.vertexBuffer, vk.vertexMemory);
    void* vPtr;
    vkMapMemory(vk.device, vk.vertexMemory, 0, vSize, 0, &vPtr);
    memcpy(vPtr, verts.data(), vSize);
    vkUnmapMemory(vk.device, vk.vertexMemory);

    // インデックスバッファ
    VkDeviceSize iSize = sizeof(uint32_t) * iCount;
    CreateBuffer(vk, iSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vk.indexBuffer, vk.indexMemory);
    void* iPtr;
    vkMapMemory(vk.device, vk.indexMemory, 0, iSize, 0, &iPtr);
    memcpy(iPtr, indices.data(), iSize);
    vkUnmapMemory(vk.device, vk.indexMemory);
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

    // フレーム計測
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point lastLogTime;
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

    // multiview feature を有効化
    VkPhysicalDeviceMultiviewFeatures multiviewFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES};
    multiviewFeatures.multiview = VK_TRUE;

    VkDeviceCreateInfo devCI{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devCI.pNext                = &multiviewFeatures;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos    = &qci;

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
    sc.width  = vcViews[0].recommendedImageRectWidth;
    sc.height = vcViews[0].recommendedImageRectHeight;

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
    VkAttachmentDescription attachments[2]{};

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

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
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
    rpCI.attachmentCount = 2;
    rpCI.pAttachments    = attachments;
    rpCI.subpassCount    = 1;
    rpCI.pSubpasses      = &subpass;
    rpCI.dependencyCount = 1;
    rpCI.pDependencies   = &dep;
    CHECK_VK(vkCreateRenderPass(app.vk.device, &rpCI, nullptr, &app.vk.renderPass));
}

// ---- グラフィックスパイプライン作成 ----
static void CreatePipeline(App& app, uint32_t width, uint32_t height) {
    // シェーダーモジュール（埋め込みSPIR-Vから）
    auto makeShader = [&](const uint32_t* spv, uint32_t size) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = size;
        ci.pCode    = spv;
        VkShaderModule mod;
        CHECK_VK(vkCreateShaderModule(app.vk.device, &ci, nullptr, &mod));
        return mod;
    };

    VkShaderModule vertMod = makeShader(vertex_spv,   vertex_spv_size);
    VkShaderModule fragMod = makeShader(fragment_spv, fragment_spv_size);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName  = "main";

    // 頂点入力（FatVertex: pos のみ使用、stride=32B でパディング分も含む）
    VkVertexInputBindingDescription binding{0, sizeof(FatVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(FatVertex, pos)};

    VkPipelineVertexInputStateCreateInfo viCI{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    viCI.vertexBindingDescriptionCount   = 1;
    viCI.pVertexBindingDescriptions      = &binding;
    viCI.vertexAttributeDescriptionCount = 1;
    viCI.pVertexAttributeDescriptions    = &attr;

    VkPipelineInputAssemblyStateCreateInfo iaCI{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    iaCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{0.f, 0.f, (float)width, (float)height, 0.f, 1.f};
    VkRect2D   scissor{{0, 0}, {width, height}};
    VkPipelineViewportStateCreateInfo vpCI{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpCI.viewportCount = 1;
    vpCI.pViewports    = &viewport;
    vpCI.scissorCount  = 1;
    vpCI.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rsCI{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rsCI.polygonMode = VK_POLYGON_MODE_FILL;
    rsCI.cullMode    = VK_CULL_MODE_NONE;
    rsCI.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsCI.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msCI{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dsCI{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dsCI.depthTestEnable  = VK_TRUE;
    dsCI.depthWriteEnable = VK_TRUE;
    dsCI.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cbCI{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbCI.attachmentCount = 1;
    cbCI.pAttachments    = &cbAtt;

    // Push constants: mat4 mvp[2] + int aluIters = 132 bytes（multiview）
    VkPushConstantRange pcRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) * 2 + sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges    = &pcRange;
    CHECK_VK(vkCreatePipelineLayout(app.vk.device, &plCI, nullptr, &app.vk.pipelineLayout));

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
    gpCI.layout              = app.vk.pipelineLayout;
    gpCI.renderPass          = app.vk.renderPass;
    gpCI.subpass             = 0;
    CHECK_VK(vkCreateGraphicsPipelines(
        app.vk.device, VK_NULL_HANDLE, 1, &gpCI, nullptr, &app.vk.pipeline));

    vkDestroyShaderModule(app.vk.device, vertMod, nullptr);
    vkDestroyShaderModule(app.vk.device, fragMod, nullptr);
    LOGI("GraphicsPipeline created");
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

        VkImageView fbAttachments[] = {si.view, sc.depthView};
        VkFramebufferCreateInfo fbCI{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbCI.renderPass      = app.vk.renderPass;
        fbCI.attachmentCount = 2;
        fbCI.pAttachments    = fbAttachments;
        fbCI.width           = sc.width;
        fbCI.height          = sc.height;
        fbCI.layers          = 1; // multiview では layers=1
        CHECK_VK(vkCreateFramebuffer(app.vk.device, &fbCI, nullptr, &si.framebuffer));
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
    CreatePipeline(app, w, h);
    CreateFrameResources(app, colorFormat);
    GenerateGridMesh(app.vk, app.gridN);

    app.initialized = true;
    app.lastLogTime = App::Clock::now();
    LOGI("=== Initialization complete. GRID_N=%d, polygons≈%d ===",
         app.gridN, (app.gridN - 1) * (app.gridN - 1) * 2);
}

// ---- XrView から MVP 行列を計算（view_mat 不使用：カメラ空間固定）----
static glm::mat4 ComputeMVP(const XrView& view) {
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

    glm::mat4 model = glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, -3.f))
                    * glm::scale(glm::mat4(1.f), glm::vec3(2.f, 2.f, 1.f));
    return proj * model;
}

// ---- 両目をまとめて描画（multiview: 1 draw call で左右同時） ----
static void RenderStereo(App& app, uint32_t imageIdx,
                         const std::vector<XrView>& views,
                         uint32_t swapW, uint32_t swapH) {
    VkCommandBuffer cmd = app.vk.cmdBuffers[imageIdx];
    VkFramebuffer   fb  = app.xr.swapchains[0].images[imageIdx].framebuffer;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(cmd, &bi));

    VkClearValue clears[2];
    clears[0].color        = {0.1f, 0.1f, 0.15f, 1.0f};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBI.renderPass        = app.vk.renderPass;
    rpBI.framebuffer       = fb;
    rpBI.renderArea.offset = {0, 0};
    rpBI.renderArea.extent = {swapW, swapH};
    rpBI.clearValueCount   = 2;
    rpBI.pClearValues      = clears;

    vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, app.vk.pipeline);

    // 両目の MVP を push constants に詰める
    struct PushConst { glm::mat4 mvp[2]; int32_t aluIters; };
    PushConst pc;
    pc.mvp[0]   = ComputeMVP(views[0]);
    pc.mvp[1]   = ComputeMVP(views[1]);
    pc.aluIters = (int32_t)app.aluIters;
    vkCmdPushConstants(cmd, app.vk.pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConst), &pc);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &app.vk.vertexBuffer, &offset);
    vkCmdBindIndexBuffer(cmd, app.vk.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    // multiview: 1 draw call で gl_ViewIndex=0,1 両方が走る
    vkCmdDrawIndexed(cmd, app.vk.indexCount, (uint32_t)app.instCount, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
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
        RenderStereo(app, imgIdx, views, sc.width, sc.height);

        // ProjectionView を設定（左右眼それぞれ、arrayIndex で layer を指定）
        for (uint32_t eye = 0; eye < viewCount; eye++) {
            projViews[eye] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            projViews[eye].pose = views[eye].pose;
            projViews[eye].fov  = views[eye].fov;
            projViews[eye].subImage.swapchain             = sc.handle;
            projViews[eye].subImage.imageRect.offset      = {0, 0};
            projViews[eye].subImage.imageRect.extent      = {(int32_t)sc.width, (int32_t)sc.height};
            projViews[eye].subImage.imageArrayIndex       = eye; // 0=左, 1=右
        }

        // フェンス待機 & リセット
        vkWaitForFences(app.vk.device, 1, &app.vk.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(app.vk.device, 1, &app.vk.fence);

        // キューサブミット（コマンドバッファ1つ）
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &app.vk.cmdBuffers[imgIdx];
        CHECK_VK(vkQueueSubmit(app.vk.queue, 1, &si, app.vk.fence));

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
        int    polysInst = (app.gridN - 1) * (app.gridN - 1) * 2;
        int    polysTotal = polysInst * app.instCount;
        if (app.instCount > 1) {
            LOGI("[PERF] FPS=%.1f  FrameTime=%.2fms  Polygons=%d  (inst=%d x %d poly)",
                 fps, avgMs, polysTotal, app.instCount, polysInst);
        } else {
            LOGI("[PERF] FPS=%.1f  FrameTime=%.2fms  Polygons=%d  Vertices=%d",
                 fps, avgMs, polysTotal, (int)app.vk.vertexCount);
        }
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

        if (app.vk.fence)          vkDestroyFence(app.vk.device, app.vk.fence, nullptr);
        if (app.vk.pipeline)       vkDestroyPipeline(app.vk.device, app.vk.pipeline, nullptr);
        if (app.vk.pipelineLayout) vkDestroyPipelineLayout(app.vk.device, app.vk.pipelineLayout, nullptr);
        if (app.vk.renderPass)     vkDestroyRenderPass(app.vk.device, app.vk.renderPass, nullptr);

        if (app.vk.vertexBuffer)   vkDestroyBuffer(app.vk.device, app.vk.vertexBuffer, nullptr);
        if (app.vk.vertexMemory)   vkFreeMemory(app.vk.device, app.vk.vertexMemory, nullptr);
        if (app.vk.indexBuffer)    vkDestroyBuffer(app.vk.device, app.vk.indexBuffer, nullptr);
        if (app.vk.indexMemory)    vkFreeMemory(app.vk.device, app.vk.indexMemory, nullptr);

        for (auto& sc : app.xr.swapchains) {
            for (auto& si : sc.images) {
                if (si.framebuffer) vkDestroyFramebuffer(app.vk.device, si.framebuffer, nullptr);
                if (si.view)        vkDestroyImageView(app.vk.device, si.view, nullptr);
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
static void GetIntentParams(android_app* aApp, int& outGridN, int& outInstCount, int& outAluIters) {
    JNIEnv* env = nullptr;
    aApp->activity->vm->AttachCurrentThread(&env, nullptr);

    jobject activity   = aApp->activity->clazz;
    jclass  actClass   = env->GetObjectClass(activity);

    jmethodID getIntent   = env->GetMethodID(actClass, "getIntent", "()Landroid/content/Intent;");
    jobject   intent      = env->CallObjectMethod(activity, getIntent);
    jclass    intentClass = env->GetObjectClass(intent);
    jmethodID getIntExtra = env->GetMethodID(intentClass, "getIntExtra",
                                             "(Ljava/lang/String;I)I");

    jstring keyGridN = env->NewStringUTF("grid_n");
    jint    gridN    = env->CallIntMethod(intent, getIntExtra, keyGridN, (jint)outGridN);
    env->DeleteLocalRef(keyGridN);

    jstring keyInst  = env->NewStringUTF("inst_count");
    jint    instCount = env->CallIntMethod(intent, getIntExtra, keyInst, (jint)outInstCount);
    env->DeleteLocalRef(keyInst);

    jstring keyAlu   = env->NewStringUTF("alu_iters");
    jint    aluIters = env->CallIntMethod(intent, getIntExtra, keyAlu, (jint)outAluIters);
    env->DeleteLocalRef(keyAlu);

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
    GetIntentParams(aApp, app.gridN, app.instCount, app.aluIters);
    int polysPerInst = (app.gridN - 1) * (app.gridN - 1) * 2;
    LOGI("grid_n=%d  inst_count=%d  alu_iters=%d  polygons≈%d (total≈%d)",
         app.gridN, app.instCount, app.aluIters, polysPerInst, polysPerInst * app.instCount);

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
