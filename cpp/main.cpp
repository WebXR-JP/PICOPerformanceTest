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

#define VMA_IMPLEMENTATION

#include "common.hpp"

#include "mesh.hpp"
#include "openxr_setup.hpp"
#include "pipelines.hpp"
#include "render.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

PFN_xrGetVulkanGraphicsRequirements2KHR pfn_xrGetVulkanGraphicsRequirements2KHR = nullptr;
PFN_xrCreateVulkanInstanceKHR           pfn_xrCreateVulkanInstanceKHR           = nullptr;
PFN_xrGetVulkanGraphicsDevice2KHR       pfn_xrGetVulkanGraphicsDevice2KHR       = nullptr;
PFN_xrCreateVulkanDeviceKHR             pfn_xrCreateVulkanDeviceKHR             = nullptr;

PFN_vkCmdBeginRenderingKHR    pfn_vkCmdBeginRenderingKHR    = nullptr;
PFN_vkCmdEndRenderingKHR      pfn_vkCmdEndRenderingKHR      = nullptr;
PFN_vkCmdPipelineBarrier2KHR  pfn_vkCmdPipelineBarrier2KHR  = nullptr;
PFN_vkCmdWriteTimestamp2KHR   pfn_vkCmdWriteTimestamp2KHR   = nullptr;
PFN_vkCmdPushDescriptorSetKHR pfn_vkCmdPushDescriptorSetKHR = nullptr;

void Initialize(App& app) {
    InitializeOpenXRLoader(app);
    CreateXrInstance(app);

    XrSystemGetInfo sgi{XR_TYPE_SYSTEM_GET_INFO};
    sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    CHECK_XR(xrGetSystem(app.xr.instance, &sgi, &app.xr.systemId));

    CreateVulkanDevice(app);
    CreateXrSession(app);
    CreateSwapchains(app);
    CreateXrInput(app);

    app.vk.colorFormat = VK_FORMAT_R8G8B8A8_SRGB;

    uint32_t w = app.xr.swapchains[0].width;
    uint32_t h = app.xr.swapchains[0].height;
    CreateFrameResources(app, app.vk.colorFormat);
    GenerateMultiCubeMesh(app.vk, app.gridN, app.cubeCount);
    CreateVsSkinPipeline(app, w, h);
    CreateSkinCullLdsPipeline(app);
    CreateHiZSpdPipeline(app);
    CreateMeshletDebugPipeline(app, w, h);
    CreateQueryPool(app);

    app.initialized = true;
    app.lastLogTime = App::Clock::now();
    app.startTime   = app.lastLogTime;
    uint32_t computeGroups = (app.vk.meshletCount + 63) / 64;
    LOGI("=== Initialization complete. GRID_N=%d cubes=%d meshlets=%u computeGroups=%u ===",
         app.gridN, app.cubeCount, app.vk.meshletCount, computeGroups);
}

void HandleXrEvents(App& app) {
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

void HandleAndroidCmd(android_app* aApp, int32_t cmd) {
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

void Cleanup(App& app) {
    if (IsValid(app.vk.device)) {
        CheckVkResult(vkDeviceWaitIdle(Raw(app.vk.device)));

        app.vk.cmdBuffers.clear();
        app.vk.fence = nullptr;

        app.vk.meshletDebugPipeline = nullptr;
        app.vk.meshletDebugPipelineLayout = nullptr;
        app.vk.meshletDebugDescLayout = nullptr;
        app.vk.hiZSpdPipeline = nullptr;
        app.vk.hiZSpdPipelineLayout = nullptr;
        app.vk.hiZSpdDescLayout = nullptr;
        app.vk.hiZSpdAtomicBuffer = nullptr;
        app.vk.hiZNaiveInitPipeline = nullptr;
        app.vk.hiZNaiveInitPipelineLayout = nullptr;
        app.vk.hiZNaiveInitDescLayout = nullptr;
        app.vk.hiZNaiveStepPipeline = nullptr;
        app.vk.hiZNaiveStepPipelineLayout = nullptr;
        app.vk.hiZNaiveStepDescLayout = nullptr;
        app.vk.prevFrameBuffer = nullptr;
        app.vk.hiZSampler = nullptr;
        app.vk.prevDepthSampler = nullptr;
        app.vk.skinCullLdsPipeline = nullptr;
        app.vk.skinCullLdsPipelineLayout = nullptr;
        app.vk.skinCullLdsDescLayout = nullptr;
        app.vk.vsSkinPipeline = nullptr;
        app.vk.vsSkinPipelineLayout = nullptr;
        app.vk.vsSkinDescLayout = nullptr;
        app.vk.boneBuffer = nullptr;
        app.vk.cullStatsBuffer = nullptr;
        app.vk.bfDrawCmdBuffer = nullptr;
        app.vk.compactIndexBuffer = nullptr;
        app.vk.queryPool = nullptr;
        app.vk.meshletBuffer = nullptr;
        app.vk.indexBuffer = nullptr;
        app.vk.vertexBuffer = nullptr;

        for (auto& sc : app.xr.swapchains) {
            for (auto& si : sc.images) {
                si.view = nullptr;
            }
            for (uint32_t ping = 0; ping < 2; ++ping) {
                sc.hiZMipViews[ping].clear();
                sc.hiZViews[ping] = nullptr;
                sc.hiZImages[ping] = nullptr;
                sc.motionViews[ping] = nullptr;
                sc.motionImages[ping] = nullptr;
                sc.prevDepthViews[ping] = nullptr;
                sc.prevDepthImages[ping] = nullptr;
            }
            sc.depthSampleView = nullptr;
            sc.depthView = nullptr;
            sc.depthImage = nullptr;
        }

        app.vk.cmdPool = nullptr;
        app.vk.allocator = nullptr;
        app.vk.queue = nullptr;
        app.vk.device = nullptr;
        app.vk.physDevice = nullptr;
    }
    app.vk.instance = nullptr;

    if (app.xr.appSpace) xrDestroySpace(app.xr.appSpace);
    for (auto& sc : app.xr.swapchains) {
        if (sc.handle) xrDestroySwapchain(sc.handle);
    }
    if (app.xr.session)  xrDestroySession(app.xr.session);
    if (app.xr.instance) xrDestroyInstance(app.xr.instance);
}

void GetIntentParams(android_app* aApp, int& outGridN, int& outInstCount, int& outAluIters, int& outCubeCount, int& outMode7Frustum, float& outResScale) {
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

void android_main(android_app* aApp) {
    LOGI("android_main: start");
    aApp->onAppCmd = HandleAndroidCmd;

    App app;
    app.androidApp = aApp;

    {
        int events;
        android_poll_source* source;
        while (ALooper_pollAll(0, nullptr, &events, (void**)&source) >= 0) {
            if (source) source->process(aApp, source);
        }
    }

    if (aApp->destroyRequested) return;

    app.gridN = 97;
    GetIntentParams(aApp, app.gridN, app.instCount, app.aluIters, app.cubeCount, app.mode7Frustum, app.resScale);
    LOGI("grid_n=%d  num_cubes=%d  inst_count=%d  alu_iters=%d  mode7_frustum=%d  res_scale=%.2f",
         app.gridN, app.cubeCount, app.instCount, app.aluIters, app.mode7Frustum, app.resScale);

    JNIEnv* jniEnv = nullptr;
    aApp->activity->vm->AttachCurrentThread(&jniEnv, nullptr);

    Initialize(app);

    aApp->activity->vm->DetachCurrentThread();

    LOGI("=== Entering main loop ===");
    while (!aApp->destroyRequested && !app.xr.exitRequested) {
        int events;
        android_poll_source* source;
        while (ALooper_pollAll(0, nullptr, &events, (void**)&source) >= 0) {
            if (source) source->process(aApp, source);
            if (aApp->destroyRequested) break;
        }

        HandleXrEvents(app);

        if (app.xr.sessionRunning) {
            RenderFrame(app);
        }
    }

    LOGI("=== Exiting, cleaning up ===");
    Cleanup(app);
}
