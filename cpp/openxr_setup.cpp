#include "openxr_setup.hpp"

void LoadXrExtFunctions(XrInstance instance) {
    CHECK_XR(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirements2KHR",
        (PFN_xrVoidFunction*)&pfn_xrGetVulkanGraphicsRequirements2KHR));
    CHECK_XR(xrGetInstanceProcAddr(instance, "xrCreateVulkanInstanceKHR",
        (PFN_xrVoidFunction*)&pfn_xrCreateVulkanInstanceKHR));
    CHECK_XR(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDevice2KHR",
        (PFN_xrVoidFunction*)&pfn_xrGetVulkanGraphicsDevice2KHR));
    CHECK_XR(xrGetInstanceProcAddr(instance, "xrCreateVulkanDeviceKHR",
        (PFN_xrVoidFunction*)&pfn_xrCreateVulkanDeviceKHR));
}

void InitializeOpenXRLoader(App& app) {
    PFN_xrInitializeLoaderKHR initLoader = nullptr;
    CHECK_XR(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
        (PFN_xrVoidFunction*)&initLoader));

    XrLoaderInitInfoAndroidKHR loaderInfo{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
    loaderInfo.applicationVM      = app.androidApp->activity->vm;
    loaderInfo.applicationContext = app.androidApp->activity->clazz;
    CHECK_XR(initLoader((const XrLoaderInitInfoBaseHeaderKHR*)&loaderInfo));
    LOGI("OpenXR loader initialized");
}

void CreateXrInstance(App& app) {
    const char* extensions[] = {
        "XR_KHR_android_create_instance",
        "XR_KHR_vulkan_enable2",
        "XR_BD_controller_interaction",
    };

    XrInstanceCreateInfoAndroidKHR androidInfo{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
    androidInfo.applicationVM       = app.androidApp->activity->vm;
    androidInfo.applicationActivity = app.androidApp->activity->clazz;

    XrApplicationInfo appInfo{};
    strncpy(appInfo.applicationName, "PICOPerfTest", XR_MAX_APPLICATION_NAME_SIZE - 1);
    strncpy(appInfo.engineName,      "None",          XR_MAX_ENGINE_NAME_SIZE - 1);
    appInfo.apiVersion = XR_MAKE_VERSION(1, 0, 34);

    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
    ci.next                  = &androidInfo;
    ci.applicationInfo       = appInfo;
    ci.enabledExtensionCount = (uint32_t)(sizeof(extensions) / sizeof(extensions[0]));
    ci.enabledExtensionNames = extensions;

    CHECK_XR(xrCreateInstance(&ci, &app.xr.instance));
    LOGI("XrInstance created");

    LoadXrExtFunctions(app.xr.instance);
}

void CreateVulkanDevice(App& app) {
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    XrGraphicsRequirementsVulkan2KHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
    CHECK_XR(pfn_xrGetVulkanGraphicsRequirements2KHR(
        app.xr.instance, app.xr.systemId, &req));
    LOGI("Vulkan version requirement: min=%d.%d, max=%d.%d",
        XR_VERSION_MAJOR(req.minApiVersionSupported),
        XR_VERSION_MINOR(req.minApiVersionSupported),
        XR_VERSION_MAJOR(req.maxApiVersionSupported),
        XR_VERSION_MINOR(req.maxApiVersionSupported));

    VkApplicationInfo vkApp{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    vkApp.pApplicationName = "PICOPerfTest";
    vkApp.apiVersion       = VK_API_VERSION_1_0;

    VkInstanceCreateInfo vkInstCI{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    vkInstCI.pApplicationInfo        = &vkApp;
    vkInstCI.enabledExtensionCount   = 0;
    vkInstCI.ppEnabledExtensionNames = nullptr;

    XrVulkanInstanceCreateInfoKHR xrInstCI{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
    xrInstCI.systemId               = app.xr.systemId;
    xrInstCI.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    xrInstCI.vulkanCreateInfo       = &vkInstCI;

    VkInstance rawInstance = VK_NULL_HANDLE;
    VkResult vkResult = VK_SUCCESS;
    CHECK_XR(pfn_xrCreateVulkanInstanceKHR(
        app.xr.instance, &xrInstCI, &rawInstance, &vkResult));
    vk::detail::resultCheck(static_cast<vk::Result>(vkResult), "xrCreateVulkanInstanceKHR");
    app.vk.instance = vk::raii::Instance(app.vk.context, rawInstance);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*app.vk.instance);
    LOGI("VkInstance created");

    XrVulkanGraphicsDeviceGetInfoKHR devGetInfo{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
    devGetInfo.systemId       = app.xr.systemId;
    devGetInfo.vulkanInstance = Raw(app.vk.instance);
    VkPhysicalDevice rawPhysDevice = VK_NULL_HANDLE;
    CHECK_XR(pfn_xrGetVulkanGraphicsDevice2KHR(
        app.xr.instance, &devGetInfo, &rawPhysDevice));
    app.vk.physDevice = vk::raii::PhysicalDevice(app.vk.instance, rawPhysDevice);
    LOGI("VkPhysicalDevice selected");

    const auto qfProps = app.vk.physDevice.getQueueFamilyProperties();
    app.vk.queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfProps.size(); ++i) {
        if (qfProps[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            app.vk.queueFamily = i;
            break;
        }
    }
    assert(app.vk.queueFamily != UINT32_MAX);

    const auto devProps = app.vk.physDevice.getProperties();
    LOGI("maxPushConstantsSize=%u", devProps.limits.maxPushConstantsSize);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = app.vk.queueFamily;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;

    {
        const auto exts = app.vk.physDevice.enumerateDeviceExtensionProperties();
        bool found = false;
        for (const auto& e : exts) {
            if (strcmp(e.extensionName, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME) == 0) {
                found = true; break;
            }
        }
        LOGI("VK_KHR_draw_indirect_count: %s", found ? "YES" : "NO");
        if (!found) { LOGE("VK_KHR_draw_indirect_count not supported"); assert(false); }
    }

    VkPhysicalDeviceMultiviewFeatures multiviewFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES};
    multiviewFeatures.multiview = VK_TRUE;

    VkPhysicalDeviceVulkan13Features vulkan13Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    vulkan13Features.subgroupSizeControl  = VK_TRUE;
    vulkan13Features.computeFullSubgroups = VK_TRUE;
    vulkan13Features.dynamicRendering     = VK_TRUE;
    vulkan13Features.synchronization2     = VK_TRUE;
    multiviewFeatures.pNext = &vulkan13Features;

    VkPhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.multiDrawIndirect = VK_TRUE;

    const char* deviceExtensions[] = {
        VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    };

    VkDeviceCreateInfo devCI{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devCI.pNext                   = &multiviewFeatures;
    devCI.queueCreateInfoCount    = 1;
    devCI.pQueueCreateInfos       = &qci;
    devCI.pEnabledFeatures        = &enabledFeatures;
    devCI.enabledExtensionCount   = (uint32_t)(sizeof(deviceExtensions) / sizeof(deviceExtensions[0]));
    devCI.ppEnabledExtensionNames = deviceExtensions;

    XrVulkanDeviceCreateInfoKHR xrDevCI{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
    xrDevCI.systemId               = app.xr.systemId;
    xrDevCI.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    xrDevCI.vulkanPhysicalDevice   = Raw(app.vk.physDevice);
    xrDevCI.vulkanCreateInfo       = &devCI;

    VkDevice rawDevice = VK_NULL_HANDLE;
    CHECK_XR(pfn_xrCreateVulkanDeviceKHR(
        app.xr.instance, &xrDevCI, &rawDevice, &vkResult));
    vk::detail::resultCheck(static_cast<vk::Result>(vkResult), "xrCreateVulkanDeviceKHR");
    app.vk.device = vk::raii::Device(app.vk.physDevice, rawDevice);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*app.vk.device);
    LOGI("VkDevice created");

    app.vk.queue = app.vk.device.getQueue(app.vk.queueFamily, 0);

    pfn_vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
        vkGetDeviceProcAddr(Raw(app.vk.device), "vkCmdBeginRenderingKHR"));
    pfn_vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
        vkGetDeviceProcAddr(Raw(app.vk.device), "vkCmdEndRenderingKHR"));
    pfn_vkCmdPipelineBarrier2KHR = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(
        vkGetDeviceProcAddr(Raw(app.vk.device), "vkCmdPipelineBarrier2KHR"));
    pfn_vkCmdWriteTimestamp2KHR = reinterpret_cast<PFN_vkCmdWriteTimestamp2KHR>(
        vkGetDeviceProcAddr(Raw(app.vk.device), "vkCmdWriteTimestamp2KHR"));
    pfn_vkCmdPushDescriptorSetKHR = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        vkGetDeviceProcAddr(Raw(app.vk.device), "vkCmdPushDescriptorSetKHR"));
    assert(pfn_vkCmdBeginRenderingKHR && pfn_vkCmdEndRenderingKHR);
    assert(pfn_vkCmdPipelineBarrier2KHR && pfn_vkCmdWriteTimestamp2KHR);
    assert(pfn_vkCmdPushDescriptorSetKHR);

    vma::AllocatorCreateInfo aci{};
    aci.physicalDevice   = *app.vk.physDevice;
    aci.vulkanApiVersion = VK_API_VERSION_1_1;
    app.vk.allocator = vma::raii::Allocator(app.vk.instance, app.vk.device, aci);
    LOGI("VmaAllocator created");
}

void CreateXrSession(App& app) {
    XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    binding.instance         = Raw(app.vk.instance);
    binding.physicalDevice   = Raw(app.vk.physDevice);
    binding.device           = Raw(app.vk.device);
    binding.queueFamilyIndex = app.vk.queueFamily;
    binding.queueIndex       = 0;

    XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
    sci.next     = &binding;
    sci.systemId = app.xr.systemId;
    CHECK_XR(xrCreateSession(app.xr.instance, &sci, &app.xr.session));
    LOGI("XrSession created");

    uint32_t spaceCount = 0;
    CHECK_XR(xrEnumerateReferenceSpaces(app.xr.session, 0, &spaceCount, nullptr));
    std::vector<XrReferenceSpaceType> supportedSpaces(spaceCount);
    CHECK_XR(xrEnumerateReferenceSpaces(app.xr.session, spaceCount, &spaceCount, supportedSpaces.data()));
    bool hasStage = false;
    for (auto t : supportedSpaces) if (t == XR_REFERENCE_SPACE_TYPE_STAGE) hasStage = true;

    XrReferenceSpaceCreateInfo rci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    rci.referenceSpaceType = hasStage ? XR_REFERENCE_SPACE_TYPE_STAGE
                                       : XR_REFERENCE_SPACE_TYPE_LOCAL;
    rci.poseInReferenceSpace.orientation = {0, 0, 0, 1};
    rci.poseInReferenceSpace.position    = {0, 0, 0};
    CHECK_XR(xrCreateReferenceSpace(app.xr.session, &rci, &app.xr.appSpace));
    LOGI("appSpace type: %s", hasStage ? "STAGE" : "LOCAL");
}

void CreateSwapchains(App& app) {
    uint32_t viewCount = 0;
    CHECK_XR(xrEnumerateViewConfigurationViews(
        app.xr.instance, app.xr.systemId,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr));
    std::vector<XrViewConfigurationView> vcViews(viewCount,
        {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    CHECK_XR(xrEnumerateViewConfigurationViews(
        app.xr.instance, app.xr.systemId,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, vcViews.data()));

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
        colorFmt = fmts[0];
        LOGW("VK_FORMAT_R8G8B8A8_SRGB not supported, using format %lld", (long long)colorFmt);
    }
    LOGI("Swapchain color format: %lld", (long long)colorFmt);

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
    sci.arraySize   = 2;
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

void CreateXrInput(App& app) {
    XrActionSetCreateInfo asci{XR_TYPE_ACTION_SET_CREATE_INFO};
    strcpy(asci.actionSetName,          "locomotion");
    strcpy(asci.localizedActionSetName, "Locomotion");
    asci.priority = 0;
    CHECK_XR(xrCreateActionSet(app.xr.instance, &asci, &app.xr.actionSet));

    auto makeAction = [&](const char* name, const char* locName, XrActionType type) {
        XrActionCreateInfo aci{XR_TYPE_ACTION_CREATE_INFO};
        strcpy(aci.actionName,          name);
        strcpy(aci.localizedActionName, locName);
        aci.actionType = type;
        XrAction a = XR_NULL_HANDLE;
        CHECK_XR(xrCreateAction(app.xr.actionSet, &aci, &a));
        return a;
    };
    app.xr.moveAction        = makeAction("move",         "Move",         XR_ACTION_TYPE_VECTOR2F_INPUT);
    app.xr.turnAction        = makeAction("turn",         "Turn",         XR_ACTION_TYPE_VECTOR2F_INPUT);
    app.xr.debugToggleAction = makeAction("debug_toggle", "Debug Toggle", XR_ACTION_TYPE_BOOLEAN_INPUT);

    XrPath leftStick = 0, rightStick = 0, rightA = 0, rightTrigger = 0, leftMenu = 0;
    CHECK_XR(xrStringToPath(app.xr.instance, "/user/hand/left/input/thumbstick",  &leftStick));
    CHECK_XR(xrStringToPath(app.xr.instance, "/user/hand/right/input/thumbstick", &rightStick));
    CHECK_XR(xrStringToPath(app.xr.instance, "/user/hand/right/input/a/click",    &rightA));
    CHECK_XR(xrStringToPath(app.xr.instance, "/user/hand/right/input/trigger/click", &rightTrigger));
    CHECK_XR(xrStringToPath(app.xr.instance, "/user/hand/left/input/menu/click",  &leftMenu));

    XrActionSuggestedBinding bindings[3];
    bindings[0] = {app.xr.moveAction, leftStick};
    bindings[1] = {app.xr.turnAction, rightStick};
    bindings[2] = {app.xr.debugToggleAction, rightA};  // default: A button

    const char* profiles[] = {
        "/interaction_profiles/khr/simple_controller",
        "/interaction_profiles/oculus/touch_controller",
        "/interaction_profiles/bytedance/pico_neo3_controller",
        "/interaction_profiles/bytedance/pico4_controller",
    };
    for (const char* p : profiles) {
        XrPath profilePath = 0;
        XrResult r = xrStringToPath(app.xr.instance, p, &profilePath);
        if (XR_FAILED(r)) continue;
        // simple_controller lacks A button — use trigger instead
        bool isSimple = strstr(p, "simple_controller") != nullptr;
        XrActionSuggestedBinding localBindings[3] = {bindings[0], bindings[1], bindings[2]};
        if (isSimple) localBindings[2].binding = rightTrigger;
        XrInteractionProfileSuggestedBinding s{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        s.interactionProfile = profilePath;
        s.countSuggestedBindings = 3;
        s.suggestedBindings = localBindings;
        XrResult sr = xrSuggestInteractionProfileBindings(app.xr.instance, &s);
        if (XR_FAILED(sr)) {
            LOGI("Suggest bindings failed for %s: %d", p, (int)sr);
        }
    }
    (void)leftMenu;

    XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attach.countActionSets = 1;
    attach.actionSets = &app.xr.actionSet;
    CHECK_XR(xrAttachSessionActionSets(app.xr.session, &attach));

    LOGI("OpenXR input: locomotion action set attached");
}

void PollXrInput(App& app, XrTime predictedDisplayTime) {
    if (!app.xr.actionSet) return;

    XrActiveActionSet active{app.xr.actionSet, XR_NULL_PATH};
    XrActionsSyncInfo si{XR_TYPE_ACTIONS_SYNC_INFO};
    si.countActiveActionSets = 1;
    si.activeActionSets = &active;
    XrResult sr = xrSyncActions(app.xr.session, &si);
    if (XR_FAILED(sr)) return;

    auto readVec2 = [&](XrAction a, float& x, float& y) {
        x = 0; y = 0;
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = a;
        XrActionStateVector2f st{XR_TYPE_ACTION_STATE_VECTOR2F};
        if (XR_SUCCEEDED(xrGetActionStateVector2f(app.xr.session, &gi, &st)) && st.isActive) {
            x = st.currentState.x;
            y = st.currentState.y;
        }
    };

    float mx = 0, my = 0, tx = 0, ty = 0;
    readVec2(app.xr.moveAction, mx, my);
    readVec2(app.xr.turnAction, tx, ty);

    {
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = app.xr.debugToggleAction;
        XrActionStateBoolean st{XR_TYPE_ACTION_STATE_BOOLEAN};
        if (XR_SUCCEEDED(xrGetActionStateBoolean(app.xr.session, &gi, &st)) && st.isActive) {
            bool pressed = st.currentState != XR_FALSE;
            if (pressed && !app.xr.debugTogglePrev) {
                app.debugAabbEnabled = !app.debugAabbEnabled;
                LOGI("AABB debug draw: %s", app.debugAabbEnabled ? "ON" : "OFF");
            }
            app.xr.debugTogglePrev = pressed;
        }
    }

    float dt = 0.f;
    if (app.lastFrameTime != 0) {
        dt = (float)((predictedDisplayTime - app.lastFrameTime) * 1e-9);
    }
    app.lastFrameTime = predictedDisplayTime;
    if (dt > 0.1f) dt = 0.1f;

    const float moveSpeed = 2.0f;  // m/s
    const float turnSpeed = glm::radians(90.f); // rad/s (smooth turn)
    const float deadzone  = 0.15f;

    auto dz = [&](float v) { return fabsf(v) < deadzone ? 0.f : v; };
    mx = dz(mx); my = dz(my); tx = dz(tx); ty = dz(ty);

    // move in yaw-rotated XZ plane (forward = -Z in view, right = +X)
    float c = cosf(app.playerYaw), s = sinf(app.playerYaw);
    glm::vec3 forward( -s, 0,  -c);
    glm::vec3 right  (  c, 0,  -s);
    app.playerPos += (right * mx + forward * my) * (moveSpeed * dt);
    app.playerPos.y += ty * moveSpeed * dt;
    app.playerYaw -= tx * turnSpeed * dt;
}
