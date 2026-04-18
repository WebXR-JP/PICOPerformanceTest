#include "pipelines.hpp"

#include "vk_resources.hpp"

#include "fragment_spv.h"
#include "hiz_spd_spv.h"
#include "hiz_naive_init_spv.h"
#include "hiz_naive_step_spv.h"
#include "meshlet_aabb_debug_frag_spv.h"
#include "meshlet_aabb_debug_vert_spv.h"
#include "skin_cull_lds_spv.h"
#include "vertex_vs_skin_spv.h"

vk::raii::ImageView CreateImageView(VulkanCtx& vk,
                                    VkImage image,
                                    VkFormat format,
                                    VkImageAspectFlags aspectMask,
                                    VkImageViewType viewType,
                                    uint32_t baseMipLevel,
                                    uint32_t levelCount,
                                    uint32_t baseArrayLayer,
                                    uint32_t layerCount) {
    vk::ImageViewCreateInfo ci{};
    ci.image = image;
    ci.viewType = static_cast<vk::ImageViewType>(viewType);
    ci.format = static_cast<vk::Format>(format);
    ci.subresourceRange = vk::ImageSubresourceRange(
        static_cast<vk::ImageAspectFlags>(aspectMask), baseMipLevel, levelCount, baseArrayLayer, layerCount);
    return vk::raii::ImageView(vk.device, ci);
}

void CreateQueryPool(App& app) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(Raw(app.vk.physDevice), &props);
    app.vk.timestampPeriod = props.limits.timestampPeriod;

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(Raw(app.vk.physDevice), &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(Raw(app.vk.physDevice), &qfCount, qfProps.data());
    uint32_t validBits = qfProps[app.vk.queueFamily].timestampValidBits;
    LOGI("timestampPeriod=%.2f ns  timestampValidBits=%u", app.vk.timestampPeriod, validBits);
    if (validBits == 0) {
        LOGW("Timestamp queries not supported on this queue family");
        return;
    }

    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = TS_COUNT;
    VkQueryPool rawQueryPool = VK_NULL_HANDLE;
    CheckVkResult(vkCreateQueryPool(Raw(app.vk.device), &ci, nullptr, &rawQueryPool));
    app.vk.queryPool = vk::raii::QueryPool(app.vk.device, rawQueryPool);
    LOGI("QueryPool (timestamp x%u) created", TS_COUNT);
}

void CreateVsSkinPipeline(App& app, uint32_t width, uint32_t height) {
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t i = 0; i < 2; i++) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    dslCI.bindingCount = 2;
    dslCI.pBindings    = bindings;
    VkDescriptorSetLayout rawDescLayout = VK_NULL_HANDLE;
    CheckVkResult(vkCreateDescriptorSetLayout(Raw(app.vk.device), &dslCI, nullptr, &rawDescLayout));
    app.vk.vsSkinDescLayout = vk::raii::DescriptorSetLayout(app.vk.device, rawDescLayout);

    VkPushConstantRange pcRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) * 2 + sizeof(int32_t) * 3};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount         = 1;
    VkDescriptorSetLayout vsSkinDescLayout = Raw(app.vk.vsSkinDescLayout);
    plCI.pSetLayouts            = &vsSkinDescLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges    = &pcRange;
    VkPipelineLayout rawPipelineLayout = VK_NULL_HANDLE;
    CheckVkResult(vkCreatePipelineLayout(Raw(app.vk.device), &plCI, nullptr, &rawPipelineLayout));
    app.vk.vsSkinPipelineLayout = vk::raii::PipelineLayout(app.vk.device, rawPipelineLayout);

    auto makeShader = [&](const uint32_t* spv, uint32_t size) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = size; ci.pCode = spv;
        VkShaderModule rawModule = VK_NULL_HANDLE;
        CheckVkResult(vkCreateShaderModule(Raw(app.vk.device), &ci, nullptr, &rawModule));
        return vk::raii::ShaderModule(app.vk.device, rawModule);
    };
    vk::raii::ShaderModule vertMod = makeShader(vertex_vs_skin_spv, vertex_vs_skin_spv_size);
    vk::raii::ShaderModule fragMod = makeShader(fragment_spv,       fragment_spv_size);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = Raw(vertMod); stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = Raw(fragMod); stages[1].pName = "main";

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
    dsCI.depthCompareOp   = VK_COMPARE_OP_GREATER;
    VkPipelineColorBlendAttachmentState cbAtt[3]{};
    cbAtt[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAtt[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    cbAtt[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cbCI{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbCI.attachmentCount = 3; cbCI.pAttachments = cbAtt;

    VkFormat colorFmts[3] = {
        app.vk.colorFormat, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};
    VkPipelineRenderingCreateInfoKHR dynRI{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    dynRI.viewMask                = 0b11u;
    dynRI.colorAttachmentCount    = 3;
    dynRI.pColorAttachmentFormats = colorFmts;
    dynRI.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gpCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpCI.pNext               = &dynRI;
    gpCI.stageCount          = 2;
    gpCI.pStages             = stages;
    gpCI.pVertexInputState   = &viCI;
    gpCI.pInputAssemblyState = &iaCI;
    gpCI.pViewportState      = &vpCI;
    gpCI.pRasterizationState = &rsCI;
    gpCI.pMultisampleState   = &msCI;
    gpCI.pDepthStencilState  = &dsCI;
    gpCI.pColorBlendState    = &cbCI;
    gpCI.layout              = Raw(app.vk.vsSkinPipelineLayout);
    VkPipeline rawPipeline = VK_NULL_HANDLE;
    CheckVkResult(vkCreateGraphicsPipelines(
        Raw(app.vk.device), VK_NULL_HANDLE, 1, &gpCI, nullptr, &rawPipeline));
    app.vk.vsSkinPipeline = vk::raii::Pipeline(app.vk.device, rawPipeline);
    LOGI("GraphicsPipeline (VS-skin mode=5) created");
}

void CreateSkinCullLdsPipeline(App& app) {
    VkDescriptorSetLayoutBinding bindings[8]{};
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
    bindings[7].binding         = 7;
    bindings[7].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    dslCI.bindingCount = 8;
    dslCI.pBindings    = bindings;
    VkDescriptorSetLayout rawDescLayout = VK_NULL_HANDLE;
    CheckVkResult(vkCreateDescriptorSetLayout(Raw(app.vk.device), &dslCI, nullptr, &rawDescLayout));
    app.vk.skinCullLdsDescLayout = vk::raii::DescriptorSetLayout(app.vk.device, rawDescLayout);

    VkSamplerCreateInfo samplerCI{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCI.magFilter = VK_FILTER_NEAREST;
    samplerCI.minFilter = VK_FILTER_NEAREST;
    samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = 0.0f;
    VkSampler rawPrevDepthSampler = VK_NULL_HANDLE;
    CheckVkResult(vkCreateSampler(Raw(app.vk.device), &samplerCI, nullptr, &rawPrevDepthSampler));
    app.vk.prevDepthSampler = vk::raii::Sampler(app.vk.device, rawPrevDepthSampler);

    samplerCI.maxLod = (float)std::max<int32_t>((int32_t)app.xr.swapchains[0].hiZMipCount - 1, 0);
    VkSampler rawHiZSampler = VK_NULL_HANDLE;
    CheckVkResult(vkCreateSampler(Raw(app.vk.device), &samplerCI, nullptr, &rawHiZSampler));
    app.vk.hiZSampler = vk::raii::Sampler(app.vk.device, rawHiZSampler);

    VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(glm::mat4) * 2 + sizeof(glm::vec4) * 2 + sizeof(uint32_t) * 4 + sizeof(glm::vec4)};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkDescriptorSetLayout skinCullDescLayout = Raw(app.vk.skinCullLdsDescLayout);
    plCI.setLayoutCount = 1; plCI.pSetLayouts = &skinCullDescLayout;
    plCI.pushConstantRangeCount = 1; plCI.pPushConstantRanges = &pcRange;
    VkPipelineLayout rawPipelineLayout = VK_NULL_HANDLE;
    CheckVkResult(vkCreatePipelineLayout(Raw(app.vk.device), &plCI, nullptr, &rawPipelineLayout));
    app.vk.skinCullLdsPipelineLayout = vk::raii::PipelineLayout(app.vk.device, rawPipelineLayout);

    VkShaderModuleCreateInfo smCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smCI.codeSize = skin_cull_lds_spv_size;
    smCI.pCode    = skin_cull_lds_spv;
    VkShaderModule rawShaderModule = VK_NULL_HANDLE;
    CheckVkResult(vkCreateShaderModule(Raw(app.vk.device), &smCI, nullptr, &rawShaderModule));
    vk::raii::ShaderModule mod(app.vk.device, rawShaderModule);

    VkComputePipelineCreateInfo cpCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroupSizeCI{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
    subgroupSizeCI.requiredSubgroupSize = 64;
    cpCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpCI.stage.module = Raw(mod);
    cpCI.stage.pName  = "main";
    cpCI.stage.flags  = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    cpCI.stage.pNext  = &subgroupSizeCI;
    cpCI.layout       = Raw(app.vk.skinCullLdsPipelineLayout);
    VkPipeline rawPipeline = VK_NULL_HANDLE;
    CheckVkResult(vkCreateComputePipelines(Raw(app.vk.device), VK_NULL_HANDLE, 1, &cpCI, nullptr, &rawPipeline));
    app.vk.skinCullLdsPipeline = vk::raii::Pipeline(app.vk.device, rawPipeline);
    LOGI("ComputePipeline (skin+cull LDS mode=7) created, meshlets=%u", app.vk.meshletCount);
}

void CreateHiZSpdPipeline(App& app) {
    EyeSwapchain& sc = app.xr.swapchains[0];
    if (sc.hiZMipCount > HIZ_SPD_MAX_MIPS) {
        LOGE("Hi-Z mip count %u exceeds SPD limit %u", sc.hiZMipCount, HIZ_SPD_MAX_MIPS);
        assert(false);
    }

    CreateBuffer(app.vk, sizeof(uint32_t) * 2u,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app.vk.hiZSpdAtomicBuffer);
    {
        void* p = app.vk.hiZSpdAtomicBuffer.getAllocation().map();
        std::memset(p, 0, sizeof(uint32_t) * 2u);
        app.vk.hiZSpdAtomicBuffer.getAllocation().unmap();
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
    dslCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    dslCI.bindingCount = 15;
    dslCI.pBindings = bindings;
    VkDescriptorSetLayout rawDescLayout = VK_NULL_HANDLE;
    CheckVkResult(vkCreateDescriptorSetLayout(Raw(app.vk.device), &dslCI, nullptr, &rawDescLayout));
    app.vk.hiZSpdDescLayout = vk::raii::DescriptorSetLayout(app.vk.device, rawDescLayout);

    VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1;
    VkDescriptorSetLayout hiZDescLayout = Raw(app.vk.hiZSpdDescLayout);
    plCI.pSetLayouts = &hiZDescLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pcRange;
    VkPipelineLayout rawPipelineLayout = VK_NULL_HANDLE;
    CheckVkResult(vkCreatePipelineLayout(Raw(app.vk.device), &plCI, nullptr, &rawPipelineLayout));
    app.vk.hiZSpdPipelineLayout = vk::raii::PipelineLayout(app.vk.device, rawPipelineLayout);

    VkShaderModuleCreateInfo smCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smCI.codeSize = hiz_spd_spv_size;
    smCI.pCode = hiz_spd_spv;
    VkShaderModule rawShaderModule = VK_NULL_HANDLE;
    CheckVkResult(vkCreateShaderModule(Raw(app.vk.device), &smCI, nullptr, &rawShaderModule));
    vk::raii::ShaderModule mod(app.vk.device, rawShaderModule);

    VkComputePipelineCreateInfo cpCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpCI.stage.module = Raw(mod);
    cpCI.stage.pName = "main";
    cpCI.layout = Raw(app.vk.hiZSpdPipelineLayout);
    VkPipeline rawPipeline = VK_NULL_HANDLE;
    CheckVkResult(vkCreateComputePipelines(Raw(app.vk.device), VK_NULL_HANDLE, 1, &cpCI, nullptr, &rawPipeline));
    app.vk.hiZSpdPipeline = vk::raii::Pipeline(app.vk.device, rawPipeline);

    // ---- Naive multi-pass Hi-Z pipelines ----
    auto makeNaivePipeline = [&](const uint32_t* spv, uint32_t size,
                                 VkDescriptorType srcType,
                                 vk::raii::DescriptorSetLayout& outDesc,
                                 vk::raii::PipelineLayout& outLayout,
                                 vk::raii::Pipeline& outPipe) {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0;
        b[0].descriptorType = srcType;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dsl.bindingCount = 2; dsl.pBindings = b;
        VkDescriptorSetLayout rawDsl = VK_NULL_HANDLE;
        CheckVkResult(vkCreateDescriptorSetLayout(Raw(app.vk.device), &dsl, nullptr, &rawDsl));
        outDesc = vk::raii::DescriptorSetLayout(app.vk.device, rawDsl);

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 4};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkDescriptorSetLayout rawDslRef = Raw(outDesc);
        pli.setLayoutCount = 1; pli.pSetLayouts = &rawDslRef;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        VkPipelineLayout rawPli = VK_NULL_HANDLE;
        CheckVkResult(vkCreatePipelineLayout(Raw(app.vk.device), &pli, nullptr, &rawPli));
        outLayout = vk::raii::PipelineLayout(app.vk.device, rawPli);

        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = size; sm.pCode = spv;
        VkShaderModule rawSm = VK_NULL_HANDLE;
        CheckVkResult(vkCreateShaderModule(Raw(app.vk.device), &sm, nullptr, &rawSm));
        vk::raii::ShaderModule m(app.vk.device, rawSm);

        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = Raw(m);
        cp.stage.pName = "main";
        cp.layout = Raw(outLayout);
        VkPipeline rawPipe = VK_NULL_HANDLE;
        CheckVkResult(vkCreateComputePipelines(Raw(app.vk.device), VK_NULL_HANDLE, 1, &cp, nullptr, &rawPipe));
        outPipe = vk::raii::Pipeline(app.vk.device, rawPipe);
    };

    makeNaivePipeline(hiz_naive_init_spv, hiz_naive_init_spv_size,
                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      app.vk.hiZNaiveInitDescLayout,
                      app.vk.hiZNaiveInitPipelineLayout,
                      app.vk.hiZNaiveInitPipeline);

    makeNaivePipeline(hiz_naive_step_spv, hiz_naive_step_spv_size,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                      app.vk.hiZNaiveStepDescLayout,
                      app.vk.hiZNaiveStepPipelineLayout,
                      app.vk.hiZNaiveStepPipeline);
}

void CreateMeshletDebugPipeline(App& app, uint32_t width, uint32_t height) {
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo dslCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    dslCI.bindingCount = 2;
    dslCI.pBindings = bindings;
    VkDescriptorSetLayout rawDescLayout = VK_NULL_HANDLE;
    CheckVkResult(vkCreateDescriptorSetLayout(Raw(app.vk.device), &dslCI, nullptr, &rawDescLayout));
    app.vk.meshletDebugDescLayout = vk::raii::DescriptorSetLayout(app.vk.device, rawDescLayout);

    VkPushConstantRange pcRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) * 2 + sizeof(glm::vec4)};
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1;
    VkDescriptorSetLayout meshletDescLayout = Raw(app.vk.meshletDebugDescLayout);
    plCI.pSetLayouts = &meshletDescLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pcRange;
    VkPipelineLayout rawPipelineLayout = VK_NULL_HANDLE;
    CheckVkResult(vkCreatePipelineLayout(Raw(app.vk.device), &plCI, nullptr, &rawPipelineLayout));
    app.vk.meshletDebugPipelineLayout = vk::raii::PipelineLayout(app.vk.device, rawPipelineLayout);

    auto makeShader = [&](const uint32_t* spv, uint32_t size) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = size;
        ci.pCode = spv;
        VkShaderModule rawShaderModule = VK_NULL_HANDLE;
        CheckVkResult(vkCreateShaderModule(Raw(app.vk.device), &ci, nullptr, &rawShaderModule));
        return vk::raii::ShaderModule(app.vk.device, rawShaderModule);
    };
    vk::raii::ShaderModule vertMod = makeShader(meshlet_aabb_debug_vert_spv, meshlet_aabb_debug_vert_spv_size);
    vk::raii::ShaderModule fragMod = makeShader(meshlet_aabb_debug_frag_spv, meshlet_aabb_debug_frag_spv_size);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = Raw(vertMod);
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = Raw(fragMod);
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo viCI{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo iaCI{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    iaCI.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    VkViewport viewport{0.f, 0.f, (float)width, (float)height, 0.f, 1.f};
    VkRect2D scissor{{0, 0}, {width, height}};
    VkPipelineViewportStateCreateInfo vpCI{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpCI.viewportCount = 1;
    vpCI.pViewports = &viewport;
    vpCI.scissorCount = 1;
    vpCI.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo rsCI{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rsCI.polygonMode = VK_POLYGON_MODE_FILL;
    rsCI.cullMode = VK_CULL_MODE_NONE;
    rsCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsCI.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo msCI{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo dsCI{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dsCI.depthTestEnable = VK_FALSE;
    dsCI.depthWriteEnable = VK_FALSE;
    dsCI.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendAttachmentState cbAtt[3]{};
    cbAtt[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAtt[1].colorWriteMask = {};
    cbAtt[2].colorWriteMask = {};
    VkPipelineColorBlendStateCreateInfo cbCI{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbCI.attachmentCount = 3;
    cbCI.pAttachments = cbAtt;

    VkFormat dbgColorFmts[3] = {
        app.vk.colorFormat, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};
    VkPipelineRenderingCreateInfoKHR dbgDynRI{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    dbgDynRI.viewMask                = 0b11u;
    dbgDynRI.colorAttachmentCount    = 3;
    dbgDynRI.pColorAttachmentFormats = dbgColorFmts;
    dbgDynRI.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gpCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpCI.pNext = &dbgDynRI;
    gpCI.stageCount = 2;
    gpCI.pStages = stages;
    gpCI.pVertexInputState = &viCI;
    gpCI.pInputAssemblyState = &iaCI;
    gpCI.pViewportState = &vpCI;
    gpCI.pRasterizationState = &rsCI;
    gpCI.pMultisampleState = &msCI;
    gpCI.pDepthStencilState = &dsCI;
    gpCI.pColorBlendState = &cbCI;
    gpCI.layout = Raw(app.vk.meshletDebugPipelineLayout);
    VkPipeline rawPipeline = VK_NULL_HANDLE;
    CheckVkResult(vkCreateGraphicsPipelines(Raw(app.vk.device), VK_NULL_HANDLE, 1, &gpCI, nullptr, &rawPipeline));
    app.vk.meshletDebugPipeline = vk::raii::Pipeline(app.vk.device, rawPipeline);
}

void CreateFrameResources(App& app, VkFormat colorFormat) {
    VkCommandPoolCreateInfo cpCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpCI.queueFamilyIndex = app.vk.queueFamily;
    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool rawCommandPool = VK_NULL_HANDLE;
    CheckVkResult(vkCreateCommandPool(Raw(app.vk.device), &cpCI, nullptr, &rawCommandPool));
    app.vk.cmdPool = vk::raii::CommandPool(app.vk.device, rawCommandPool);

    EyeSwapchain& sc = app.xr.swapchains[0];
    uint32_t imgCount = (uint32_t)sc.images.size();
    sc.hiZW = (sc.width + 1u) / 2u;
    sc.hiZH = (sc.height + 1u) / 2u;
    sc.hiZMipCount = 0;
    for (uint32_t w = sc.hiZW, h = sc.hiZH; w > 0 && h > 0; w = (w + 1u) / 2u, h = (h + 1u) / 2u) {
        sc.hiZMipCount++;
        if (w == 1u && h == 1u) break;
    }

    CreateImage(app.vk, sc.width, sc.height,
                VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                sc.depthImage,
                2);

    VkImageViewCreateInfo dvCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dvCI.image    = Raw(sc.depthImage);
    dvCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    dvCI.format   = VK_FORMAT_D32_SFLOAT;
    dvCI.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT,
                             0, 1, 0, 2};
    VkImageView rawDepthView = VK_NULL_HANDLE;
    CheckVkResult(vkCreateImageView(Raw(app.vk.device), &dvCI, nullptr, &rawDepthView));
    sc.depthView = vk::raii::ImageView(app.vk.device, rawDepthView);
    sc.depthSampleView = CreateImageView(
        app.vk, Raw(sc.depthImage), VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, 1, 0, 2);

    for (uint32_t ping = 0; ping < 2; ++ping) {
        CreateImage(app.vk, sc.width, sc.height,
                    VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    sc.prevDepthImages[ping],
                    2);

        VkImageViewCreateInfo pvCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        pvCI.image    = Raw(sc.prevDepthImages[ping]);
        pvCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        pvCI.format   = VK_FORMAT_R32_SFLOAT;
        pvCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
        VkImageView rawPrevDepthView = VK_NULL_HANDLE;
        CheckVkResult(vkCreateImageView(Raw(app.vk.device), &pvCI, nullptr, &rawPrevDepthView));
        sc.prevDepthViews[ping] = vk::raii::ImageView(app.vk.device, rawPrevDepthView);

        CreateImage(app.vk, sc.width, sc.height,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    sc.motionImages[ping],
                    2);
        pvCI.image  = Raw(sc.motionImages[ping]);
        pvCI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkImageView rawMotionView = VK_NULL_HANDLE;
        CheckVkResult(vkCreateImageView(Raw(app.vk.device), &pvCI, nullptr, &rawMotionView));
        sc.motionViews[ping] = vk::raii::ImageView(app.vk.device, rawMotionView);

        CreateImage(app.vk, sc.hiZW, sc.hiZH,
                    VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    sc.hiZImages[ping],
                    2, sc.hiZMipCount);
        sc.hiZViews[ping] = CreateImageView(
            app.vk, Raw(sc.hiZImages[ping]), VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, sc.hiZMipCount, 0, 2);
        sc.hiZMipViews[ping].clear();
        sc.hiZMipViews[ping].reserve(sc.hiZMipCount);
        for (uint32_t level = 0; level < sc.hiZMipCount; ++level) {
            sc.hiZMipViews[ping].emplace_back(CreateImageView(
                app.vk, Raw(sc.hiZImages[ping]), VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_VIEW_TYPE_2D_ARRAY, level, 1, 0, 2));
        }

    }

    for (uint32_t i = 0; i < imgCount; i++) {
        SwapchainImage& si = sc.images[i];

        VkImageViewCreateInfo civCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        civCI.image    = si.image;
        civCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        civCI.format   = colorFormat;
        civCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
        VkImageView rawSwapchainView = VK_NULL_HANDLE;
        CheckVkResult(vkCreateImageView(Raw(app.vk.device), &civCI, nullptr, &rawSwapchainView));
        si.view = vk::raii::ImageView(app.vk.device, rawSwapchainView);
    }

    VkCommandBufferAllocateInfo cbAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool        = Raw(app.vk.cmdPool);
    cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = imgCount;
    std::vector<VkCommandBuffer> rawCommandBuffers(imgCount, VK_NULL_HANDLE);
    CheckVkResult(vkAllocateCommandBuffers(Raw(app.vk.device), &cbAI, rawCommandBuffers.data()));
    app.vk.cmdBuffers.clear();
    app.vk.cmdBuffers.reserve(imgCount);
    for (VkCommandBuffer rawCommandBuffer : rawCommandBuffers) {
        app.vk.cmdBuffers.emplace_back(app.vk.device, rawCommandBuffer, Raw(app.vk.cmdPool));
    }

    VkFenceCreateInfo fCI{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence rawFence = VK_NULL_HANDLE;
    CheckVkResult(vkCreateFence(Raw(app.vk.device), &fCI, nullptr, &rawFence));
    app.vk.fence = vk::raii::Fence(app.vk.device, rawFence);

    for (uint32_t ping = 0; ping < 2; ++ping) {
        TransitionImageLayoutNow(app.vk, Raw(sc.hiZImages[ping]), VK_IMAGE_ASPECT_COLOR_BIT,
                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 2, sc.hiZMipCount);
        auto cmdBuffers = app.vk.device.allocateCommandBuffers(
            vk::CommandBufferAllocateInfo(Raw(app.vk.cmdPool), vk::CommandBufferLevel::ePrimary, 1));
        auto& cmd = cmdBuffers.front();
        cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        VkClearColorValue clearVal{};
        clearVal.float32[0] = 0.0f;
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, sc.hiZMipCount, 0, 2};
        vkCmdClearColorImage(Raw(cmd), Raw(sc.hiZImages[ping]),
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &range);
        cmd.end();
        vk::CommandBuffer submitCmd = Raw(cmd);
        vk::SubmitInfo si{};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &submitCmd;
        app.vk.queue.submit(si, nullptr);
        app.vk.queue.waitIdle();
        TransitionImageLayoutNow(app.vk, Raw(sc.hiZImages[ping]), VK_IMAGE_ASPECT_COLOR_BIT,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                 2, sc.hiZMipCount);
    }
}
