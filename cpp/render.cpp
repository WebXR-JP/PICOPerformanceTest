#include "render.hpp"

#include "mesh.hpp"
#include "openxr_setup.hpp"

glm::mat4 CubeModelMatrix() {
    return glm::mat4(1.f);
}

glm::mat4 ComputeMVP(const XrView& view, const glm::mat4& model,
                     const glm::vec3& playerPos, float playerYaw) {
    const XrFovf& fov = view.fov;
    float l = tanf(fov.angleLeft);
    float r = tanf(fov.angleRight);
    float d = tanf(fov.angleDown);
    float u = tanf(fov.angleUp);
    float zNear = 0.05f, zFar = 100.f;
    glm::mat4 proj(0.f);
    proj[0][0] = 2.f / (r - l);
    proj[1][1] = -2.f / (u - d);
    proj[2][0] = (r + l) / (r - l);
    proj[2][1] = (u + d) / (u - d);
    proj[2][2] = zNear / (zFar - zNear);
    proj[2][3] = -1.f;
    proj[3][2] = (zNear * zFar) / (zFar - zNear);

    const XrQuaternionf& q = view.pose.orientation;
    const XrVector3f&    p = view.pose.position;
    glm::mat4 rot   = glm::mat4_cast(glm::quat(q.w, q.x, q.y, q.z));
    glm::mat4 trans = glm::translate(glm::mat4(1.f), glm::vec3(p.x, p.y, p.z));
    glm::mat4 playerYawMat = glm::rotate(glm::mat4(1.f), playerYaw, glm::vec3(0, 1, 0));
    glm::mat4 playerTrans  = glm::translate(glm::mat4(1.f), playerPos);
    glm::mat4 view_mat = glm::inverse(playerTrans * playerYawMat * trans * rot);

    return proj * view_mat * model;
}

void RenderStereo(App& app, uint32_t imageIdx,
                  const std::vector<XrView>& views,
                  uint32_t swapW, uint32_t swapH) {
    const uint32_t prevDepthReadIdx  = app.frameParity & 1u;
    const uint32_t prevDepthWriteIdx = prevDepthReadIdx ^ 1u;
    VkCommandBuffer cmd = Raw(app.vk.cmdBuffers[imageIdx]);
    EyeSwapchain&   sc  = app.xr.swapchains[0];

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CheckVkResult(vkBeginCommandBuffer(cmd, &bi));

    glm::mat4 model = CubeModelMatrix();
    glm::mat4 mvp0  = ComputeMVP(views[0], model, app.playerPos, app.playerYaw);
    glm::mat4 mvp1  = ComputeMVP(views[1], model, app.playerPos, app.playerYaw);


    if (IsValid(app.vk.queryPool)) {
        vkCmdResetQueryPool(cmd, Raw(app.vk.queryPool), 0, TS_COUNT);
    }

    if (IsValid(app.vk.queryPool)) {
        vkCmdWriteTimestamp2KHR(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT_KHR,
                               Raw(app.vk.queryPool), TS_CS_BEGIN);
    }

    uint32_t drawCmdInit[5] = {
        0,
        (uint32_t)app.instCount,
        0,
        0,
        0,
    };
    vkCmdUpdateBuffer(cmd, Raw(app.vk.bfDrawCmdBuffer), 0, sizeof(drawCmdInit), drawCmdInit);
    uint32_t cullStatsInit[40] = {0};
    vkCmdUpdateBuffer(cmd, Raw(app.vk.cullStatsBuffer), 0, sizeof(cullStatsInit), cullStatsInit);

    VkMemoryBarrier2KHR fillToCs{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR};
    fillToCs.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR;
    fillToCs.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    fillToCs.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    fillToCs.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT_KHR;
    {
        VkDependencyInfoKHR dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &fillToCs;
        vkCmdPipelineBarrier2KHR(cmd, &dep);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Raw(app.vk.skinCullLdsPipeline));
    {
        VkDescriptorBufferInfo bufs[7] = {
            {Raw(app.vk.meshletBuffer),      0, VK_WHOLE_SIZE},
            {Raw(app.vk.vertexBuffer),       0, VK_WHOLE_SIZE},
            {Raw(app.vk.boneBuffer),         0, VK_WHOLE_SIZE},
            {Raw(app.vk.indexBuffer),        0, VK_WHOLE_SIZE},
            {Raw(app.vk.compactIndexBuffer), 0, VK_WHOLE_SIZE},
            {Raw(app.vk.bfDrawCmdBuffer),    0, VK_WHOLE_SIZE},
            {Raw(app.vk.cullStatsBuffer),    0, VK_WHOLE_SIZE},
        };
        VkDescriptorImageInfo hiZImg{};
        hiZImg.sampler     = Raw(app.vk.hiZSampler);
        hiZImg.imageView   = Raw(sc.hiZViews[prevDepthReadIdx]);
        hiZImg.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w[8]{};
        for (uint32_t i = 0; i < 6; i++) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo     = &bufs[i];
        }
        w[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[6].dstBinding      = 6;
        w[6].descriptorCount = 1;
        w[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[6].pImageInfo      = &hiZImg;
        w[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[7].dstBinding      = 7;
        w[7].descriptorCount = 1;
        w[7].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[7].pBufferInfo     = &bufs[6];
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  Raw(app.vk.skinCullLdsPipelineLayout), 0, 8, w);
    }

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
    glm::mat4 playerYawMat = glm::rotate(glm::mat4(1.f), app.playerYaw, glm::vec3(0, 1, 0));
    auto cam_world = [&](const XrView& v) {
        glm::vec3 vp(v.pose.position.x, v.pose.position.y, v.pose.position.z);
        glm::vec3 w = glm::vec3(playerYawMat * glm::vec4(vp, 1.f)) + app.playerPos;
        return glm::vec4(w, 0.f);
    };
    lpc.camPos[0]    = cam_world(views[0]);
    lpc.camPos[1]    = cam_world(views[1]);
    lpc.meshletCount = app.vk.meshletCount;
    lpc.cullEnabled  = 0u;  // 一時的に triangle backface cull 無効
    lpc.frustumEnabled = (uint32_t)app.mode7Frustum;
    lpc.prevDepthEnabled = app.prevDepthValid ? 1u : 0u;
    lpc.prevDepthParams = glm::vec4((float)swapW, (float)swapH, 0.0003f,
                                    (float)app.xr.swapchains[0].hiZMipCount);
    vkCmdPushConstants(cmd, Raw(app.vk.skinCullLdsPipelineLayout),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lpc), &lpc);
    if (app.prevDepthValid) {
        VkImageMemoryBarrier2KHR prevDepthToRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR};
        prevDepthToRead.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        prevDepthToRead.srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
        prevDepthToRead.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        prevDepthToRead.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT_KHR;
        prevDepthToRead.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        prevDepthToRead.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        prevDepthToRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prevDepthToRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prevDepthToRead.image               = Raw(sc.hiZImages[prevDepthReadIdx]);
        prevDepthToRead.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                               sc.hiZMipCount, 0, 2};
        VkDependencyInfoKHR dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &prevDepthToRead;
        vkCmdPipelineBarrier2KHR(cmd, &dep);
    }
    vkCmdDispatch(cmd, app.vk.meshletCount, 1, 1);

    if (IsValid(app.vk.queryPool)) {
        vkCmdWriteTimestamp2KHR(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
                               Raw(app.vk.queryPool), TS_CS_END);
    }

    VkMemoryBarrier2KHR csToDraw{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR};
    csToDraw.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    csToDraw.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
    csToDraw.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT_KHR
                           | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT_KHR;
    csToDraw.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT_KHR
                           | VK_ACCESS_2_INDEX_READ_BIT_KHR;
    {
        VkDependencyInfoKHR dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &csToDraw;
        vkCmdPipelineBarrier2KHR(cmd, &dep);
    }

    if (IsValid(app.vk.queryPool)) {
        vkCmdWriteTimestamp2KHR(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT_KHR,
                               Raw(app.vk.queryPool), TS_GFX_BEGIN);
    }

    {
        VkImageMemoryBarrier2KHR barriers[4]{};
        for (auto& b : barriers) {
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
            b.srcStageMask        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT_KHR;
            b.srcAccessMask       = {};
            b.dstStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;
            b.dstAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT_KHR;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        }
        barriers[0].image            = sc.images[imageIdx].image;
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
        barriers[1].dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR
                                  | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT_KHR;
        barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR
                                  | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT_KHR;
        barriers[1].image            = Raw(sc.depthImage);
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 2};
        barriers[2].image            = Raw(sc.prevDepthImages[prevDepthWriteIdx]);
        barriers[2].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
        barriers[3].image            = Raw(sc.motionImages[prevDepthWriteIdx]);
        barriers[3].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};

        VkDependencyInfoKHR dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR};
        dep.imageMemoryBarrierCount = 4;
        dep.pImageMemoryBarriers    = barriers;
        vkCmdPipelineBarrier2KHR(cmd, &dep);
    }

    VkRenderingAttachmentInfoKHR colorAtts[3]{};
    colorAtts[0].sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAtts[0].imageView   = Raw(sc.images[imageIdx].view);
    colorAtts[0].imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
    colorAtts[0].loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtts[0].storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtts[0].clearValue.color.float32[0] = 0.1f;
    colorAtts[0].clearValue.color.float32[1] = 0.1f;
    colorAtts[0].clearValue.color.float32[2] = 0.15f;
    colorAtts[0].clearValue.color.float32[3] = 1.0f;

    colorAtts[1].sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAtts[1].imageView   = Raw(sc.prevDepthViews[prevDepthWriteIdx]);
    colorAtts[1].imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
    colorAtts[1].loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtts[1].storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtts[1].clearValue.color.float32[0] = 1.0f;

    colorAtts[2].sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAtts[2].imageView   = Raw(sc.motionViews[prevDepthWriteIdx]);
    colorAtts[2].imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
    colorAtts[2].loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtts[2].storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfoKHR depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
    depthAtt.imageView              = Raw(sc.depthView);
    depthAtt.imageLayout            = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
    depthAtt.loadOp                 = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp                = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {0.0f, 0};

    VkRenderingInfoKHR ri{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
    ri.renderArea.offset        = {0, 0};
    ri.renderArea.extent        = {swapW, swapH};
    ri.layerCount               = 1;
    ri.viewMask                 = 0b11u;
    ri.colorAttachmentCount     = 3;
    ri.pColorAttachments        = colorAtts;
    ri.pDepthAttachment         = &depthAtt;

    vkCmdBeginRenderingKHR(cmd, &ri);

    struct GfxPushConst { glm::mat4 mvp[2]; int32_t aluIters; int32_t vPerFace; int32_t vPerCube; };
    GfxPushConst pc;
    pc.mvp[0]   = mvp0;
    pc.mvp[1]   = mvp1;
    pc.aluIters = (int32_t)app.aluIters;
    pc.vPerFace = (int32_t)app.vk.vPerFace;
    pc.vPerCube = (int32_t)app.vk.vPerCube;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, Raw(app.vk.vsSkinPipeline));
    vkCmdPushConstants(cmd, Raw(app.vk.vsSkinPipelineLayout),
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    {
        VkDescriptorBufferInfo bufs[2] = {
            {Raw(app.vk.vertexBuffer), 0, VK_WHOLE_SIZE},
            {Raw(app.vk.boneBuffer),   0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet w[2]{};
        for (uint32_t i = 0; i < 2; i++) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo     = &bufs[i];
        }
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  Raw(app.vk.vsSkinPipelineLayout), 0, 2, w);
    }

    vkCmdBindIndexBuffer(cmd, Raw(app.vk.compactIndexBuffer), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(cmd, Raw(app.vk.bfDrawCmdBuffer), 0, 1, 0);

    if (app.debugAabbEnabled) {
        struct MeshletDebugPC {
            glm::mat4 mvp[2];
            glm::vec4 viewportAndBias;
        } dbgPc{};
        dbgPc.mvp[0] = mvp0;
        dbgPc.mvp[1] = mvp1;
        dbgPc.viewportAndBias = glm::vec4((float)swapW, (float)swapH, 0.0003f,
                                          (float)app.xr.swapchains[0].hiZMipCount);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, Raw(app.vk.meshletDebugPipeline));
        {
            VkDescriptorBufferInfo meshletBuf{Raw(app.vk.meshletBuffer), 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo hiZImg{};
            hiZImg.sampler     = Raw(app.vk.hiZSampler);
            hiZImg.imageView   = Raw(sc.hiZViews[prevDepthReadIdx]);
            hiZImg.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet w[2]{};
            w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[0].dstBinding      = 0;
            w[0].descriptorCount = 1;
            w[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[0].pBufferInfo     = &meshletBuf;
            w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[1].dstBinding      = 1;
            w[1].descriptorCount = 1;
            w[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[1].pImageInfo      = &hiZImg;
            vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                      Raw(app.vk.meshletDebugPipelineLayout), 0, 2, w);
        }
        vkCmdPushConstants(cmd, Raw(app.vk.meshletDebugPipelineLayout),
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(dbgPc), &dbgPc);
        vkCmdDraw(cmd, 24u, app.vk.meshletCount, 0u, 0u);
    }

    vkCmdEndRenderingKHR(cmd);

    if (IsValid(app.vk.queryPool)) {
        vkCmdWriteTimestamp2KHR(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
                               Raw(app.vk.queryPool), TS_GFX_END);
    }

    VkImageMemoryBarrier2KHR depthToRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR};
    depthToRead.srcStageMask        = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT_KHR;
    depthToRead.srcAccessMask       = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR;
    depthToRead.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    depthToRead.dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT_KHR;
    depthToRead.oldLayout           = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
    depthToRead.newLayout           = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR;
    depthToRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthToRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthToRead.image               = Raw(sc.depthImage);
    depthToRead.subresourceRange    = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 2};
    {
        VkDependencyInfoKHR dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &depthToRead;
        vkCmdPipelineBarrier2KHR(cmd, &dep);
    }

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

    if (IsValid(app.vk.queryPool)) {
        vkCmdWriteTimestamp2KHR(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
                               Raw(app.vk.queryPool), TS_HIZ_BEGIN);
    }

    // ---- Naive multi-pass Hi-Z generation ----
    // Pass 0: source depth → hiZ mip 0
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Raw(app.vk.hiZNaiveInitPipeline));
        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler     = Raw(app.vk.prevDepthSampler);
        srcInfo.imageView   = Raw(sc.depthSampleView);
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR;
        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = Raw(sc.hiZMipViews[prevDepthWriteIdx][0]);
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &srcInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &dstInfo;
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  Raw(app.vk.hiZNaiveInitPipelineLayout), 0, 2, w);
        int32_t pc[4] = {(int32_t)sc.width, (int32_t)sc.height, (int32_t)sc.hiZW, (int32_t)sc.hiZH};
        vkCmdPushConstants(cmd, Raw(app.vk.hiZNaiveInitPipelineLayout),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        uint32_t gx = (sc.hiZW + 7u) / 8u;
        uint32_t gy = (sc.hiZH + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 2u);
    }

    auto mipBarrier = [&](uint32_t mip) {
        VkImageMemoryBarrier2KHR b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR};
        b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
        b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR | VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = Raw(sc.hiZImages[prevDepthWriteIdx]);
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 2};
        VkDependencyInfoKHR dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2KHR(cmd, &dep);
    };

    // Subsequent passes: hiZ mip i-1 → hiZ mip i
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Raw(app.vk.hiZNaiveStepPipeline));
    uint32_t prevW = sc.hiZW, prevH = sc.hiZH;
    for (uint32_t i = 1; i < sc.hiZMipCount; ++i) {
        mipBarrier(i - 1u);  // ensure previous mip's write is visible
        uint32_t dstW = std::max<uint32_t>(1u, (prevW + 1u) / 2u);
        uint32_t dstH = std::max<uint32_t>(1u, (prevH + 1u) / 2u);
        VkDescriptorImageInfo srcInfo{};
        srcInfo.imageView   = Raw(sc.hiZMipViews[prevDepthWriteIdx][i - 1u]);
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = Raw(sc.hiZMipViews[prevDepthWriteIdx][i]);
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[0].pImageInfo = &srcInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &dstInfo;
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  Raw(app.vk.hiZNaiveStepPipelineLayout), 0, 2, w);
        int32_t pc[4] = {(int32_t)prevW, (int32_t)prevH, (int32_t)dstW, (int32_t)dstH};
        vkCmdPushConstants(cmd, Raw(app.vk.hiZNaiveStepPipelineLayout),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        uint32_t gx = (dstW + 7u) / 8u;
        uint32_t gy = (dstH + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 2u);
        prevW = dstW; prevH = dstH;
    }

    // Final barrier: all hiZ mips ready for next-frame cull-shader sample
    {
        VkImageMemoryBarrier2KHR b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR};
        b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
        b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = Raw(sc.hiZImages[prevDepthWriteIdx]);
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, sc.hiZMipCount, 0, 2};
        VkDependencyInfoKHR dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2KHR(cmd, &dep);
    }

    if (IsValid(app.vk.queryPool)) {
        vkCmdWriteTimestamp2KHR(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
                               Raw(app.vk.queryPool), TS_HIZ_END);
    }

    CheckVkResult(vkEndCommandBuffer(cmd));
}

void RenderFrame(App& app) {
    using Clock = App::Clock;
    auto t0 = Clock::now();

    XrFrameWaitInfo fwi{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState    fst{XR_TYPE_FRAME_STATE};
    CHECK_XR(xrWaitFrame(app.xr.session, &fwi, &fst));
    CHECK_XR(xrBeginFrame(app.xr.session, nullptr));

    PollXrInput(app, fst.predictedDisplayTime);

    std::vector<XrCompositionLayerBaseHeader*> layers;
    XrCompositionLayerProjection projLayer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    std::vector<XrCompositionLayerProjectionView> projViews;

    if (fst.shouldRender) {
        uint32_t viewCount = 2;
        std::vector<XrView> views(viewCount, {XR_TYPE_VIEW});
        XrViewLocateInfo vli{XR_TYPE_VIEW_LOCATE_INFO};
        vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        vli.displayTime           = fst.predictedDisplayTime;
        vli.space                 = app.xr.appSpace;
        XrViewState vs{XR_TYPE_VIEW_STATE};
        CHECK_XR(xrLocateViews(app.xr.session, &vli, &vs,
                               viewCount, &viewCount, views.data()));
        const XrView* renderViews = views.data();

        projViews.resize(viewCount);

        EyeSwapchain& sc = app.xr.swapchains[0];

        XrSwapchainImageAcquireInfo acqInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        uint32_t imgIdx = 0;
        CHECK_XR(xrAcquireSwapchainImage(sc.handle, &acqInfo, &imgIdx));

        XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        waitInfo.timeout = XR_INFINITE_DURATION;
        CHECK_XR(xrWaitSwapchainImage(sc.handle, &waitInfo));

        std::vector<XrView> renderViewsVec(viewCount, {XR_TYPE_VIEW});
        for (uint32_t eye = 0; eye < viewCount; ++eye) {
            renderViewsVec[eye] = renderViews[eye];
        }
        RenderStereo(app, imgIdx, renderViewsVec, sc.width, sc.height);

        for (uint32_t eye = 0; eye < viewCount; eye++) {
            projViews[eye] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            projViews[eye].pose = renderViews[eye].pose;
            projViews[eye].fov  = renderViews[eye].fov;
            projViews[eye].subImage.swapchain             = sc.handle;
            projViews[eye].subImage.imageRect.offset      = {0, 0};
            projViews[eye].subImage.imageRect.extent      = {(int32_t)sc.width, (int32_t)sc.height};
            projViews[eye].subImage.imageArrayIndex       = eye;
        }

        VkFence fence = Raw(app.vk.fence);
        CheckVkResult(vkWaitForFences(Raw(app.vk.device), 1, &fence, VK_TRUE, UINT64_MAX));

        // Skinning disabled for culling debug — bones stay at init-time identity
        // {
        //     double elapsed = std::chrono::duration<double>(
        //         App::Clock::now() - app.startTime).count();
        //     UpdateBones(app.vk, (float)elapsed);
        // }

        if (IsValid(app.vk.queryPool)) {
            uint64_t ts[TS_COUNT] = {};
            VkResult qr = vkGetQueryPoolResults(
                Raw(app.vk.device), Raw(app.vk.queryPool), 0, TS_COUNT,
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

        {
            void* p = app.vk.bfDrawCmdBuffer.getAllocation().map();
            app.lastBfIndexCount = *reinterpret_cast<uint32_t*>(p);
            app.vk.bfDrawCmdBuffer.getAllocation().unmap();
        }
        {
            void* p = app.vk.cullStatsBuffer.getAllocation().map();
            uint32_t* stats = reinterpret_cast<uint32_t*>(p);
            app.lastFrustumMeshlets = stats[0];
            app.lastDepthRejectedMeshlets = stats[1];
            app.lastVisibleMeshlets = stats[2];
            for (int i = 0; i < 8; i++) app.lastDeltaHist[i] = stats[3 + i];
            app.lastDeltaHistTotal = stats[11];
            for (int i = 0; i < 6; i++) app.lastMinHHist[i] = stats[12 + i];
            for (int i = 0; i < 6; i++) app.lastDepthLimitHist[i] = stats[18 + i];
            for (int i = 0; i < 6; i++) {
                uint32_t u = stats[24 + i];
                float f; std::memcpy(&f, &u, 4);
                app.lastHiZProbe[i] = f;
            }
            for (int i = 0; i < 6; i++) app.lastMinHHist2[i] = stats[30 + i];
            app.vk.cullStatsBuffer.getAllocation().unmap();
        }
        CheckVkResult(vkResetFences(Raw(app.vk.device), 1, &fence));

        VkCommandBuffer submitCmd = Raw(app.vk.cmdBuffers[imgIdx]);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &submitCmd;
        CheckVkResult(vkQueueSubmit(Raw(app.vk.queue), 1, &si, Raw(app.vk.fence)));
        app.prevDepthValid = true;
        app.frameParity ^= 1u;

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

    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    app.frameMsAccum += ms;
    app.frameCount++;

    double elapsed = std::chrono::duration<double>(t1 - app.lastLogTime).count();
    if (elapsed >= 1.0) {
        double avgMs    = app.frameMsAccum / app.frameCount;
        double fps      = app.frameCount / elapsed;
        int    polysTotal = (int)app.vk.meshletCount * MESHLET_TILE * MESHLET_TILE * 2 * app.instCount;
        uint32_t liveTris = app.lastBfIndexCount / 3;
        LOGI("[PERF] FPS=%.1f  FrameTime=%.2fms  CS=%.3fms  GFX=%.3fms  DS=%.3fms  HiZ=%.3fms  Polys=%d  LiveTris=%u  Meshlets=%u  FrustumM=%u  DepthRejectM=%u  VisibleM=%u",
             fps, avgMs, app.lastCsMs, app.lastGfxMs, app.lastDownsampleMs, app.lastHiZMs,
             polysTotal, liveTris, app.vk.meshletCount,
             app.lastFrustumMeshlets, app.lastDepthRejectedMeshlets, app.lastVisibleMeshlets);
        LOGI("[HIST] minH-depthLimit (n=%u)  <-0.1:%u  -0.1~-.01:%u  -.01~-.001:%u  -.001~0:%u  | 0~.001:%u  .001~.01:%u  .01~.1:%u  >0.1:%u",
             app.lastDeltaHistTotal,
             app.lastDeltaHist[0], app.lastDeltaHist[1], app.lastDeltaHist[2], app.lastDeltaHist[3],
             app.lastDeltaHist[4], app.lastDeltaHist[5], app.lastDeltaHist[6], app.lastDeltaHist[7]);
        LOGI("[HIST] minH          =0:%u  <1e-4:%u  <1e-3:%u  <1e-2:%u  <1e-1:%u  >=1e-1:%u",
             app.lastMinHHist2[0], app.lastMinHHist2[1], app.lastMinHHist2[2],
             app.lastMinHHist2[3], app.lastMinHHist2[4], app.lastMinHHist2[5]);
        LOGI("[HIST] maxH          =0:%u  <1e-4:%u  <1e-3:%u  <1e-2:%u  <1e-1:%u  >=1e-1:%u",
             app.lastMinHHist[0], app.lastMinHHist[1], app.lastMinHHist[2],
             app.lastMinHHist[3], app.lastMinHHist[4], app.lastMinHHist[5]);
        LOGI("[HIST] depthLimit    =0:%u  <1e-4:%u  <1e-3:%u  <1e-2:%u  <1e-1:%u  >=1e-1:%u",
             app.lastDepthLimitHist[0], app.lastDepthLimitHist[1], app.lastDepthLimitHist[2],
             app.lastDepthLimitHist[3], app.lastDepthLimitHist[4], app.lastDepthLimitHist[5]);
        LOGI("[PROBE] L3@(.5,.5) L3@(0,0) L3@(1,1) L0@(.5,.5) L1@(.5,.5) L2@(.5,.5) = %.6f %.6f %.6f %.6f %.6f %.6f",
             app.lastHiZProbe[0], app.lastHiZProbe[1], app.lastHiZProbe[2],
             app.lastHiZProbe[3], app.lastHiZProbe[4], app.lastHiZProbe[5]);
        app.frameMsAccum = 0.0;
        app.frameCount   = 0;
        app.lastLogTime  = t1;
    }
}
