#pragma once

#include "common.hpp"

void CreateQueryPool(App& app);
void CreateVsSkinPipeline(App& app, uint32_t width, uint32_t height);
void CreateSkinCullLdsPipeline(App& app);
void CreateHiZSpdPipeline(App& app);
void CreateMeshletDebugPipeline(App& app, uint32_t width, uint32_t height);
void CreateFrameResources(App& app, VkFormat colorFormat);
