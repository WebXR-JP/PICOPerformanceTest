#pragma once

#include "common.hpp"

void RenderStereo(App& app, uint32_t imageIdx, uint32_t mvImageIdx, uint32_t aswDepthImageIdx,
                  const std::vector<XrView>& views,
                  uint32_t swapW, uint32_t swapH);
void RenderFrame(App& app);
