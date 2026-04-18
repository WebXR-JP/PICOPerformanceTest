#pragma once

#include "common.hpp"

void LoadXrExtFunctions(XrInstance instance);
void InitializeOpenXRLoader(App& app);
void CreateXrInstance(App& app);
void CreateVulkanDevice(App& app);
void CreateXrSession(App& app);
void CreateSwapchains(App& app);
void CreateXrInput(App& app);
void PollXrInput(App& app, XrTime predictedDisplayTime);
