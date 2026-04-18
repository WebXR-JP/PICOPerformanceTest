#pragma once

#include "common.hpp"

void CreateBuffer(VulkanCtx& vk, VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags memProps,
                  vma::raii::Buffer& buffer);
void UploadBufferData(VulkanCtx& vk, const vma::raii::Buffer& dstBuffer, const void* srcData, VkDeviceSize size);
void CreateImage(VulkanCtx& vk, uint32_t w, uint32_t h,
                 VkFormat format, VkImageUsageFlags usage,
                 vma::raii::Image& image,
                 uint32_t arrayLayers = 1,
                 uint32_t mipLevels = 1);
void TransitionImageLayoutNow(VulkanCtx& vk,
                              VkImage image,
                              VkImageAspectFlags aspectMask,
                              VkImageLayout oldLayout,
                              VkImageLayout newLayout,
                              uint32_t layerCount = 1,
                              uint32_t mipCount = 1);
