#include "vk_resources.hpp"

void CreateBuffer(VulkanCtx& vk, VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags memProps,
                  vma::raii::Buffer& buffer) {
    vk::BufferCreateInfo bi{};
    bi.size        = size;
    bi.usage       = static_cast<vk::BufferUsageFlags>(usage);
    bi.sharingMode = vk::SharingMode::eExclusive;

    vma::AllocationCreateInfo aci{};
    aci.usage         = vma::MemoryUsage::eUnknown;
    aci.requiredFlags = static_cast<vk::MemoryPropertyFlags>(memProps);
    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        aci.flags = vma::AllocationCreateFlagBits::eMapped
                  | vma::AllocationCreateFlagBits::eHostAccessRandom;
    }
    buffer = vma::raii::Buffer(vk.allocator, bi, aci);
}

void UploadBufferData(VulkanCtx& vk, const vma::raii::Buffer& dstBuffer, const void* srcData, VkDeviceSize size) {
    vk::BufferCreateInfo bi{};
    bi.size        = size;
    bi.usage       = vk::BufferUsageFlagBits::eTransferSrc;
    bi.sharingMode = vk::SharingMode::eExclusive;

    vma::AllocationCreateInfo aci{};
    aci.usage = vma::MemoryUsage::eAuto;
    aci.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite
              | vma::AllocationCreateFlagBits::eMapped;
    vma::raii::Buffer stagingBuffer(vk.allocator, bi, aci);
    void* mapped = stagingBuffer.getAllocation().map();
    std::memcpy(mapped, srcData, (size_t)size);
    stagingBuffer.getAllocation().unmap();

    auto cmdBuffers = vk.device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(Raw(vk.cmdPool), vk::CommandBufferLevel::ePrimary, 1));
    auto& cmd = cmdBuffers.front();

    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    cmd.copyBuffer(Raw(stagingBuffer), Raw(dstBuffer), vk::BufferCopy(0, 0, size));
    cmd.end();

    vk::CommandBuffer submitCmd = Raw(cmd);
    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &submitCmd;
    vk.queue.submit(submitInfo, nullptr);
    vk.queue.waitIdle();
}

void CreateImage(VulkanCtx& vk, uint32_t w, uint32_t h,
                 VkFormat format, VkImageUsageFlags usage,
                 vma::raii::Image& image,
                 uint32_t arrayLayers,
                 uint32_t mipLevels) {
    vk::ImageCreateInfo ci{};
    ci.imageType     = vk::ImageType::e2D;
    ci.format        = static_cast<vk::Format>(format);
    ci.extent        = vk::Extent3D(w, h, 1);
    ci.mipLevels     = mipLevels;
    ci.arrayLayers   = arrayLayers;
    ci.samples       = vk::SampleCountFlagBits::e1;
    ci.tiling        = vk::ImageTiling::eOptimal;
    ci.usage         = static_cast<vk::ImageUsageFlags>(usage);
    ci.sharingMode   = vk::SharingMode::eExclusive;
    ci.initialLayout = vk::ImageLayout::eUndefined;

    vma::AllocationCreateInfo aci{};
    aci.usage         = vma::MemoryUsage::eUnknown;
    aci.requiredFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    image = vma::raii::Image(vk.allocator, ci, aci);
}

void TransitionImageLayoutNow(VulkanCtx& vk,
                              VkImage image,
                              VkImageAspectFlags aspectMask,
                              VkImageLayout oldLayout,
                              VkImageLayout newLayout,
                              uint32_t layerCount,
                              uint32_t mipCount) {
    auto cmdBuffers = vk.device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(Raw(vk.cmdPool), vk::CommandBufferLevel::ePrimary, 1));
    auto& cmd = cmdBuffers.front();

    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask        = vk::PipelineStageFlagBits2::eTopOfPipe;
    barrier.srcAccessMask       = {};
    barrier.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.dstAccessMask       = (newLayout == VK_IMAGE_LAYOUT_GENERAL)
        ? (vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite)
        : vk::AccessFlags2{};
    barrier.oldLayout           = static_cast<vk::ImageLayout>(oldLayout);
    barrier.newLayout           = static_cast<vk::ImageLayout>(newLayout);
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = vk::ImageSubresourceRange(
        static_cast<vk::ImageAspectFlags>(aspectMask), 0, mipCount, 0, layerCount);

    cmd.pipelineBarrier2(vk::DependencyInfo({}, {}, {}, barrier));
    cmd.end();

    vk::CommandBuffer submitCmd = Raw(cmd);
    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &submitCmd;
    vk.queue.submit(submitInfo, nullptr);
    vk.queue.waitIdle();
}
