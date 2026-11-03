// Render-thread half of the overlay (gpu_overlay.h): uploads the UI's textures and draws its newest frame.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>

#include <plume_render_interface.h>

namespace ae::gpu {

struct OverlayFrame;
struct OverlayTextureData;

// Upload memory that stays valid until the current slot's fence passes.
struct OverlayUpload {
  plume::RenderBuffer* buffer = nullptr;
  uint64_t offset = 0;
  uint8_t* data = nullptr;
};
using OverlayAllocator = std::function<OverlayUpload(uint64_t size, uint64_t align)>;

// After each present: hands the app's frame callback (SetOverlayFrameCallback) its turn.
void NotifyOverlayPresented();

class OverlayPass {
 public:
  bool Create(plume::RenderDevice* device, plume::RenderFormat target_format);
  void Destroy();
  // Outside a render pass: takes the newest UI frame and uploads textures it hasn't seen; false when there's nothing
  // to draw. `slot` keeps the frame alive until that slot's fence passes.
  bool Prepare(plume::RenderCommandList* list, uint32_t slot, const OverlayAllocator& upload);
  // With the target bound: draws the prepared frame over it.
  void Draw(plume::RenderCommandList* list, uint32_t width, uint32_t height, const OverlayAllocator& upload);

 private:
  struct HostTexture {
    std::weak_ptr<OverlayTextureData> owner;
    std::unique_ptr<plume::RenderTexture> texture;
    std::unique_ptr<plume::RenderDescriptorSet> set;
  };

  plume::RenderDevice* device_ = nullptr;
  std::unique_ptr<plume::RenderSampler> point_sampler_, linear_sampler_;
  plume::RenderDescriptorSetBuilder set_builder_;
  std::unique_ptr<plume::RenderPipelineLayout> layout_;
  std::unique_ptr<plume::RenderShader> vs_, ps_;
  std::unique_ptr<plume::RenderPipeline> pipeline_;
  std::shared_ptr<OverlayTextureData> white_;
  std::unordered_map<const OverlayTextureData*, HostTexture> textures_;
  std::array<std::shared_ptr<const OverlayFrame>, 4> in_flight_;
  const OverlayFrame* prepared_ = nullptr;
};

}  // namespace ae::gpu
