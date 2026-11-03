// Overlay recording on the UI thread and its replay on the render thread (gpu_overlay.h, gpu_overlay_pass.h).

#include "gpu/gpu_overlay.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <mutex>
#include <utility>
#include <vector>

#include <rex/logging.h>
#include <rex/ui/immediate_drawer.h>

#include "gpu/gpu_overlay_pass.h"
#include "gpu/shaders/overlay_shaders.h"

namespace ae::gpu {

struct OverlayTextureData {
  uint32_t width = 0;
  uint32_t height = 0;
  bool linear = true;
  // RGBA8 rows; the render thread drops them once uploaded.
  std::vector<uint8_t> rgba;
};

struct OverlayDraw {
  std::shared_ptr<OverlayTextureData> texture;
  uint32_t index_count = 0;
  uint32_t first_index = 0;
  int32_t base_vertex = 0;
  float clip[4] = {};  // left, top, right, bottom in the frame's coordinate space
};

struct OverlayFrame {
  float width = 0.0f;
  float height = 0.0f;
  std::vector<rex::ui::ImmediateVertex> vertices;
  std::vector<uint16_t> indices;
  std::vector<OverlayDraw> draws;
};

namespace {

std::mutex g_mutex;
std::shared_ptr<const OverlayFrame> g_latest;
bool g_published = false;
std::function<void()> g_frame_callback;

struct OverlayConstants {
  float scale[2];
  float offset[2];
  uint32_t linear_filter;
};

class OverlayTexture final : public rex::ui::ImmediateTexture {
 public:
  explicit OverlayTexture(std::shared_ptr<OverlayTextureData> data)
      : ImmediateTexture(data->width, data->height), data(std::move(data)) {}
  std::shared_ptr<OverlayTextureData> data;
};

// Records the ImGui drawer's batches into a frame instead of touching the device, which belongs to the render thread.
class OverlayDrawer final : public rex::ui::ImmediateDrawer {
 public:
  std::unique_ptr<rex::ui::ImmediateTexture> CreateTexture(uint32_t width, uint32_t height,
                                                           rex::ui::ImmediateTextureFilter filter, bool,
                                                           const uint8_t* data) override {
    auto texture = std::make_shared<OverlayTextureData>();
    texture->width = std::max(width, 1u);
    texture->height = std::max(height, 1u);
    texture->linear = filter == rex::ui::ImmediateTextureFilter::kLinear;
    const size_t bytes = size_t(texture->width) * texture->height * 4;
    if (data && width && height) {
      texture->rgba.assign(data, data + bytes);
    } else {
      texture->rgba.assign(bytes, 0);
    }
    return std::make_unique<OverlayTexture>(std::move(texture));
  }

  void Begin(rex::ui::UIDrawContext& context, float width, float height) override {
    ImmediateDrawer::Begin(context, width, height);
    frame_ = std::make_shared<OverlayFrame>();
    frame_->width = width;
    frame_->height = height;
  }

  void BeginDrawBatch(const rex::ui::ImmediateDrawBatch& batch) override {
    batch_vertex_ = uint32_t(frame_->vertices.size());
    batch_index_ = uint32_t(frame_->indices.size());
    batch_indexed_ = batch.indices != nullptr;
    frame_->vertices.insert(frame_->vertices.end(), batch.vertices, batch.vertices + batch.vertex_count);
    if (batch_indexed_) frame_->indices.insert(frame_->indices.end(), batch.indices, batch.indices + batch.index_count);
  }

  // ImGui only sends indexed triangles.
  void Draw(const rex::ui::ImmediateDraw& draw) override {
    if (!frame_ || !batch_indexed_ || draw.count <= 0 ||
        draw.primitive_type != rex::ui::ImmediatePrimitiveType::kTriangles) {
      return;
    }
    OverlayDraw d;
    if (draw.texture) d.texture = static_cast<OverlayTexture*>(draw.texture)->data;
    d.index_count = uint32_t(draw.count);
    d.first_index = batch_index_ + uint32_t(draw.index_offset);
    d.base_vertex = int32_t(batch_vertex_) + draw.base_vertex;
    if (draw.scissor) {
      d.clip[0] = draw.scissor_left;
      d.clip[1] = draw.scissor_top;
      d.clip[2] = draw.scissor_right;
      d.clip[3] = draw.scissor_bottom;
    } else {
      d.clip[2] = frame_->width;
      d.clip[3] = frame_->height;
    }
    frame_->draws.push_back(std::move(d));
  }

  void EndDrawBatch() override {}

  void End() override {
    if (frame_) {
      std::lock_guard lock(g_mutex);
      g_latest = std::move(frame_);
      g_published = true;
    }
    frame_.reset();
    ImmediateDrawer::End();
  }

 private:
  std::shared_ptr<OverlayFrame> frame_;
  uint32_t batch_vertex_ = 0;
  uint32_t batch_index_ = 0;
  bool batch_indexed_ = false;
};

}  // namespace

std::unique_ptr<rex::ui::ImmediateDrawer> CreateOverlayDrawer() { return std::make_unique<OverlayDrawer>(); }

void SetOverlayFrameCallback(std::function<void()> callback) {
  std::lock_guard lock(g_mutex);
  g_frame_callback = std::move(callback);
}

void BeginOverlayFrame() {
  std::lock_guard lock(g_mutex);
  g_published = false;
}

void EndOverlayFrame() {
  std::lock_guard lock(g_mutex);
  if (!g_published) g_latest.reset();
}

// Render thread, after a present.
void NotifyOverlayPresented() {
  std::function<void()> callback;
  {
    std::lock_guard lock(g_mutex);
    callback = g_frame_callback;
  }
  if (callback) callback();
}

// ------------------------------------------------------------------------------------------------ render thread

bool OverlayPass::Create(plume::RenderDevice* device, plume::RenderFormat target_format) {
  device_ = device;
  plume::RenderSamplerDesc sampler;
  sampler.minFilter = sampler.magFilter = plume::RenderFilter::NEAREST;
  sampler.mipmapMode = plume::RenderMipmapMode::NEAREST;
  sampler.addressU = sampler.addressV = sampler.addressW = plume::RenderTextureAddressMode::CLAMP;
  point_sampler_ = device->createSampler(sampler);
  sampler.minFilter = sampler.magFilter = plume::RenderFilter::LINEAR;
  linear_sampler_ = device->createSampler(sampler);

  set_builder_.begin();
  set_builder_.addTexture(0);
  set_builder_.addImmutableSampler(1, point_sampler_.get());
  set_builder_.addImmutableSampler(2, linear_sampler_.get());
  set_builder_.end();
  plume::RenderPipelineLayoutBuilder layout;
  layout.begin(false, true);
  layout.addDescriptorSet(set_builder_);
  layout.addPushConstant(0, 1, sizeof(OverlayConstants),
                         plume::RenderShaderStageFlag::VERTEX | plume::RenderShaderStageFlag::PIXEL);
  layout.end();
  layout_ = layout.create(device);

  vs_ = device->createShader(shaders::kOverlay_vs_dxil, sizeof(shaders::kOverlay_vs_dxil), "VsMain",
                             plume::RenderShaderFormat::DXIL);
  ps_ = device->createShader(shaders::kOverlay_ps_dxil, sizeof(shaders::kOverlay_ps_dxil), "PsMain",
                             plume::RenderShaderFormat::DXIL);
  if (!layout_ || !vs_ || !ps_) return false;

  const plume::RenderInputSlot slot(0, sizeof(rex::ui::ImmediateVertex));
  const plume::RenderInputElement elements[] = {
      {"POSITION", 0, 0, plume::RenderFormat::R32G32_FLOAT, 0, offsetof(rex::ui::ImmediateVertex, x)},
      {"TEXCOORD", 0, 1, plume::RenderFormat::R32G32_FLOAT, 0, offsetof(rex::ui::ImmediateVertex, u)},
      {"COLOR", 0, 2, plume::RenderFormat::R8G8B8A8_UNORM, 0, offsetof(rex::ui::ImmediateVertex, color)},
  };
  plume::RenderGraphicsPipelineDesc desc;
  desc.pipelineLayout = layout_.get();
  desc.vertexShader = vs_.get();
  desc.pixelShader = ps_.get();
  desc.inputSlots = &slot;
  desc.inputSlotsCount = 1;
  desc.inputElements = elements;
  desc.inputElementsCount = uint32_t(std::size(elements));
  desc.depthFunction = plume::RenderComparisonFunction::ALWAYS;
  desc.primitiveTopology = plume::RenderPrimitiveTopology::TRIANGLE_LIST;
  desc.cullMode = plume::RenderCullMode::NONE;
  desc.renderTargetCount = 1;
  desc.renderTargetFormat[0] = target_format;
  desc.renderTargetBlend[0] = plume::RenderBlendDesc::AlphaBlend();
  pipeline_ = device->createGraphicsPipeline(desc);

  white_ = std::make_shared<OverlayTextureData>();
  white_->width = white_->height = 1;
  white_->rgba = {0xFF, 0xFF, 0xFF, 0xFF};
  return pipeline_ != nullptr;
}

void OverlayPass::Destroy() {
  prepared_ = nullptr;
  for (auto& frame : in_flight_) frame.reset();
  textures_.clear();
  pipeline_.reset();
  vs_.reset();
  ps_.reset();
  layout_.reset();
  linear_sampler_.reset();
  point_sampler_.reset();
  white_.reset();
  device_ = nullptr;
}

bool OverlayPass::Prepare(plume::RenderCommandList* list, uint32_t slot, const OverlayAllocator& upload) {
  std::shared_ptr<const OverlayFrame> frame;
  {
    std::lock_guard lock(g_mutex);
    frame = g_latest;
  }
  // This slot's fence has passed, so the frame it held (and the textures only it used) can go.
  in_flight_[slot % in_flight_.size()] = frame;
  prepared_ = nullptr;
  std::erase_if(textures_, [](const auto& entry) { return entry.second.owner.expired(); });
  if (!frame || frame->draws.empty() || !pipeline_ || frame->width <= 0.0f || frame->height <= 0.0f) return false;

  std::vector<OverlayTextureData*> fresh;
  for (const OverlayDraw& draw : frame->draws) {
    OverlayTextureData* data = draw.texture ? draw.texture.get() : white_.get();
    auto [it, inserted] = textures_.try_emplace(data);
    if (!inserted && !it->second.owner.expired()) continue;
    HostTexture& host = it->second;
    host = {};
    host.owner = draw.texture ? draw.texture : white_;
    host.texture = device_->createTexture(
        plume::RenderTextureDesc::Texture2D(data->width, data->height, 1, plume::RenderFormat::R8G8B8A8_UNORM));
    host.set = host.texture ? set_builder_.create(device_) : nullptr;
    if (!host.set) {
      REXGPU_WARN("[gpu] overlay texture {}x{} unavailable", data->width, data->height);
      textures_.erase(it);
      continue;
    }
    host.set->setTexture(0, host.texture.get(), plume::RenderTextureLayout::SHADER_READ);
    fresh.push_back(data);
  }
  if (!fresh.empty()) {
    std::vector<plume::RenderTextureBarrier> barriers;
    for (OverlayTextureData* data : fresh) {
      barriers.emplace_back(textures_[data].texture.get(), plume::RenderTextureLayout::COPY_DEST);
    }
    list->barriers(plume::RenderBarrierStage::COPY, barriers.data(), uint32_t(barriers.size()));
    for (OverlayTextureData* data : fresh) {
      const uint32_t row = data->width * 4;
      const uint32_t pitch = (row + 255) & ~255u;
      const OverlayUpload span = upload(uint64_t(pitch) * data->height, 512);
      if (!span.data || data->rgba.size() < size_t(row) * data->height) continue;
      for (uint32_t y = 0; y < data->height; ++y) {
        std::memcpy(span.data + size_t(y) * pitch, data->rgba.data() + size_t(y) * row, row);
      }
      list->copyTextureRegion(plume::RenderTextureCopyLocation::Subresource(textures_[data].texture.get()),
                              plume::RenderTextureCopyLocation::PlacedFootprint(
                                  span.buffer, plume::RenderFormat::R8G8B8A8_UNORM, data->width, data->height, 1,
                                  pitch / 4, span.offset));
      if (data != white_.get()) std::vector<uint8_t>().swap(data->rgba);
    }
    for (auto& barrier : barriers) barrier.layout = plume::RenderTextureLayout::SHADER_READ;
    list->barriers(plume::RenderBarrierStage::GRAPHICS, barriers.data(), uint32_t(barriers.size()));
  }
  prepared_ = frame.get();
  return true;
}

void OverlayPass::Draw(plume::RenderCommandList* list, uint32_t width, uint32_t height,
                       const OverlayAllocator& upload) {
  const OverlayFrame* frame = prepared_;
  prepared_ = nullptr;
  if (!frame || !width || !height || frame->vertices.empty() || frame->indices.empty()) return;
  const uint64_t vertex_bytes = frame->vertices.size() * sizeof(rex::ui::ImmediateVertex);
  const uint64_t index_bytes = (frame->indices.size() * sizeof(uint16_t) + 3) & ~3ull;
  const OverlayUpload vertices = upload(vertex_bytes, 256);
  const OverlayUpload indices = upload(index_bytes, 256);
  if (!vertices.data || !indices.data) return;
  std::memcpy(vertices.data, frame->vertices.data(), vertex_bytes);
  std::memcpy(indices.data, frame->indices.data(), frame->indices.size() * sizeof(uint16_t));

  list->setGraphicsPipelineLayout(layout_.get());
  list->setPipeline(pipeline_.get());
  const plume::RenderVertexBufferView vertex_view(plume::RenderBufferReference(vertices.buffer, vertices.offset),
                                                  uint32_t(vertex_bytes));
  const plume::RenderInputSlot slot(0, sizeof(rex::ui::ImmediateVertex));
  list->setVertexBuffers(0, &vertex_view, 1, &slot);
  const plume::RenderIndexBufferView index_view(plume::RenderBufferReference(indices.buffer, indices.offset),
                                                uint32_t(index_bytes), plume::RenderFormat::R16_UINT);
  list->setIndexBuffer(&index_view);
  list->setViewports(plume::RenderViewport(0.0f, 0.0f, float(width), float(height)));

  // The UI's coordinate space (logical pixels) stretched over the whole target.
  const float sx = float(width) / frame->width;
  const float sy = float(height) / frame->height;
  OverlayConstants constants{};
  constants.scale[0] = 2.0f / frame->width;
  constants.scale[1] = -2.0f / frame->height;
  constants.offset[0] = -1.0f;
  constants.offset[1] = 1.0f;
  for (const OverlayDraw& draw : frame->draws) {
    const OverlayTextureData* data = draw.texture ? draw.texture.get() : white_.get();
    const auto it = textures_.find(data);
    if (it == textures_.end() || !it->second.set) continue;
    const int32_t left = std::clamp(int32_t(draw.clip[0] * sx), 0, int32_t(width));
    const int32_t top = std::clamp(int32_t(draw.clip[1] * sy), 0, int32_t(height));
    const int32_t right = std::clamp(int32_t(draw.clip[2] * sx + 0.5f), 0, int32_t(width));
    const int32_t bottom = std::clamp(int32_t(draw.clip[3] * sy + 0.5f), 0, int32_t(height));
    if (right <= left || bottom <= top) continue;
    list->setScissors(plume::RenderRect(left, top, right, bottom));
    list->setGraphicsDescriptorSet(it->second.set.get(), 0);
    constants.linear_filter = data->linear ? 1u : 0u;
    list->setGraphicsPushConstants(0, &constants, 0, sizeof(constants));
    list->drawIndexedInstanced(draw.index_count, 1, draw.first_index, draw.base_vertex, 0);
  }
}

}  // namespace ae::gpu
