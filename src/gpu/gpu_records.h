// Immutable records the guest thread hands to the host render thread.
// Everything the host needs travels in the record, so the host thread never reads guest memory.
#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <type_traits>
#include <variant>
#include <vector>

#include "gpu/gpu_formats.h"

namespace ae::gpu {

// Host resources are named by ids the tracker hands out; the host creates each one the first time a record names it.
inline constexpr uint32_t kNoResource = 0;

struct ResourceDesc {
  uint32_t id = kNoResource;
  uint32_t width = 0;               // guest texels; the host scales render targets and resolve destinations
  uint32_t height = 0;
  HostFormat format = HostFormat::kUnsupported;
  ResourceKind kind = ResourceKind::kColorTarget;
  TextureShape shape = TextureShape::k2D;
  uint8_t samples = 1;              // guest MSAA samples of an EDRAM surface
  uint32_t mip_levels = 1;
  uint32_t depth = 1;               // 3D depth or stacked layers
};

struct Rect {
  int32_t left = 0;
  int32_t top = 0;
  int32_t right = 0;
  int32_t bottom = 0;
};

struct ClearRecord {
  std::array<ResourceDesc, 4> color{};  // D3DCLEAR_TARGET0..3, id 0 when not cleared
  ResourceDesc depth{};                 // cleared when clear_depth or clear_stencil
  bool clear_depth = false;
  bool clear_stencil = false;
  std::array<float, 4> color_value{};   // RGBA from ClearF's float colour argument
  float depth_value = 1.0f;
  uint32_t stencil_value = 0;
  std::vector<Rect> rects;              // empty = whole surface
};

struct ResolveRecord {
  ResourceDesc source{};
  ResourceDesc dest{};
  Rect source_rect{};                   // already clamped to the source surface
  int32_t dest_x = 0;
  int32_t dest_y = 0;
  float scale = 1.0f;                   // 2^exponent bias from D3DRESOLVE_EXPONENTBIAS
  bool clear_color = false;             // D3DRESOLVE_CLEARRENDERTARGET
  std::array<float, 4> clear_color_value{};
  ResourceDesc clear_depth_target{};    // D3DRESOLVE_CLEARDEPTHSTENCIL
  float clear_z = 1.0f;
  uint32_t clear_stencil = 0;
};

struct SwapRecord {
  ResourceDesc front{};
  uint64_t frame = 0;
};

// Filled by the host thread; the guest thread waits on it.
struct ReadbackResult {
  bool ok = false;
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> rgbx;
};

struct ReadbackSlot {
  std::mutex mutex;
  std::condition_variable done_cv;
  bool done = false;
  ReadbackResult result;
};

struct ReadbackRecord {
  std::shared_ptr<ReadbackSlot> slot;
};

// ---------------------------------------------------------------------------------------------------------------
// Draws

struct ShaderEntry;  // gpu_shader_pack.h; immutable once published, shared by both threads

enum class GeometryExpansion : uint8_t { kNone = 0, kRectList = 1, kPointList = 2 };

// Blend, depth and raster state decoded from the device registers; the host hashes it for its pipeline cache.
// Enum fields hold the guest xenos values, translated to plume on the host.
struct PipelineState {
  uint8_t topology = 0;            // xenos::PrimitiveType after host expansion (list, strip, line, point)
  uint8_t geometry = 0;            // GeometryExpansion
  uint8_t cull = 0;                // 0 none, 1 front, 2 back
  uint8_t front_ccw = 0;
  uint8_t wireframe = 0;
  uint8_t depth_enable = 0;
  uint8_t depth_write = 0;
  uint8_t depth_func = 7;          // xenos::CompareFunction
  uint8_t stencil_enable = 0;
  uint8_t stencil_func = 7;
  uint8_t stencil_fail = 0;
  uint8_t stencil_zfail = 0;
  uint8_t stencil_pass = 0;
  uint8_t stencil_func_bf = 7;
  uint8_t stencil_fail_bf = 0;
  uint8_t stencil_zfail_bf = 0;
  uint8_t stencil_pass_bf = 0;
  uint8_t stencil_ref = 0;
  uint8_t stencil_read_mask = 0xFF;
  uint8_t stencil_write_mask = 0xFF;
  uint8_t color_count = 0;         // bound colour targets, packed from slot 0
  uint8_t depth_format = 0;        // HostFormat of the depth target, 0 = none
  std::array<uint8_t, 4> color_format{};  // HostFormat per colour target
  uint8_t _pad0[2] = {};           // keeps the hashed bytes free of implicit padding
  std::array<uint32_t, 4> blend{};        // RB_BLENDCONTROL per target (after GpuSetBlendControl overrides)
  std::array<uint8_t, 4> blend_enable{};
  std::array<uint8_t, 4> write_mask{};    // RGBA bits
  uint8_t use_trimmed_vs = 0;
  uint8_t clip_planes = 0;         // user clip planes through the pack's clip_planes GS
  uint8_t _pad1[2] = {};
  int32_t depth_bias = 0;          // PA_SU_POLY_OFFSET in host depth-bias units
  float slope_bias = 0;            // PA_SU_POLY_OFFSET scale per guest pixel
  uint32_t vs_entry = 0;           // ShaderEntry::id
  uint32_t ps_entry = 0;
};
static_assert(std::is_trivially_copyable_v<PipelineState>);
static_assert(sizeof(PipelineState) == 72, "PipelineState is hashed as bytes and must not gain padding");

// One vertex fetch constant's data: the host places it and rewrites dword 0 of that constant to the placement.
struct StreamData {
  uint32_t fetch_index = 0;   // hardware vertex fetch constant 0-95
  uint32_t guest_base = 0;    // physical byte address
  uint32_t size = 0;          // bytes
  uint64_t key = 0;           // range key; 0 = inline data private to this draw
  uint32_t version = 0;
  bool persistent = false;    // kept across frames; otherwise shared by the draws of one frame
  std::shared_ptr<const std::vector<uint8_t>> bytes;  // null when the host already holds (key, version)
};

// Guest texture contents decoded to the host format, one entry per mip level. A decode that runs on a worker
// thread publishes through ready; the host waits before reading the levels.
struct TextureUpload {
  std::vector<std::vector<uint8_t>> levels;
  std::vector<uint32_t> row_pitch;  // bytes per block row per level
  mutable std::mutex mutex;
  mutable std::condition_variable cv;
  bool ready = true;
  void Wait() const {
    std::unique_lock lock(mutex);
    cv.wait(lock, [this] { return ready; });
  }
  void Publish() {
    {
      std::lock_guard lock(mutex);
      ready = true;
    }
    cv.notify_all();
  }
};

struct TextureBinding {
  uint8_t slot = 0;           // b4 index
  uint8_t stage = 1;          // 0 = vertex, 1 = pixel
  uint8_t dimension = 1;      // 1 = 2D, 2 = 3D, 3 = cube
  ResourceDesc texture{};     // id 0 = bind the blank of that dimension
  uint32_t version = 0;
  std::shared_ptr<const TextureUpload> upload;  // new contents for (id, version), else null
  std::array<uint8_t, 4> swizzle{0, 1, 2, 3};   // fetch-constant swizzle: 0-3 = xyzw, 4 = 0, 5 = 1
};

// Sampler state decoded from the fetch constant with the translator's overrides, packed for the host sampler cache:
// bits 0-8 clamp x/y/z, 9 mag linear, 10 min linear, 11-12 mip (0 point, 1 linear, 2 base level only),
// 13-15 anisotropy (0 off, n = 2^(n-1):1), 16-17 border colour, 18-21 min LOD, 22-25 max LOD.
struct SamplerBinding {
  uint8_t slot = 0;
  uint8_t sampler = 0;        // index into the fixed palette, used when the draw's samplers do not fit a cached set
  uint8_t stage = 1;
  uint32_t state = 0;
};

// b0 system constants, byte offsets per docs/research/pack_contract.md section 1.3.
struct SystemConstants {
  uint32_t flags = 0;                 // 0
  float pixel_position_scale[2] = {1, 1};  // 4: host pixel coordinates to guest pixel coordinates
  uint32_t line_loop_closing_index = 0;  // 12
  uint32_t vertex_index_endian = 0;   // 16
  uint32_t vertex_index_offset = 0;   // 20
  uint32_t vertex_index_min = 0;      // 24
  uint32_t vertex_index_max = 0xFFFFFF;  // 28
  float user_clip_planes[6][4] = {};  // 32
  float ndc_scale[3] = {1, 1, 1};     // 128
  float _pad1 = 0;
  float ndc_offset[3] = {};           // 144
  float _pad2 = 0;
  float point_size[4] = {};           // 160: constant diameter xy, min/max zw, read by the point GS
  float _pad3[11] = {};
  float alpha_test_reference = 0;     // 220
  float _pad4[4] = {};
  float color_exp_bias[4] = {1, 1, 1, 1};  // 240
  uint32_t feature_mask = 0;          // 256
  uint32_t _pad5 = 0;
  float shadow_texel[2] = {};         // 264
};
static_assert(sizeof(SystemConstants) == 272);
static_assert(offsetof(SystemConstants, ndc_scale) == 128);
static_assert(offsetof(SystemConstants, alpha_test_reference) == 220);
static_assert(offsetof(SystemConstants, color_exp_bias) == 240);

struct DrawConstants {
  SystemConstants system{};
  std::array<float, 256 * 4> vs_float{};
  std::array<float, 256 * 4> ps_float{};
  std::array<uint32_t, 40> bool_loop{};   // VS bools, PS bools, VS loops, PS loops
  std::array<uint32_t, 32 * 6> fetch{};   // host-order dwords; vertex dword 0 is rewritten by the host
  std::array<uint32_t, 32 * 4> vfetch{};  // b5 vfetch layout records (contract 2.4)
};

struct PreparedIndices {
  uint32_t count = 0;
  uint8_t topology = 0;
  std::shared_ptr<const std::vector<uint8_t>> bytes;
};

struct PreparedDrawState {
  PipelineState state{};
  std::array<float, 6> viewport{};
  Rect scissor{};
  bool depth_clip = true;
  bool blend_constant_used = false;
  std::array<float, 4> blend_constant{};
  std::vector<SamplerBinding> samplers;
};

struct DrawRecord {
  PipelineState state{};
  const ShaderEntry* vs = nullptr;
  const ShaderEntry* ps = nullptr;
  std::array<ResourceDesc, 4> color{};
  ResourceDesc depth{};
  float viewport[6] = {};             // x, y, w, h, min z, max z in target pixels
  Rect scissor{};
  bool depth_clip = true;
  bool blend_constant_used = false;   // a bound target blends with a constant factor
  float blend_constant[4] = {};       // RB_BLEND_RED..ALPHA, or the alpha in all four for constant-alpha factors
  std::shared_ptr<DrawConstants> constants;
  std::vector<StreamData> streams;
  std::vector<TextureBinding> textures;  // PS bindings
  std::vector<SamplerBinding> samplers;
  // Geometry: indices are already little-endian and expanded for quads and fans, and travel like a stream (a key
  // and version name the converted run, bytes come along only when the host lacks that version, inline draws use
  // key 0 with private bytes). The host fills index_offset when it places them.
  bool indexed = false;
  bool index32 = false;
  uint32_t count = 0;                 // vertices or indices to draw
  StreamData index;
  uint64_t index_offset = 0;
  std::shared_future<PreparedIndices> prepared_indices;
  bool send_prepared_indices = false;
  std::shared_ptr<PreparedDrawState> prepared_state;
};

// Host texture ids the guest no longer refers to (resolve aliases retired by a CPU write or a new registration).
struct ReleaseRecord {
  std::vector<uint32_t> ids;
};

using Record = std::variant<ClearRecord, ResolveRecord, SwapRecord, ReadbackRecord, DrawRecord, ReleaseRecord>;

// One queue item: the records since the last hand-off, ending at a Swap or a readback request.
struct Packet {
  std::vector<Record> records;
};

}  // namespace ae::gpu
