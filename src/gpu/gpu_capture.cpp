// Commit-point capture: ClearF, Resolve, the tiling bracket and Swap build immutable records from the D3D arguments
// and the device in effect, then call through to the runtime so its own bookkeeping stays intact.
// Argument registers follow the XDK d3d9.h prototypes; float arguments take a GPR slot, so late ones land on the stack.

#include <algorithm>
#include <atomic>
#include <mutex>
#include <tuple>
#include <set>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc/context.h>
#include <rex/ppc/func.h>
#include <rex/system/kernel_state.h>
#include <rex/types.h>
#include <rex/ui/presenter.h>

#include "gpu/gpu.h"
#include "gpu/gpu_cmdbuf.h"
#include "gpu/gpu_tracker.h"
#include "gpu/guest/d3d_device.h"
#include "gpu/precise_sleep.h"

REXCVAR_DEFINE_INT32(gpu_trace_frame, -1, "GPU", "Log every captured record during this frame number, -1 = off");
REXCVAR_DEFINE_INT32(gpu_trace_frames, 1, "GPU", "How many consecutive frames a trace covers from gpu_trace_frame");
REXCVAR_DEFINE_INT32(gpu_fps_cap, 60, "GPU", "Pace D3DDevice_Swap to this many frames per second; 0 leaves the title unpaced");
REXCVAR_DEFINE_BOOL(gpu_fps_title, false, "GPU", "Show the guest frame rate in the window title");
REXCVAR_DEFINE_INT32(gpu_trace_at_second, 0, "GPU",
                     "Trace the first frame presented after this many seconds, for screens reached by hand; 0 = off");
REXCVAR_DEFINE_INT32(gpu_dump_every, 0, "GPU",
                     "Write every N-th presented frame to gpu_dumps/frame_<n>.bmp next to the exe, at most 60 files; 0 = off");

extern "C" REX_FUNC(__imp__sub_92124920);
extern "C" REX_FUNC(__imp__sub_92123790);
extern "C" REX_FUNC(__imp__sub_9211DA80);
extern "C" REX_FUNC(__imp__sub_9211F940);
extern "C" REX_FUNC(__imp__sub_9211FCB0);
extern "C" REX_FUNC(__imp__sub_92128978);
extern "C" REX_FUNC(__imp__sub_92129888);
extern "C" REX_FUNC(__imp__sub_92128418);
extern "C" REX_FUNC(__imp__sub_92129018);

namespace {

using ae::gpu::Rect;
using ae::gpu::ResourceDesc;
using ae::gpu::ResourceKind;
using ae::gpu::guest::D3DDevice;
namespace tracker = ae::gpu::tracker;

// D3DCLEAR_* and D3DRESOLVE_* from XDK d3d9types.h.
constexpr uint32_t kClearTarget0 = 0x1;
constexpr uint32_t kClearZBuffer = 0x10;
constexpr uint32_t kClearStencil = 0x20;
constexpr uint32_t kResolveSourceMask = 0x7;
constexpr uint32_t kResolveDepthStencil = 0x4;
constexpr uint32_t kResolveClearRenderTarget = 0x100;
constexpr uint32_t kResolveClearDepthStencil = 0x200;
constexpr int kResolveExponentBiasShift = 26;

template <typename T>
const T* Guest(uint32_t address) {
  return REX_KERNEL_MEMORY()->TranslateVirtual<const T*>(address);
}

// D3DRECT {x1, y1, x2, y2}, LONGs.
Rect ReadRect(uint32_t address) {
  const auto* v = Guest<rex::be<int32_t>>(address);
  return Rect{v[0], v[1], v[2], v[3]};
}

Rect ClampRect(Rect r, uint32_t width, uint32_t height) {
  r.left = std::clamp<int32_t>(r.left, 0, int32_t(width));
  r.right = std::clamp<int32_t>(r.right, r.left, int32_t(width));
  r.top = std::clamp<int32_t>(r.top, 0, int32_t(height));
  r.bottom = std::clamp<int32_t>(r.bottom, r.top, int32_t(height));
  return r;
}

bool Tracing() {
  const int32_t frame = REXCVAR_GET(gpu_trace_frame);
  return frame >= 0 && ae::gpu::FrameIndex() == uint64_t(frame);
}

void Once(std::atomic<bool>& flag, const char* what, uint32_t value) {
  if (!flag.exchange(true)) REXGPU_ERROR("[gpu] {} ({:#x})", what, value);
}

// Union of the open tiling bracket's tile rects, the area a null-rect ClearF covers across all tiles.
bool TileUnion(const D3DDevice* device, Rect& out) {
  const uint32_t count = std::min<uint32_t>(device->tile_count, 16);
  out = Rect{};
  if (!count) return false;
  out = Rect{INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN};
  for (uint32_t i = 0; i < count; ++i) {
    const auto& t = device->tile_rects[i];
    out.left = std::min<int32_t>(out.left, t.left);
    out.top = std::min<int32_t>(out.top, t.top);
    out.right = std::max<int32_t>(out.right, t.right);
    out.bottom = std::max<int32_t>(out.bottom, t.bottom);
  }
  return out.right > out.left && out.bottom > out.top;
}

// Set while BeginTiling 0x9211F940 runs, so its own ClearF is recognised.
thread_local bool t_in_begin_tiling = false;

// 24-bit bottom-up BMP from the RGBX readback.
bool WriteBmp(const std::filesystem::path& path, const rex::ui::RawImage& image) {
  const uint32_t w = image.width, h = image.height;
  const uint32_t row = (w * 3 + 3) & ~3u;
  const uint32_t pixels = row * h;
  uint8_t header[54] = {'B', 'M'};
  auto put = [&](size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) header[at + i] = uint8_t(v >> (8 * i));
  };
  put(2, 54 + pixels);
  put(10, 54);
  put(14, 40);
  put(18, w);
  put(22, h);
  header[26] = 1;
  header[28] = 24;
  put(34, pixels);
  FILE* f = std::fopen(path.string().c_str(), "wb");
  if (!f) return false;
  std::fwrite(header, 1, sizeof(header), f);
  std::vector<uint8_t> line(row, 0);
  for (uint32_t y = 0; y < h; ++y) {
    const uint8_t* src = image.data.data() + size_t(h - 1 - y) * image.stride;
    for (uint32_t x = 0; x < w; ++x) {
      line[x * 3 + 0] = src[x * 4 + 2];
      line[x * 3 + 1] = src[x * 4 + 1];
      line[x * 3 + 2] = src[x * 4 + 0];
    }
    std::fwrite(line.data(), 1, row, f);
  }
  std::fclose(f);
  return true;
}

// gpu_dump_every: reads back the frame queued last and writes it, stopping after 60 files.
void MaybeDumpFrame(uint64_t frame) {
  static std::atomic<int> s_written{0};
  const int32_t every = REXCVAR_GET(gpu_dump_every);
  if (every <= 0 || frame % uint64_t(every) != 0 || s_written.load() >= 60) return;
  rex::ui::RawImage image;
  if (!ae::gpu::ReadbackFrontBuffer(image) || !image.width || !image.height) {
    REXGPU_WARN("[gpu] frame {} dump: readback failed", frame);
    return;
  }
  const auto dir = rex::filesystem::GetExecutableFolder() / "gpu_dumps";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const auto path = dir / ("frame_" + std::to_string(frame) + ".bmp");
  if (WriteBmp(path, image)) {
    s_written.fetch_add(1);
    REXGPU_INFO("[gpu] frame {} dumped to {} ({}x{})", frame, path.string(), image.width, image.height);
  }
}

}  // namespace

namespace ae::gpu::capture {

ClearRecord MakeClear(const D3DDevice* device, const ClearArgs& a) {
  ClearRecord record;
  for (uint32_t i = 0; i < 4; ++i) {
    if (a.flags & (kClearTarget0 << i)) {
      record.color[i] = tracker::Surface(device->render_targets[i], ResourceKind::kColorTarget);
    }
  }
  record.clear_depth = (a.flags & kClearZBuffer) != 0;
  record.clear_stencil = (a.flags & kClearStencil) != 0;
  if (record.clear_depth || record.clear_stencil) {
    record.depth = tracker::Surface(device->depth_stencil, ResourceKind::kDepthTarget);
  }
  record.color_value = a.color;
  record.depth_value = a.z;
  record.stencil_value = a.stencil & 0xFF;
  record.rects = a.rects;
  if (Tracing()) {
    REXGPU_INFO("[gpu] clear flags {:#x} colour {} {} {} {} z {} stencil {} rects {} -> rt0 {} depth {}", a.flags,
                a.color[0], a.color[1], a.color[2], a.color[3],
                record.depth_value, record.stencil_value, record.rects.size(), record.color[0].id, record.depth.id);
  }
  return record;
}

}  // namespace ae::gpu::capture

// D3DDevice_ClearF 0x92124920 (pDevice, Flags, pRect, pColor, Z, Stencil); pColor points at four big-endian floats,
// RGBA. D3DDevice_Clear 0x92124A48 converts its D3DCOLOR and calls this once per rect, so one hook covers both.
REX_HOOK_RAW(sub_92124920) {
  ae::gpu::draw::FinishPending();
  const uint32_t device = ctx.r3.u32;
  const auto* d = Guest<D3DDevice>(device);
  ae::gpu::capture::ClearArgs a;
  a.flags = ctx.r4.u32;
  if (ctx.r6.u32) {
    const auto* v = Guest<rex::be<float>>(ctx.r6.u32);
    a.color = {v[0], v[1], v[2], v[3]};
  }
  a.z = float(ctx.f1.f64);
  a.stencil = ctx.r8.u32;
  Rect tiles;
  if (t_in_begin_tiling && TileUnion(d, tiles)) {
    // BeginTiling clears one tile's size at the origin and hardware replays it per tile, so the host clears them all.
    a.rects.push_back(tiles);
  } else if (ctx.r5.u32) {
    a.rects.push_back(ReadRect(ctx.r5.u32));
  } else if (d->device_flags0 & 0x20) {
    if (TileUnion(d, tiles)) a.rects.push_back(tiles);
  }
  ae::gpu::cmdbuf::NotePending(device);
  if (ae::gpu::cmdbuf::Recording(device)) {
    ae::gpu::cmdbuf::RecordClear(device, a);
  } else {
    ae::gpu::Submit(ae::gpu::capture::MakeClear(d, a));
  }
  __imp__sub_92124920(ctx, base);
}

// D3DDevice_BeginTiling 0x9211F940 (pDevice, Flags, Count, pRects, pClearColor, ClearZ, ClearStencil) clears through
// ClearF, and EndTiling 0x9211FCB0 (pDevice, Flags, pDestPoint, pDestTexture, pClearColor, ClearZ) resolves each tile
// through D3DDevice_Resolve. The bound surfaces are one tile in EDRAM, so Begin marks them whole-area on the host.
REX_HOOK_RAW(sub_9211F940) {
  ae::gpu::draw::FinishPending();
  const uint32_t device = ctx.r3.u32;
  const uint32_t count = ctx.r5.u32;
  const auto* d = Guest<D3DDevice>(device);
  // Registered before the call-through, since the runtime's own clear inside BeginTiling already needs the whole area.
  Rect tiles{};
  if (ctx.r6.u32 && count) {
    tiles = ReadRect(ctx.r6.u32);
    for (uint32_t i = 1; i < std::min<uint32_t>(count, 16); ++i) {
      const Rect t = ReadRect(ctx.r6.u32 + i * 16);
      tiles.left = std::min(tiles.left, t.left);
      tiles.top = std::min(tiles.top, t.top);
      tiles.right = std::max(tiles.right, t.right);
      tiles.bottom = std::max(tiles.bottom, t.bottom);
    }
  }
  if (tiles.right > 0 && tiles.bottom > 0) {
    for (uint32_t i = 0; i < 4; ++i) {
      tracker::RegisterTiledSurface(d->render_targets[i], uint32_t(tiles.right), uint32_t(tiles.bottom));
    }
    tracker::RegisterTiledSurface(d->depth_stencil, uint32_t(tiles.right), uint32_t(tiles.bottom));
  }
  t_in_begin_tiling = true;
  __imp__sub_9211F940(ctx, base);
  t_in_begin_tiling = false;
  // Every distinct bracket shape is logged once: which targets, how many tiles, the area they cover.
  static std::mutex s_seen_mutex;
  static std::set<std::tuple<uint32_t, uint32_t, uint32_t, int32_t, int32_t, int32_t, int32_t>> s_seen;
  const auto shape = std::make_tuple(uint32_t(d->render_targets[0]), uint32_t(d->depth_stencil), count, tiles.left,
                                     tiles.top, tiles.right, tiles.bottom);
  bool fresh = false;
  {
    std::lock_guard lock(s_seen_mutex);
    fresh = s_seen.insert(shape).second;
  }
  if (fresh || Tracing()) {
    REXGPU_INFO("[gpu] tiling bracket: {} tiles over ({},{})-({},{}), rt0 {:#010x} depth {:#010x} flags {:#x}", count,
                tiles.left, tiles.top, tiles.right, tiles.bottom, uint32_t(d->render_targets[0]),
                uint32_t(d->depth_stencil), ctx.r4.u32);
  }
}

REX_HOOK_RAW(sub_9211FCB0) {
  ae::gpu::draw::FinishPending();
  __imp__sub_9211FCB0(ctx, base);
}

// D3DDevice_BeginCommandBuffer 0x92128978 (pDevice, pCommandBuffer, Flags, pInheritTags, pPersistTags, pTilingRects,
// TileCount) and EndCommandBuffer 0x92129888 (pDevice): the recording window on that device.
REX_HOOK_RAW(sub_92128978) {
  ae::gpu::draw::FinishPending();
  ae::gpu::cmdbuf::Begin(ctx.r3.u32, ctx.r4.u32, ctx.r5.u32);
  __imp__sub_92128978(ctx, base);
}

REX_HOOK_RAW(sub_92129888) {
  ae::gpu::draw::FinishPending();
  const uint32_t device = ctx.r3.u32;
  __imp__sub_92129888(ctx, base);
  ae::gpu::cmdbuf::End(device);
}

// D3DDevice_SetCommandBufferPredication 0x92128418 (pDevice, TilePredication, RunPredication).
REX_HOOK_RAW(sub_92128418) {
  ae::gpu::cmdbuf::SetPredication(ctx.r3.u32, ctx.r5.u32);
  __imp__sub_92128418(ctx, base);
}

// D3DDevice_RunCommandBuffer 0x92129018 (pDevice, pCommandBuffer, PredicationSelect).
REX_HOOK_RAW(sub_92129018) {
  ae::gpu::draw::FinishPending();
  ae::gpu::cmdbuf::Run(ctx, base, ctx.r3.u32, ctx.r4.u32, ctx.r5.u32);
  __imp__sub_92129018(ctx, base);
}

// D3DDevice_Resolve 0x92123790 (pDevice, Flags, pSourceRect, pDestTexture, pDestPoint, DestLevel, DestSliceOrFace,
// pClearColor, ClearZ, ClearStencil, pParameters); ClearStencil is the stack slot at r1+0x5C.
REX_HOOK_RAW(sub_92123790) {
  ae::gpu::draw::FinishPending();
  static std::atomic<bool> s_depth_once{false};
  static std::atomic<bool> s_level_once{false};
  const auto* device = Guest<D3DDevice>(ctx.r3.u32);
  const uint32_t flags = ctx.r4.u32;
  const uint32_t source_index = flags & kResolveSourceMask;
  ae::gpu::ResolveRecord record;
  record.source = source_index == kResolveDepthStencil
                      ? tracker::Surface(device->depth_stencil, ResourceKind::kDepthTarget)
                      : tracker::Surface(device->render_targets[source_index & 3], ResourceKind::kColorTarget);
  record.dest = tracker::Texture(ctx.r6.u32);
  const uint32_t level = ctx.r8.u32;
  const uint32_t slice = ctx.r9.u32;
  if (level || slice) Once(s_level_once, "resolve into a mip level or slice other than 0 is not handled yet and is skipped", level | (slice << 16));
  if (source_index == kResolveDepthStencil && record.dest.format != ae::gpu::HostFormat::kR32Float) {
    Once(s_depth_once, "depth resolve into a non-depth texture is not handled yet; its destination shows magenta", flags);
  }
  const bool usable = record.source.id && record.dest.id && !level && !slice;
  if (usable) {
    const Rect full{0, 0, int32_t(record.source.width), int32_t(record.source.height)};
    record.source_rect = ClampRect(ctx.r5.u32 ? ReadRect(ctx.r5.u32) : full, record.source.width, record.source.height);
    if (ctx.r7.u32) {
      const auto* point = Guest<rex::be<int32_t>>(ctx.r7.u32);
      record.dest_x = point[0];
      record.dest_y = point[1];
    }
    const int bias = int32_t(flags) >> kResolveExponentBiasShift;
    record.scale = std::ldexp(1.0f, bias);
    record.clear_color = (flags & kResolveClearRenderTarget) && source_index != kResolveDepthStencil;
    if (record.clear_color && ctx.r10.u32) {
      const auto* v = Guest<rex::be<float>>(ctx.r10.u32);
      record.clear_color_value = {v[0], v[1], v[2], v[3]};
    }
    if (flags & kResolveClearDepthStencil) {
      record.clear_depth_target = tracker::Surface(device->depth_stencil, ResourceKind::kDepthTarget);
      record.clear_z = float(ctx.f1.f64);
      record.clear_stencil = *Guest<rex::be<uint32_t>>(ctx.r1.u32 + 0x5C) & 0xFF;
    }
    if (Tracing()) {
      REXGPU_INFO("[gpu] resolve flags {:#x} host {} ({},{})-({},{}) -> host {} at ({},{}) scale {} clear {} depth {} z {} "
                  "stencil {}",
                  flags, record.source.id, record.source_rect.left, record.source_rect.top, record.source_rect.right,
                  record.source_rect.bottom, record.dest.id, record.dest_x, record.dest_y, record.scale,
                  record.clear_color, record.clear_depth_target.id, record.clear_z, record.clear_stencil);
    }
    ae::gpu::Submit(std::move(record));
  }
  __imp__sub_92123790(ctx, base);
}

// D3DDevice_Swap 0x9211DA80 (pDevice, pFrontBuffer, pParameters). Nothing throttles the title without a GPU, so the
// cap paces frames here; a frame that falls one behind restarts the deadline instead of catching up.
REX_HOOK_RAW(sub_9211DA80) {
  ae::gpu::draw::FinishPending();
  using Clock = std::chrono::steady_clock;
  static const bool raised = (ae::thread::RaiseGuestThread(), true);
  (void)raised;
  // A time-armed trace picks the frame after the deadline, so the owner can navigate to the screen first.
  if (const int32_t at = REXCVAR_GET(gpu_trace_at_second); at > 0) {
    static const Clock::time_point start = Clock::now();
    static bool armed = false;
    if (!armed && Clock::now() - start >= std::chrono::seconds(at)) {
      armed = true;
      REXCVAR_SET(gpu_trace_frame, int32_t(ae::gpu::FrameIndex() + 1));
      REXGPU_INFO("[gpu] trace armed for frame {}", ae::gpu::FrameIndex() + 1);
    }
  }
  const int32_t cap = REXCVAR_GET(gpu_fps_cap);
  if (cap > 0) {
    const auto kFrame = std::chrono::nanoseconds(1'000'000'000 / cap);
    static Clock::time_point deadline = Clock::now();
    deadline += kFrame;
    const auto now = Clock::now();
    if (deadline > now) {
      ae::thread::SleepPrecise(deadline - now);
      ae::gpu::AddFrameTiming(0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - now).count()));
    } else if (now - deadline > kFrame) {
      deadline = now;
    }
  }
  if (REXCVAR_GET(gpu_fps_title)) {
    static Clock::time_point window_start = Clock::now();
    static uint32_t window_frames = 0;
    ++window_frames;
    const auto elapsed = Clock::now() - window_start;
    if (elapsed >= std::chrono::milliseconds(500)) {
      const double fps = double(window_frames) * 1e9 / double(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
      char text[64];
      std::snprintf(text, sizeof text, "%.1f fps", fps);
      ae::gpu::SetWindowTitle(std::string("Avatar Editor | ") + text);
      window_start = Clock::now();
      window_frames = 0;
    }
  }
  const uint64_t frame = ae::gpu::FrameIndex() + 1;
  ae::gpu::SwapRecord record;
  record.front = tracker::Texture(ctx.r4.u32);
  record.frame = frame;
  if (Tracing()) REXGPU_INFO("[gpu] swap front host {}", record.front.id);
  if (frame == 1 || frame % 600 == 0) {
    REXGPU_DEBUG("[gpu] swap {} front {:#010x} host {} {}x{}", frame, ctx.r4.u32, record.front.id, record.front.width,
                record.front.height);
  }
  ae::gpu::WaitForWarmup();
  ae::gpu::Submit(std::move(record));
  ae::gpu::EndFrame();
  MaybeDumpFrame(frame);
  __imp__sub_9211DA80(ctx, base);
}
