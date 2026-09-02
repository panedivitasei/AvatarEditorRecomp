// rexvideonative renderer, see renderer.h. GPL-3.0, see LICENSE.
//
// Runtime contract implemented here (docs/native_shaders.md):
//   root 0..4 : b0..b4 space0 (system / float / bool-loop / fetch / desc idx)
//   root 5    : t0 space0 shared-memory ByteAddressBuffer (upload ring)
//   root 6    : u0 space0 shared-memory UAV (dummy, SharedMemoryIsUAV never set)
//   set 0     : s0[] space0 bindless samplers
//   set 1..3  : t0[] space1/2/3 bindless textures (one set bound thrice,
//               UnleashedRecomp-style)

#include "renderer.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "shader_cache.h"
#include <filesystem>
#include <map>
#include <memory>
#include <deque>
#include <mutex>
#include <set>
#include <tuple>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>

#include <fmt/format.h>

#define XXH_INLINE_ALL
#include <xxhash.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>  // VirtualQuery guest-range probes
#endif

#include <thread>

#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/thread.h>
#include <rex/ui/window.h>

#include <plume_render_interface.h>
#include <plume_render_interface_builders.h>

#include "renderer_fps.h"
#include "renderer_depth_ps.h"
#include "renderer_fps_ps.h"
#include "render_queue.h"
#include "semantics/sampler_semantics.h"
#include "shader_cache.h"
#include "video_native_internal.h"

// The guest issues its one-shot UI bakes without waiting for icon loads;
// console latency hides the race, a fast native boot loses it. Give the
// first frames the same latency.
REXCVAR_DEFINE_INT32(native_video_boot_frame_delay_ms, 0, "GPU",
                     "Sleep this many ms per present for the first "
                     "boot_delay_frames presents, lets the guest's async "
                     "asset loads win the one-shot UI bake race the way "
                     "slow console boots do. 0 = off.");
REXCVAR_DEFINE_INT32(native_video_boot_delay_frames, 40, "GPU",
                     "How many initial presents boot_frame_delay_ms "
                     "applies to.");


// framerate_limit is owned by this module. resolution_scale and vsync are
// defined inside the rexgpu-xenos plugin DLL, which consumers never link;
// they are read by name through the cvar registry (QueryResolutionScale /
// QueryVsync in the anonymous namespace below), with plugin-less-boot
// fallback entries
// registered by video_native.cpp Init().
REXCVAR_DEFINE_INT32(framerate_limit, 0, "GPU",
                     "Host frame cap for the native present path, in frames "
                     "per second. 0 = uncapped (vsync still applies).");
REXCVAR_DEFINE_BOOL(native_video_tex_swizzle_full, true, "GPU",
                    "Compose fetch-constant destination swizzles through the format\'s "
                    "host base swizzle (the ring path\'s behavior). Off falls back to the "
                    "pure-reorder policy (native_video_tex_swizzle).");
REXCVAR_DEFINE_BOOL(native_video_tex_swizzle, true, "GPU",
                    "Honor pure channel-reorder destination swizzles. Swizzles containing "
                    "forced-0/1 selectors stay identity. false = full identity.");
REXCVAR_DEFINE_BOOL(native_video_tex_reupload, true, "GPU",
                    "Re-upload cached guest textures when their content changes (bounded "
                    "by the heal window and budget). Glyph atlases gain glyphs as "
                    "text streams in, so they go stale without it.");
REXCVAR_DEFINE_INT32(native_video_tex_max_heals, 0, "GPU",
                     "Max content-heal re-uploads per texture entry; glyph atlases gain "
                     "glyphs continuously, so this title raises it. 0 = unlimited.");
// Healing is budgeted per frame, not per lifetime: XUI rewrites pooled
// CPU surfaces constantly, so a lifetime cap exhausts at boot and freezes
// every later CPU write. The header_alive liveness check still gates
// every heal against recycled-memory aliasing.
REXCVAR_DEFINE_INT32(native_video_tex_heal_frame_budget, 16, "GPU",
                     "Max content-heal re-uploads per frame (bounded work). "
                     "0 = heals disabled.");
REXCVAR_DEFINE_BOOL(native_video_deferred_vb_copy, true, "GPU",
                    "Copy guest vertex/index ranges into the upload rings at present time "
                    "instead of draw time, matching when hardware actually fetches them. "
                    "UP draws stay call-time copies.");
REXCVAR_DEFINE_INT32(native_video_tex_evict_frames, 900, "GPU",
                     "Watermark eviction min-age: entries bound within "
                     "this many frames are never evicted, even at the "
                     "cache watermark.");
REXCVAR_DEFINE_INT32(native_video_tex_cache_max, 1500, "GPU",
                     "Texture-cache watermark: above this many entries "
                     "the coldest (LRU) retire fence-safe and their "
                     "descriptor slots recycle. Sized above a typical "
                     "working set so upload-once textures stay resident. "
                     "0 disables.");
REXCVAR_DEFINE_INT32(native_video_tex_cache_mb, 320, "GPU",
                     "Texture-cache VRAM watermark (approx MB): above it "
                     "the coldest entries retire like the count watermark. "
                     "Bounds long-session residency pressure on small-VRAM "
                     "GPUs. 0 disables.");
REXCVAR_DEFINE_INT32(native_video_rt_cache_mb, 384, "GPU",
                     "Render-target cache VRAM watermark (approx MB): "
                     "above it cold RT sets retire fence-safe (active and "
                     "last-drawn sets never touched). 0 disables.");
REXCVAR_DEFINE_BOOL(native_video_resolve_writeback, true, "GPU",
                    "Write color resolve results back into guest memory, delivered at the "
                    "guest\'s resolve fence waits, for engines that CPU-read resolved "
                    "bytes.");
// A depth texture no resolve ever produced is a fresh allocation the guest
// expects zeroed (reversed Z: nothing occludes). Recycled guest pages hold
// old bytes there, and the avatar tile bakes turn them into shadows.
REXCVAR_DEFINE_BOOL(native_video_unresolved_depth_zero, true, "GPU",
                    "Upload never-resolved depth-format textures as zeros "
                    "instead of their guest bytes.");
REXCVAR_DEFINE_INT32(native_video_writeback_max_dim, 256, "GPU",
                     "Skip resolve writeback for destinations wider or "
                     "taller than this (0 = no limit). Only the small "
                     "tile/preview class needs writeback; full-frame band "
                     "resolves would cost readback for pages nothing "
                     "CPU-reads.");
// Sentinel viewport rule: ViewportEnable(0) leaves the device viewport at
// the 65535 sentinel and hardware covers the current render target. In a
// bracket that is the tiling extent; outside one it is the bound
// surface's dims, which on band-widened surfaces is not the host RT's.
REXCVAR_DEFINE_BOOL(native_video_sentinel_vp_guest, true, "GPU",
                    "Sentinel (disabled) viewports map to the tiling extent "
                    "inside a tiling bracket, else to the bound guest "
                    "surface's own dimensions, instead of the host render "
                    "target's (which may be band-widened).");
REXCVAR_DEFINE_BOOL(native_video_small_resolve_rebase, false, "GPU",
                    "EDRAM-aliasing fix for small offscreen passes: when a small resolve "
                    "matches its dest dims but the picked source is larger and a same-dims "
                    "RT holds newer draws, resolve from that RT re-based to (0,0).");
REXCVAR_DEFINE_BOOL(native_video_rt_key_rbinfo, false, "GPU",
                    "Fold the guest device\'s RB surface-info shadow words into the "
                    "render-target key, for engines that retarget EDRAM through the raw "
                    "register shadows. Per-title opt-in.");
REXCVAR_DEFINE_BOOL(native_video_resolve_lazy, true, "GPU",
                    "Lazy source-side resolve layout transitions: back-to-back resolve "
                    "chains from one source pay one transition pair instead of one per "
                    "resolve. Dest transitions stay eager.");
REXCVAR_DEFINE_INT32(native_video_vs_trim, 2, "GPU",
                     "Trimmed-signature VS variants. 0 = off, 1 = on, 2 = Intel adapters "
                     "only (older AMD drivers wedge on the sparse interpolant layout).");
REXCVAR_DEFINE_BOOL(native_video_vb_gpu_local, true, "GPU",
                    "Keep the vertex ring in GPU-local (DEFAULT heap) "
                    "memory with staged copies instead of a CPU-mapped "
                    "upload heap, so vertex fetches do not cross the bus "
                    "per draw. "
                    "Takes effect only under the render queue (the inline "
                    "path keeps upload-heap semantics + deferred copies).");
REXCVAR_DEFINE_BOOL(native_video_vb_persist, true, "GPU",
                    "Cache static stream-VB uploads in a persistent region "
                    "of the vertex ring (upload once, reuse across frames). "
                    "Content changes miss by key construction; a guest VB "
                    "that changes twice is demoted to per-frame uploads for "
                    "good. Off = per-frame uploads.");
REXCVAR_DEFINE_BOOL(native_video_write_exp_bias, true, "GPU",
                    "Honor RB_COLOR_INFO\'s color exponent bias on the write side; the "
                    "sampler side re-expands via tfetch exp_adjust, so a hardcoded write "
                    "bias breaks the pair.");
REXCVAR_DEFINE_BOOL(native_video_per_axis_clamp, true, "GPU",
                    "Honor the fetch constant\'s clamp_x and clamp_y independently via "
                    "per-axis samplers. Off = clamp_x-only behavior.");
REXCVAR_DEFINE_BOOL(native_video_resolve_redirect, true, "GPU",
                    "Serve host-rendered resolve destinations to texture binds matching "
                    "their base address. Off: samples read the guest-memory upload.");
REXCVAR_DEFINE_INT32(native_video_redirect_min_dim, 0, "GPU",
                     "Minimum resolved-entry width for the base-address redirect. "
                     "0 = redirect everything.");
REXCVAR_DEFINE_BOOL(native_video_redirect_serve_subdim, true, "GPU",
                    "Serve a same-base host-rendered resolve entry to samples whose "
                    "declared dims fit inside the entry. This title composites its grid "
                    "tiles by re-sampling resolve products through smaller headers.");
REXCVAR_DEFINE_INT32(native_video_tex_heal_window, 1000000, "GPU",
                     "Frames after a texture-cache entry's creation during "
                     "which content re-checks (native_video_tex_reupload) "
                     "run; older entries are frozen. Long enough to cover "
                     "stream-in completion; 0 disables healing entirely.");
REXCVAR_DEFINE_BOOL(native_video_vp_grow, false, "GPU",
                    "Grow a tile-height host RT to the draw viewport's height "
                    "when the viewport exceeds it, for bracket-less "
                    "single-draw predicated tiling (a 2048x2048 viewport into "
                    "a 2080x544 EDRAM surface clips everything below the tile "
                    "band otherwise). Also suppresses the guest's per-tile "
                    "EDRAM-prep full clears between tile resolves on grown "
                    "RTs, which would wipe the source of tiles 2..N. "
                    "Per-title opt-in, off by default.");
REXCVAR_DEFINE_BOOL(native_video_fetch_shadow_streams, true, "GPU",
                    "Bind vertex streams from the guest device fetch-"
                    "constant shadow when no SetStreamSource state exists "
                    "(AE writes vfetch pairs directly).");
// No native band-walk dedup: the guest issues each scene draw once; the
// per-band repeats are the hardware's predicated replay on the ring side.
REXCVAR_DEFINE_BOOL(native_video_band_rt_widen, true, "GPU",
                    "Widen vertical EDRAM band surfaces (width = swapchain/"
                    "N, height = tile-aligned swapchain height) to the full "
                    "frame. AE's predicated-replay band composition.");
REXCVAR_DEFINE_INT32(native_video_colormask_offset, 12036, "GPU",
                     "Device offset of the color write mask the draw fold "
                     "reads. 12036 is the raw ColorWriteEnable request; "
                     "0 falls back to the shadow slot at dev+10580.");
REXCVAR_DEFINE_BOOL(native_video_stretch, false, "GPU",
                    "Stretch the game image to fill the window/screen. "
                    "Default off: the image keeps the game's native 16:9 "
                    "aspect on any monitor shape (16:10, 4:3, ultrawide) "
                    "with black bars filling the rest.");
REXCVAR_DEFINE_BOOL(native_video_b1_dedup, true, "GPU",
                    "Skip the per-draw 8KB float-constant re-upload when the "
                    "source bytes are unchanged within the frame (bit-exact "
                    "reuse of the previous ring allocation). Off = always "
                    "upload.");
REXCVAR_DEFINE_INT32(native_video_texture_retire_frames, 8, "GPU",
                     "Frames to keep retired (re-uploaded) texture entries "
                     "alive before destruction. Frame-count lifetime is only "
                     "safe while the GPU lags fewer frames than this; huge "
                     "value = leak-in-session.");
REXCVAR_DEFINE_BOOL(native_video_bind_scissor_reset, false, "GPU",
                    "XDK window-scissor semantics: a render-target change "
                    "resets the scissor to the full target unless the guest "
                    "set one since the bind. Without this, RTs larger than "
                    "the screen clip to the stale 1280x720 screen scissor. "
                    "Per-title opt-in, off by default: command-buffer replay "
                    "flows whose recorded SetScissorRect/bind interleavings "
                    "assume a persistent scissor need it off.");
REXCVAR_DEFINE_BOOL(native_video_gamma_ramp, true, "GPU",
                    "Apply the game's display gamma ramp (D3DDevice_"
                    "SetGammaRamp, 256-entry D3DGAMMARAMP at dev+15408) in "
                    "the native present blit, the ring backend applies it "
                    "via its DC_LUT apply-gamma pass; without it the picture "
                    "misses the game's contrast curve.");

namespace rex::videonative::renderer {

using namespace plume;

namespace {

// Registry-by-name reads of the GPU plugin's video cvars (see the cvar
// ownership note above). GetFlagInfo gates the default: Query returns a
// value-initialized T for unregistered names, which is wrong for vsync
// (default on).
int32_t QueryResolutionScale() {
  return rex::cvar::GetFlagInfo("resolution_scale")
             ? rex::cvar::Query<int32_t>("resolution_scale")
             : 1;
}
bool QueryVsync() {
  return rex::cvar::GetFlagInfo("vsync") ? rex::cvar::Query<bool>("vsync")
                                         : true;
}

// High-resolution sleep for the frame cap; plain sleeps quantize to the
// scheduler tick. Called only from the presenting thread (single shared
// timer).
void SleepPrecise(std::chrono::microseconds us) {
#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
  static HANDLE timer = [] {
    HANDLE t = CreateWaitableTimerExW(nullptr, nullptr,
                                      CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                      TIMER_ALL_ACCESS);
    if (!t) t = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    return t;
  }();
  if (timer) {
    LARGE_INTEGER due;
    due.QuadPart = -static_cast<int64_t>(us.count()) * 10;  // relative 100ns
    if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
      WaitForSingleObject(timer, INFINITE);
      return;
    }
  }
  std::this_thread::sleep_for(us);
#else
  std::this_thread::sleep_for(us);
#endif
}

// Internal resolution scale as a RATIONAL num/den (resolution_scale=2 ->
// 2/1).
// All renderer bookkeeping stays in guest units; the scale applies only at
// the D3D boundary: host RT/resolve-tex allocation, viewports/scissors/
// clear rects, and resolve copy rects. Latched at first use (mid-run
// changes would orphan every cached surface).
uint32_t g_hsNum = 1, g_hsDen = 1;
void LatchHostScale() {
  static bool latched = false;
  if (latched) return;
  latched = true;
  const int32_t v = QueryResolutionScale();
  if (v == 2) {
    g_hsNum = 2;
    REXGPU_INFO(
        "videonative: internal resolution 2x (720p content rendered at "
        "1440p-class)");
  } else if (v != 1 && v != 0) {
    REXGPU_WARN(
        "videonative: resolution_scale {} unsupported on the native "
        "path (1 or 2) - using 1",
        v);
  }
}
bool Scaled() {
  LatchHostScale();
  return g_hsNum != g_hsDen;
}
// Allocation dims: ceil, a surface is never smaller than any scaled rect
// laid inside it (floor-mapped corners below stay in bounds).
uint32_t HostDim(uint32_t x) {
  LatchHostScale();
  return std::max(1u, (x * g_hsNum + g_hsDen - 1) / g_hsDen);
}
// Rect corners / copy offsets: floor, a shared guest edge maps to one
// host value, so adjacent tile bands stay gapless and overlap-free.
// Callers pass non-negative coords (truncation == floor).
int32_t HostCoord(int32_t x) {
  LatchHostScale();
  return int32_t((int64_t(x) * g_hsNum) / g_hsDen);
}
float HostScaleF() {
  LatchHostScale();
  return float(g_hsNum) / float(g_hsDen);
}
// One-byte identity for cached-entry compatibility checks (2/1=0x21 etc.).
uint8_t HostScaleCode() {
  LatchHostScale();
  return uint8_t((g_hsNum << 4) | g_hsDen);
}

constexpr uint32_t kNumFrames = 2;
constexpr uint32_t kSwapChainBufferCount = 3;
constexpr RenderFormat kColorFormat = RenderFormat::B8G8R8A8_UNORM;
// Stencil-capable depth: the editor's mirror pass gates the reflection
// on stencil, which plain D32 cannot express. D32S8's plane 0 keeps R32
// copy/SRV semantics, so the resolve paths are unchanged.
constexpr RenderFormat kDepthFormat = RenderFormat::D32_FLOAT_S8_UINT;
// Pack contract: every pack PS bakes `min(idx, N-1)` on its descriptor
// indices, so changing this requires regenerating the packs. Kept in
// lockstep with xe_descriptor_index() in rexglue_shader_common.h.
// 16K sized for the item-grid browsing working set.
constexpr uint32_t kTextureDescriptorCount = 16384;
// Reserved dimension-matched blank descriptors (see upload_bindings): a
// descriptor whose ViewDimension mismatches the shader's declared resource
// type (TextureCube/Texture3D vs Texture2DArray) is D3D12 UB, AMD reads
// garbage harmlessly, Intel pagefaults at wild VAs (DEVICE_HUNG).
// Slots (N = kTextureDescriptorCount): 0 + N-1 = 2D blanks (sentinel
// + shader clamp target), N-2 = cube blank, N-3 = 3D blank. Allocation
// runs 1..N-4.
constexpr uint32_t kBlank3dDescriptor = kTextureDescriptorCount - 3;
constexpr uint32_t kBlankCubeDescriptor = kTextureDescriptorCount - 2;
constexpr uint32_t kFirstReservedDescriptor = kTextureDescriptorCount - 3;
constexpr uint32_t kSamplerDescriptorCount = 16;

// Sampler semantics (semantics/sampler_semantics.*): ring-contract sampler
// decode. Cvars live here so every REXCVAR_GET stays in this TU.
REXCVAR_DEFINE_BOOL(native_video_sem_sampler, true, "GPU",
                    "Ring-contract sampler decode: "
                    "filters, aniso, per-axis clamp incl. mirror-once/"
                    "border, cube/1D dimension rules. OFF = legacy "
                    "clamp-bit pick.");

// Engine-authoritative RTT targets: the title's StartTextureRendering
// hook names the texture about to be rendered into, and host RT identity
// follows it instead of EDRAM inference, which cannot separate
// differently-sized canvases sharing a base.
REXCVAR_DEFINE_BOOL(native_video_sem_rtt, true, "GPU",
                    "Engine-authoritative RTT pass targets: host RT "
                    "identity+dims from the title's TEXTUREBASEDX9 note "
                    "(canvas 1024 rows, ping-pong identity, reflection "
                    "separation). OFF = EDRAM inference only.");
REXCVAR_DEFINE_BOOL(native_video_sem_rtt_log, false, "GPU",
                    "Log every RTT note application and mismatch.");

// Depth resolves render through a fullscreen triangle instead of the
// whole-plane staging copy D3D12 would otherwise require; partial copies
// out of a depth resource are illegal, rendering is not.
REXCVAR_DEFINE_BOOL(native_video_depth_resolve_shader, false, "GPU",
                    "Resolve depth by rendering a depth SRV into the "
                    "destination instead of the whole-plane staging copy. "
                    "Off = the staging path (with its per-frame plane memo).");
REXCVAR_DEFINE_BOOL(native_video_untile_packed, true, "GPU",
                    "Decode sub-tile tiled textures from their packed-"
                    "mip-tail offset (360-correct). Off = tile-origin decode, "
                    "which another title's look is calibrated against.");

// The ring backend delivers shadow depth-resolves to guest RAM; the
// CPU-side lighting bake reads them.
REXCVAR_DEFINE_BOOL(native_video_depth_writeback, true, "GPU",
                    "Write depth resolve results back into guest memory "
                    "(D32 host -> D24S8 guest). The lighting bake "
                    "reads shadow-map bytes from RAM.");

// The avatar renders a depth-only twin of itself into a shadow map the lit
// shaders then sample; skipping those draws leaves the map fully lit.
REXCVAR_DEFINE_BOOL(model_shadows, true, "GPU",
                    "Self-shadowing on avatar models (hats, clothing, and "
                    "props shadow the model). Off skips the shadow map "
                    "draws.");

// Content is snapshotted at the resolve, so deferred delivery writes the
// same bytes a synchronous one would, without blocking the guest thread.
REXCVAR_DEFINE_BOOL(native_video_early_submit, false, "GPU",
                    "Submit the command list eagerly at writeback-record "
                    "points so sync flushes wait only the in-flight tail.");
REXCVAR_DEFINE_BOOL(native_video_writeback_sync, false, "GPU",
                    "Deliver resolve write-back synchronously at each "
                    "resolve (legacy, stalls the guest thread). OFF = "
                    "deliver at frame end.");

// The ring backend keys each pass by the full (surface, C0-C3, depth)
// register tuple. Slot 0 alone cannot separate distinct guest passes, so
// slots 1-3 and depth fold into the key.
REXCVAR_DEFINE_BOOL(native_video_pass_separation, true, "GPU",
                    "Key host RTs by the full color0-3+depth surface "
                    "tuple (ring's pass key) instead of slot 0 only. "
                    "Off = fused keying on slot 0.");

// Heavily streamed levels can demand over 30 MB of stream-VB uploads per
// frame, so the ring is sized well clear of that. The allocator fails
// instead of wrapping: wrapping would hand out regions still referenced by
// this frame's draws, corrupting geometry silently.
constexpr uint32_t kUploadBufferSize = 128 * 1024 * 1024;

// Root descriptor ordinals (addRootDescriptor order in Init).
constexpr uint32_t kRootB0System = 0;
constexpr uint32_t kRootB1FloatsVs = 1;
constexpr uint32_t kRootB1FloatsPs = 2;
constexpr uint32_t kRootB2BoolLoop = 3;
constexpr uint32_t kRootB3Fetch = 4;
constexpr uint32_t kRootB4IndicesVs = 5;
constexpr uint32_t kRootB4IndicesPs = 6;
constexpr uint32_t kRootSharedMemSrv = 7;
constexpr uint32_t kRootDummyUav = 8;

// ---------------------------------------------------------------------------
// Device / frame objects
// ---------------------------------------------------------------------------

RenderDevice* g_device;
RenderCommandQueue* g_queue;

std::unique_ptr<RenderSwapChain> g_swapChain;
bool g_swapChainValid = false;
uint32_t g_backBufferIndex = 0;
uint32_t g_frame = 0;
bool g_frameOpen = false;
std::unique_ptr<RenderCommandList> g_commandLists[kNumFrames];
std::unique_ptr<RenderCommandFence> g_commandFences[kNumFrames];
bool g_commandListPending[kNumFrames] = {};
std::unique_ptr<RenderCommandSemaphore> g_acquireSemaphores[kNumFrames];
std::unique_ptr<RenderCommandSemaphore> g_renderSemaphores[kNumFrames];
// Segmented submission: a segment is submitted (no wait) whenever a
// writeback record is created, so the GPU runs in parallel with further
// recording and the sync flush waits only the in-flight tail.
struct SegInFlight {
  std::unique_ptr<RenderCommandList> list;
  std::unique_ptr<RenderCommandFence> fence;
};
std::vector<SegInFlight> g_segInFlight[kNumFrames];
std::vector<std::unique_ptr<RenderCommandList>> g_segSpareLists;
std::vector<std::unique_ptr<RenderCommandFence>> g_segSpareFences;
bool g_segAcquirePending[kNumFrames] = {};  // first submit waits acquire
std::unique_ptr<RenderTexture> g_depthTexture;
std::vector<std::unique_ptr<RenderFramebuffer>> g_framebuffers;

std::unique_ptr<RenderPipelineLayout> g_pipelineLayout;
std::unique_ptr<RenderDescriptorSet> g_textureDescriptorSet;
std::unique_ptr<RenderDescriptorSet> g_samplerDescriptorSet;
std::unique_ptr<RenderBuffer> g_dummyUav;
std::unique_ptr<RenderTexture> g_blankTexture;
std::unique_ptr<RenderTextureView> g_blankTextureView;
std::unique_ptr<RenderTexture> g_blankTextureCube;
std::unique_ptr<RenderTextureView> g_blankTextureCubeView;
std::unique_ptr<RenderTexture> g_blankTexture3d;
std::unique_ptr<RenderTextureView> g_blankTexture3dView;
std::vector<std::unique_ptr<RenderSampler>> g_samplers;

uint64_t g_frameIndex = 0;
uint64_t g_drawsThisFrame = 0;
uint64_t g_drawsSkipped = 0;
uint64_t g_frameOpenFails = 0;
// Set at device init from the adapter description (vendor-gated features:
// the vs_trim intel-only mode).
bool g_adapterIsIntel = false;
// Draw-skip counters: why draws bail out of SetupDraw, and which primitive
// types the unsupported-topology skips carry.
uint64_t g_skipNoVs = 0;
uint64_t g_skipNoPs = 0;
uint64_t g_skipTopology = 0;
uint64_t g_skipPipeline = 0;
uint64_t g_skipPrimTypes[32] = {};
// RT-flow counters.
uint64_t g_drawsToRt = 0;
uint64_t g_drawsToSwapchain = 0;
uint64_t g_resolveCopies = 0;
// Layout transitions emitted by the resolve path. Integrated GPUs pay a
// full-surface decompression per RT transition, so lazy mode's win shows
// here first.
uint64_t g_resolveBarriersEmitted = 0;

// ---------------------------------------------------------------------------
// Upload ring (per frame), UnleashedRecomp UploadAllocator, simplified
// ---------------------------------------------------------------------------

struct UploadAllocation {
  const RenderBuffer* buffer;
  uint64_t offset;
  uint8_t* memory;
};

struct UploadAllocator {
  struct Page {
    std::unique_ptr<RenderBuffer> buffer;
    uint8_t* memory = nullptr;
  };
  std::vector<Page> pages;
  uint32_t index = 0;
  uint32_t offset = 0;

  UploadAllocation Allocate(uint32_t size, uint32_t alignment) {
    assert(size <= kUploadBufferSize);
    offset = (offset + alignment - 1) & ~(alignment - 1);
    if (offset + size > kUploadBufferSize) {
      ++index;
      offset = 0;
    }
    if (pages.size() <= index) pages.resize(index + 1);
    Page& page = pages[index];
    if (!page.buffer) {
      page.buffer = g_device->createBuffer(RenderBufferDesc::UploadBuffer(
          kUploadBufferSize, RenderBufferFlag::CONSTANT |
                                 RenderBufferFlag::VERTEX |
                                 RenderBufferFlag::INDEX |
                                 RenderBufferFlag::FORMATTED));
      page.memory = reinterpret_cast<uint8_t*>(page.buffer->map());
      page.buffer->setName(fmt::format("upload page {}", index));
      // Record each buffer's GPU VA so a DRED page-fault address can be
      // attributed to a buffer and offset.
    }
    UploadAllocation result{page.buffer.get(), offset, page.memory + offset};
    offset += size;
    return result;
  }

  void Reset() {
    index = 0;
    offset = 0;
  }
};

UploadAllocator g_upload[kNumFrames];

// Vertex ring page for the current frame's shared-memory SRV; all of a
// draw's vertex data must live in one page. A persistent region grows up
// from offset 0 for static stream VBs (same offset in every frame's
// buffer); per-frame allocations start above it and reset each frame.
// With native_video_vb_gpu_local the buffer is default-heap and writes go
// through the staging ring; otherwise it is the mapped upload heap.
constexpr uint32_t kVbStagingSize = 32 * 1024 * 1024;
bool VbGpuLocal();  // latched below (needs rq::Running at first frame)
// Bottom of every ring buffer is reserved for the persistent VB region
// from construction: frame 0 runs before the first frame-open reset, so a
// zero floor would let boot-frame allocations overwrite persistent uploads.
constexpr uint32_t kVbPersistCap = 96 * 1024 * 1024;

struct VertexRing {
  std::unique_ptr<RenderBuffer> buffer;
  std::unique_ptr<RenderBuffer> staging;
  uint8_t* memory = nullptr;          // mapped upload memory (upload mode)
  uint8_t* staging_memory = nullptr;  // mapped staging memory (gpu mode)
  uint32_t offset = kVbPersistCap;
  uint32_t staging_offset = 0;
  bool gpu_local = false;
  bool in_copy_state = false;  // barrier bookkeeping for the default buffer

  void EnsureBuffer() {
    if (buffer) return;
    gpu_local = VbGpuLocal();
    if (gpu_local) {
      buffer = g_device->createBuffer(RenderBufferDesc::DefaultBuffer(
          kUploadBufferSize, RenderBufferFlag::FORMATTED));
      buffer->setName("vertex ring (gpu)");
      staging = g_device->createBuffer(
          RenderBufferDesc::UploadBuffer(kVbStagingSize));
      staging_memory = reinterpret_cast<uint8_t*>(staging->map());
      staging->setName("vertex staging");
    } else {
      buffer = g_device->createBuffer(RenderBufferDesc::UploadBuffer(
          kUploadBufferSize, RenderBufferFlag::FORMATTED));
      memory = reinterpret_cast<uint8_t*>(buffer->map());
      buffer->setName("vertex ring");
    }
  }

  // Reserve `size` bytes of ring address space. In gpu mode the returned
  // memory pointer is STAGING memory; the caller must finish with
  // CommitCopy() after writing it.
  UploadAllocation Allocate(uint32_t size) {
    // Never hand out an allocation that cannot fit, the caller memcpys
    // `size` bytes and would run off the mapped buffer.
    if (size > kUploadBufferSize) {
      return UploadAllocation{nullptr, 0, nullptr};
    }
    EnsureBuffer();
    offset = (offset + 15) & ~15u;
    if (offset + size > kUploadBufferSize) {
      // Never wrap: earlier allocations are still referenced by this
      // frame's draws, and overlapping them corrupts geometry silently.
      // Fail loudly; the caller drops the draw.
      static uint64_t overflow_logs = 0;
      if ((overflow_logs++ & 63) == 0) {
        REXGPU_ERROR(
            "videonative: vertex ring EXHAUSTED ({} + {} > {}), draw "
            "dropped ({} so far this session)",
            offset, size, kUploadBufferSize, overflow_logs);
      }
      return UploadAllocation{nullptr, 0, nullptr};
    }
    if (!gpu_local) {
      UploadAllocation result{buffer.get(), offset, memory + offset};
      offset += size;
      return result;
    }
    staging_offset = (staging_offset + 15) & ~15u;
    if (staging_offset + size > kVbStagingSize) {
      static uint64_t staging_logs = 0;
      if ((staging_logs++ & 63) == 0) {
        REXGPU_ERROR(
            "videonative: vertex STAGING exhausted ({} + {} > {}), draw "
            "dropped ({} so far this session)",
            staging_offset, size, kVbStagingSize, staging_logs);
      }
      return UploadAllocation{nullptr, 0, nullptr};
    }
    UploadAllocation result{buffer.get(), offset,
                            staging_memory + staging_offset};
    offset += size;
    staging_offset += size;
    return result;
  }

  // gpu mode: record the staging->buffer copy for an allocation returned by
  // Allocate (alloc.offset = dest, the staging bytes just written). Must be
  // called after the write and before the draw that fetches from it.
  void CommitCopy(RenderCommandList* cl, uint32_t dest_offset,
                  uint32_t src_staging_end, uint32_t size) {
    if (!gpu_local || !size) return;
    ToCopyState(cl);
    cl->copyBufferRegion(buffer->at(dest_offset),
                         staging->at(src_staging_end - size), size);
  }

  void ToCopyState(RenderCommandList* cl) {
    if (in_copy_state) return;
    cl->barriers(RenderBarrierStage::COPY,
                 RenderBufferBarrier(buffer.get(), RenderBufferAccess::WRITE));
    in_copy_state = true;
  }
  void ToReadState(RenderCommandList* cl) {
    if (!in_copy_state) return;
    cl->barriers(RenderBarrierStage::GRAPHICS,
                 RenderBufferBarrier(buffer.get(), RenderBufferAccess::READ));
    in_copy_state = false;
  }

  void Reset(uint32_t floor) {
    offset = floor;
    staging_offset = 0;
    // Buffers decay across list boundaries; start each frame assuming the
    // read state so the first copy re-emits its write barrier.
    in_copy_state = false;
  }
};

VertexRing g_vertexRing[kNumFrames];

// Latched once at first buffer creation: both parities must agree (the
// persistent dual-parity copies assume one mode), and rq::Running() is
// stable by the first draw (the worker starts before any frame renders).
bool VbGpuLocal() {
  static const bool latched =
      REXCVAR_GET(native_video_vb_gpu_local) && rq::Running();
  return latched;
}

// Persistent-region state (worker thread only). Fixed partition: the
// persistent region owns [0, kVbPersistCap) and per-frame allocations
// start above it, so the regions can never collide. Entries are keyed by
// content hash; a key that changes content twice is demoted to the
// per-frame ring for good.
// kVbPersistCap (96MB; dynamics get 32MB) is declared above VertexRing so
// construction honors the partition from the very first frame.
uint32_t g_vbPersistWatermark = 0;
std::unordered_map<uint64_t, uint32_t> g_vbPersistCache;  // cache_key -> off
struct PersistGuestInfo {
  uint64_t last_key = 0;  // content key of the resident upload
  uint32_t changes = 0;   // content flips seen (2 = demote to per-frame)
  uint32_t size = 0;
};
std::unordered_map<uint64_t, PersistGuestInfo> g_vbPersistByGuest;
uint64_t g_vbPersistOrphans = 0;   // bytes abandoned by content changes
uint64_t g_vbPersistDemoted = 0;   // guest keys routed to per-frame forever
bool g_vbPersistFlushWanted = false;  // region full: flush at next idle
uint64_t g_vbPersistHits = 0;      // stat window: binds served with no upload
uint64_t g_vbPersistUploads = 0;   // stat window: first-time uploads

// Allocate `size` bytes at the same offset in every frame's buffer.
// Returns 0xFFFFFFFF when the persistent region is out of room (caller
// falls back to the per-frame ring, today's behavior, loudly).
uint32_t PersistAlloc(uint32_t size) {
  uint32_t off = (g_vbPersistWatermark + 15) & ~15u;
  if (off + size > kVbPersistCap) {
    g_vbPersistFlushWanted = true;  // reclaim at the next idle frame open
    static uint64_t full_logs = 4;
    if (full_logs) {
      full_logs--;
      REXGPU_WARN(
          "videonative: persistent VB region full ({} + {} > {}), "
          "flush queued (orphaned {} KB)",
          off, size, kVbPersistCap, g_vbPersistOrphans >> 10);
    }
    return 0xFFFFFFFFu;
  }
  for (uint32_t i = 0; i < kNumFrames; i++) {
    g_vertexRing[i].EnsureBuffer();
  }
  g_vbPersistWatermark = off + size;
  return off;
}

// Per-frame dedupe of vertex-data uploads: the menu re-binds the same guest
// VBs across dozens of draws, and uploading each bind separately would wrap
// the ring mid-frame and starve large buffers.
// Keyed (guest data ptr, size) -> ring byte offset. Cleared at frame open.
std::unordered_map<uint64_t, uint32_t> g_vbUploadCache;
// Per-frame map of large stream-VB binds (guest base, size, ring offset).
uint32_t g_lastBindRingOffset = 0xFFFFFFFF;
// Per-draw mask of streams whose vfetch pair translated to a ring offset
// this draw.
uint32_t g_drawVfetchWrittenMask = 0;

// ---------------------------------------------------------------------------
// Captured guest state
// ---------------------------------------------------------------------------

struct StreamState {
  uint32_t vb_object = 0;
  uint32_t offset_bytes = 0;
  uint32_t stride_bytes = 0;
  // Bind-time snapshot of the VB object's +24/+28 fields (stack-temporary
  // headers, same class as ib_snap_*, see SetIndices).
  bool snap_valid = false;
  uint32_t snap_data = 0;
  uint32_t snap_size = 0;
};

struct GuestState {
  uint32_t viewport_x = 0, viewport_y = 0, viewport_w = 1280, viewport_h = 720;
  float viewport_min_z = 0.0f, viewport_max_z = 1.0f;
  int32_t scissor[4] = {0, 0, 1280, 720};
  uint32_t textures[32] = {};  // guest texture header addresses by fetch slot
  // Bind-time snapshot of each bound header's 6 fetch dwords (XDK copy
  // semantics, see SetTexture); draws consume these, not a draw-time re-read.
  uint32_t texture_dw[32][6] = {};
  StreamState streams[16];
  uint32_t index_buffer_object = 0;
  // Bind-time snapshot of the bound IB object's fields. The avatar section
  // renderers build the IB header on the stack via XGSetIndexBufferHeader;
  // it is dead by deferred-flush time, and a draw-time object read fetches
  // stack garbage as indices. Same XDK copy semantics as texture_dw.
  bool ib_snap_valid = false;
  uint32_t ib_snap_data = 0;   // +24 raw data pointer
  uint32_t ib_snap_size = 0;   // +28 byte size
  bool ib_snap_idx32 = false;  // dword0 bit 31
  // Bound guest surface objects (0 = swapchain backbuffer / default depth).
  // color_surface[i] = MRT slot i. Slot 0 drives the RT set's dimensions,
  // edram base and depth-only detection; slots 1-3 contribute their format
  // (the lighting pass binds a 10-bit diffuse on RT0 and an 8888 specular
  // on RT1, so a single shared format would put the specular accumulation
  // in a 10-bit texture).
  uint32_t color_surface[4] = {};
  uint32_t depth_surface = 0;
  // Guest float constants, already byte-swapped to host endian.
  float vs_floats[256 * 4] = {};
  float ps_floats[256 * 4] = {};
};

GuestState g_state;

// ---------------------------------------------------------------------------
// System constants (b0), layout mirrors DxbcShaderTranslator::SystemConstants
// for the fields the native pack reads (16 float4 registers = 256 bytes).
// ---------------------------------------------------------------------------

// kSysFlag_* bits consumed by the pack (rexglue_shader_common.h).
constexpr uint32_t kSysFlagWNotReciprocal = 1u << 3;
constexpr uint32_t kSysFlagAlphaPassIfLess = 1u << 7;
constexpr uint32_t kSysFlagAlphaPassIfEqual = 1u << 8;
constexpr uint32_t kSysFlagAlphaPassIfGreater = 1u << 9;

struct SystemConstants {
  uint32_t flags;             // c0.x
  uint32_t pad0[2];           // c0.yz
  uint32_t line_loop_index;   // c0.w
  uint32_t vertex_index_endian;  // c1.x
  int32_t vertex_index_offset;   // c1.y
  uint32_t vertex_index_min;     // c1.z
  uint32_t vertex_index_max;     // c1.w
  uint32_t pad1[4 * 6];          // c2..c7
  float ndc_scale[3];            // c8.xyz
  uint32_t pad2;                 // c8.w
  float ndc_offset[3];           // c9.xyz
  uint32_t pad3;                 // c9.w
  // c10: POINTLIST sprite sizing for the pack's point_expand GS (all pixel
  // DIAMETERS, mirroring the ring backend's SystemConstants):
  //   xy = PA_SU_POINT_SIZE constant width/height (used when the VS does not
  //        export a point size), zw = PA_SU_POINT_MINMAX clamp for the
  //        vertex-exported size.
  float point_constant_diameter[2];   // c10.xy
  float point_vertex_diameter_min;    // c10.z
  float point_vertex_diameter_max;    // c10.w
  uint32_t pad4[4 * 2];          // c11..c12
  uint32_t pad5[3];              // c13.xyz
  float alpha_test_reference;    // c13.w
  uint32_t pad6[4];              // c14
  float color_exp_bias[4];       // c15
};
static_assert(sizeof(SystemConstants) == 256);

// ---------------------------------------------------------------------------
// Texture cache
// ---------------------------------------------------------------------------

struct CachedTexture {
  std::unique_ptr<RenderTexture> texture;
  std::unique_ptr<RenderTextureView> view;
  // 0 = no slot (sentinel, see the !descriptor_index draw guard). Slots 0
  // and the last slot are reserved blanks (see kBlank* above): the sentinel
  // and the shader-side garbage clamp both land on a blank descriptor,
  // never another texture's.
  uint32_t descriptor_index = 0;
  uint32_t width = 0, height = 0;
  RenderFormat host_format = RenderFormat::UNKNOWN;
  // View dimension created (guest DataDimension): 1 = 2D-array,
  // 3 = cube. Bindings of a different declared dimension must not be served
  // this descriptor (see kBlankCubeDescriptor).
  uint8_t view_dim = 1;
  // Host allocation scale of a resolve target (HostScaleCode() at
  // creation, num<<4|den); guest-uploaded file textures stay 1:1.
  uint8_t host_scale = 0x11;
  bool valid = false;
  // Content comes from host resolves, not guest memory uploads.
  bool host_rendered = false;
  // Framebuffer wrapping this texture as a single R32_FLOAT color target,
  // only depth-resolve destinations under the shader resolve path have one.
  // rt_capable records that the texture was CREATED with RENDER_TARGET (the
  // flag cannot be queried back from plume, and creating the framebuffer
  // without it is a device-removal-class error).
  std::unique_ptr<RenderFramebuffer> resolve_fb;
  bool rt_capable = false;
  uint64_t last_resolve_frame = 0;  // 0 = never resolved into
  // Count of resolves into this entry.
  uint32_t resolve_serial = 0;
  // Stamped on every cache hit/creation (GetOrCreateTexture), feeds the
  // idle-eviction sweep that bounds the cache (and the descriptor counter).
  uint64_t last_bind_frame = 0;
  // Approximate host allocation size (watermark accounting; format-aware
  // bytes/px estimate at creation). Accumulated into g_texCacheBytes on
  // insert, released by RetireEntry.
  uint32_t approx_bytes = 0;
  // Creation-time header identity for eviction-safety: a guest-uploaded
  // entry may only be evicted when the live header no longer matches
  // (the key is the header hash, a dead header means this entry is
  // unreachable for good). Evicting a reachable upload risks re-uploading
  // from guest memory the game recycled after the original upload:
  // hardware sampled the original bytes, a re-read gets the recycler's.
  uint32_t header_addr = 0;
  uint32_t header_dw[6] = {};
  RenderTextureLayout layout = RenderTextureLayout::UNKNOWN;
  // Sparse content hash of the guest texel data at bind time, to catch
  // in-place rewrites (animated textures, avatar face composites).
  uint64_t content_hash = 0;
  uint64_t last_content_check = 0;
  uint32_t content_changes = 0;
  // Upload skipped because the guest source range was not committed at
  // create time (AE binds textures whose backing commits later, e.g.
  // gamerpics). Re-checked at bind; a now-readable source re-uploads.
  bool upload_skipped = false;
  // Frame this entry was created (uploaded); bounds the healing window.
  // Content re-checks stop once the entry outlives
  // native_video_tex_heal_window frames, so stream-in poisoning heals
  // shortly after first bind but an aged entry can never be healed into
  // recycled memory.
  uint64_t created_frame = 0;
};

void* g_inspectHwnd = nullptr;
std::unordered_map<uint64_t, CachedTexture> g_textureCache;
uint32_t g_nextTextureDescriptor = 1;  // 0 = blank
// Fallback for the frontbuffer flip (see SwapFrontbuffer): last resolve
// destination by guest texture size. Element pointers are stable across
// unordered_map rehash; entries are purged when their texture is erased.
std::unordered_map<uint64_t, CachedTexture*> g_lastResolvedBySize;
// Re-uploaded (content-changed) texture entries retired here and destroyed a
// few frames later, in-flight command lists may still reference the old host
// texture. Total is capped as a descriptor-exhaustion guard (each re-upload
// allocates a fresh descriptor slot).
std::deque<std::pair<uint64_t, CachedTexture>> g_retiredTextures;

// Every CachedTexture destruction must pass through here: an immediate
// destruction can free a texture in-flight draws still sample (a GPU
// use-after-free that hangs Intel iGPUs). The drain scrubs the descriptor
// slot at the fence-safe point before destroying.
uint64_t g_texCacheBytes = 0;
// Host textures currently parked in g_retiredTextures. A retire of a texture
// already parked means a second owner was minted (a stale map node moved
// after its erase), which would otherwise be a double free. Such a duplicate
// is logged and leaked rather than destroyed twice.
std::unordered_set<RenderTexture*> g_retiredPtrs;
static void DetachRawRefs(CachedTexture& e) {
  for (auto sit = g_lastResolvedBySize.begin(); sit != g_lastResolvedBySize.end();) {
    sit = (sit->second == &e) ? g_lastResolvedBySize.erase(sit) : std::next(sit);
  }
}
void RetireEntry(CachedTexture& e, const char* who = "?") {
  DetachRawRefs(e);
  if (!e.texture) return;
  RenderTexture* raw = e.texture.get();
  if (!g_retiredPtrs.insert(raw).second) {
    static uint64_t dbl_logs = 32;
    if (dbl_logs) {
      dbl_logs--;
      REXGPU_WARN("videonative: [texdouble] retire '{}' of an already-retired texture "
                  "base={:#x} {}x{} hdr={:#x} host_rendered={} (duplicate owner leaked)",
                  who, e.header_dw[1] & 0xFFFFF000u, e.width, e.height, e.header_addr,
                  e.host_rendered ? 1 : 0);
    }
    (void)e.texture.release();
    (void)e.view.release();
    e.resolve_fb.reset();
    return;
  }
  g_texCacheBytes -= std::min<uint64_t>(g_texCacheBytes, e.approx_bytes);
  g_retiredTextures.emplace_back(g_frameIndex, std::move(e));
}

// Explicit guest-range invalidation: XamAvatarGetAssets rewrites
// recycled GPU buffers whose cache keys do not change, so the kernel
// reports each rewritten range through the runtime bridge; ranges queue
// from any thread and drain on the render thread before texture lookup.
std::mutex g_invalidateMx;
std::vector<std::pair<uint32_t, uint32_t>> g_pendingInvalidates;
// The thread that owns the cache containers (render-queue worker with
// the offload, guest main thread without). Other threads must never touch
// g_textureCache / g_retiredTextures directly; off-thread callers queue
// ranges instead.
std::atomic<uint64_t> g_rendererThreadHash{0};
static uint64_t ThisThreadHash() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }
bool OnRendererThread() {
  const uint64_t owner = g_rendererThreadHash.load(std::memory_order_acquire);
  return owner == 0 || owner == ThisThreadHash();
}
std::atomic<bool> g_hasPendingInvalidates{false};

// Frozen ranges: a guest buffer mid-rewrite (XamAvatarGetAssets bracket).
// While frozen, heals, bake-source refreshes and skipped-upload rechecks
// must not re-read the range; a mid-rewrite re-upload tears the texture.
// Cleared by the paired invalidate (rewrite complete), with a coarse frame
// expiry as a leak guard if the pair never arrives.
std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> g_frozenRanges;
std::atomic<bool> g_hasFrozenRanges{false};

bool IsGuestRangeFrozen(uint32_t base, uint32_t span) {
  if (!g_hasFrozenRanges.load(std::memory_order_acquire)) return false;
  std::lock_guard<std::mutex> lk(g_invalidateMx);
  const uint32_t b = base & 0x1FFFFFFFu;
  bool frozen = false;
  for (auto it = g_frozenRanges.begin(); it != g_frozenRanges.end();) {
    if (g_frameIndex - std::get<2>(*it) > 600) {  // ~10s leak guard
      it = g_frozenRanges.erase(it);
      continue;
    }
    if (b < std::get<0>(*it) + std::get<1>(*it) &&
        std::get<0>(*it) < b + span) {
      frozen = true;
    }
    ++it;
  }
  if (g_frozenRanges.empty())
    g_hasFrozenRanges.store(false, std::memory_order_release);
  return frozen;
}

// CPU-owned resolve dests: once the guest LockRects a base its pages are
// CPU-authoritative and the redirect must not serve the raw host resolve
// product over them. Bases queue from the guest thread; registration is
// suppressed while the stamp is fresh.
extern std::unordered_map<uint32_t, uint64_t> g_resolvedBaseToKey;
std::vector<uint32_t> g_pendingCpuLocks;
std::atomic<bool> g_hasPendingCpuLocks{false};
std::unordered_map<uint32_t, uint64_t> g_cpuLockedBases;  // base -> frame

static bool BaseIsCpuOwned(uint32_t base) {
  auto it = g_cpuLockedBases.find(base & 0x1FFFF000u);
  return it != g_cpuLockedBases.end() && g_frameIndex - it->second < 600;
}

static void DrainGuestCpuLocks() {
  if (!g_hasPendingCpuLocks.load(std::memory_order_acquire)) return;
  std::vector<uint32_t> bases;
  {
    std::lock_guard<std::mutex> lk(g_invalidateMx);
    bases.swap(g_pendingCpuLocks);
    g_hasPendingCpuLocks.store(false, std::memory_order_release);
  }
  for (uint32_t b : bases) {
    g_cpuLockedBases[b] = g_frameIndex;
    g_resolvedBaseToKey.erase(b);
    g_resolvedBaseToKey.erase(b + 0x1000);  // formula-alias registration
  }
  if (g_cpuLockedBases.size() > 512) {  // decay stale stamps
    for (auto it = g_cpuLockedBases.begin(); it != g_cpuLockedBases.end();) {
      it = g_frameIndex - it->second >= 600 ? g_cpuLockedBases.erase(it)
                                            : std::next(it);
    }
  }
}

void DrainGuestTextureInvalidates() {
  DrainGuestCpuLocks();
  if (!g_hasPendingInvalidates.load(std::memory_order_acquire)) return;
  std::vector<std::pair<uint32_t, uint32_t>> ranges;
  {
    std::lock_guard<std::mutex> lk(g_invalidateMx);
    ranges.swap(g_pendingInvalidates);
    g_hasPendingInvalidates.store(false, std::memory_order_release);
    // The paired invalidate marks the rewrite complete: thaw overlapping
    // frozen ranges so heals resume against the finished data.
    for (auto it = g_frozenRanges.begin(); it != g_frozenRanges.end();) {
      bool thaw = false;
      for (const auto& r : ranges) {
        if (std::get<0>(*it) < r.first + r.second &&
            r.first < std::get<0>(*it) + std::get<1>(*it)) {
          thaw = true;
          break;
        }
      }
      it = thaw ? g_frozenRanges.erase(it) : ++it;
    }
    if (g_frozenRanges.empty())
      g_hasFrozenRanges.store(false, std::memory_order_release);
  }
  if (ranges.empty()) return;
  for (auto it = g_textureCache.begin(); it != g_textureCache.end();) {
    CachedTexture& e = it->second;
    // Guest-uploaded entries only: resolve products (host_rendered) live in
    // host memory and are not invalidated by guest writes.
    bool hit = false;
    if (!e.host_rendered) {
      const uint32_t base = (e.header_dw[1] & 0xFFFFF000u) & 0x1FFFFFFFu;
      const uint32_t span = e.approx_bytes ? e.approx_bytes : 4096u;
      for (const auto& r : ranges) {
        if (base < r.first + r.second && r.first < base + span) {
          hit = true;
          break;
        }
      }
    }
    if (hit) {
      for (auto lit = g_lastResolvedBySize.begin();
           lit != g_lastResolvedBySize.end();) {
        if (lit->second == &it->second) {
          lit = g_lastResolvedBySize.erase(lit);
        } else {
          ++lit;
        }
      }
      RetireEntry(it->second);
      it = g_textureCache.erase(it);
    } else {
      ++it;
    }
  }
}

// Format-aware bytes/px estimate for the cache watermark.
uint32_t ApproxTexBytes(uint32_t w, uint32_t h, RenderFormat fmt) {
  uint32_t bpp = 4;
  switch (fmt) {
    case RenderFormat::R16G16B16A16_FLOAT:
      bpp = 8;
      break;
    case RenderFormat::BC1_UNORM:
    case RenderFormat::BC4_UNORM:
      bpp = 1;
      break;
    case RenderFormat::BC2_UNORM:
    case RenderFormat::BC3_UNORM:
    case RenderFormat::BC5_UNORM:
      bpp = 2;  // 1 byte/px block + mips headroom
      break;
    default:
      break;
  }
  return w * h * bpp;
}
uint32_t g_retiredTextureTotal = 0;
// Recycled bindless slots (pushed by the retire drain at its fence-safe
// point, after the slot was scrubbed to blank). Without recycling the
// monotonic counter exhausts the heap over a long session and every new
// texture silently binds the blank descriptor.
std::vector<uint32_t> g_freeDescriptors;

uint32_t AllocTextureDescriptor() {
  if (!g_freeDescriptors.empty()) {
    const uint32_t d = g_freeDescriptors.back();
    g_freeDescriptors.pop_back();
    return d;
  }
  if (g_nextTextureDescriptor < kFirstReservedDescriptor) {
    return g_nextTextureDescriptor++;
  }
  static uint64_t exhaust_logs = 8;
  if (exhaust_logs) {
    exhaust_logs--;
    REXGPU_WARN(
        "videonative: texture descriptor heap EXHAUSTED ({} slots, none "
        "free), new textures will render BLANK",
        kFirstReservedDescriptor);
  }
  return 0;
}
// Guest texel base -> host-rendered entry, for the resolve redirect.
// Keyed on the physical page: dest headers carry virtual addresses while
// sampling headers carry physical ones, and the physical page is the
// canonical identity for both.
constexpr uint32_t kGuestBaseMask = 0x1FFFF000u;
std::unordered_map<uint32_t, uint64_t> g_resolvedBaseToKey;
uint64_t g_redirHitDepth = 0;  // redirect served a host depth texture
// Texture-flow counters:
uint64_t g_regRefreshChanged = 0;   // resolve re-registration corrected a stale mapping
uint64_t g_resolveSrcMissing = 0;   // resolve dropped: bound RT not in cache (dest kept old content)
uint64_t g_regTruthDiverged = 0;    // page-table truth != console formula at registration
uint32_t g_healsThisFrame = 0;      // per-frame heal budget spent (reset at frame roll)
uint64_t g_bakeSrcRefreshes = 0;    // bake-class source uploads forced fresh
// Tile-class (<=256px, the item-grid composite range) redirect counters;
// the headline counters start at 256px and miss this class entirely.
uint64_t g_tileRedirHitCnt = 0, g_tileRedirRejectCnt = 0;
// Tile-class fetches whose base has no registration (falls through to a
// guest-memory upload). Legitimate CPU-art textures land here too.
uint64_t g_tileRedirMissCnt = 0;
double g_psoCreateMs = 0.0;      // cumulative pipeline-compile wall time
uint64_t g_psoCreateCount = 0;
uint64_t g_redirRegistered = 0;  // resolve destinations registered
uint64_t g_redirRegDepth = 0;    // ...of which depth
uint64_t g_redirHit = 0;         // sample served from a host-rendered entry
uint64_t g_redirMissBig = 0;     // sample >=256px with no registration
uint64_t g_redirReject = 0;      // registered but dimensions mismatched
// Depth-resolve staging: D3D12 only allows whole-subresource copies
// between depth textures, so the whole plane lands here and the requested
// region is copied out. Rotating slots so no single resource churns
// write->read->write within a frame (old Intel drivers mishandle that).
struct DepthStagingSlot {
  std::unique_ptr<RenderBuffer> buffer;
  uint64_t size = 0;
};
DepthStagingSlot g_depthStaging[8];
uint32_t g_depthStagingRot = 0;

// Whole-plane staging memo: several bands resolve per frame from an
// unchanged source, so stage the plane once instead of per band. Keyed on
// the source texture and a mutation counter (any write between resolves
// forces a fresh copy); same-frame only.
struct DepthStageMemo {
  const RenderTexture* src = nullptr;
  RenderBuffer* buffer = nullptr;
  uint64_t frame = ~0ull;
  uint64_t mutation = ~0ull;
  uint32_t host_w = 0, host_h = 0, row_pitch = 0;
};
DepthStageMemo g_depthStageMemo;
// Bumped on every draw and clear: the bound RT's contents may have changed.
uint64_t g_rtMutationSeq = 0;
uint64_t g_depthStageReuses = 0;  // plane copies avoided
uint64_t g_depthShaderResolves = 0;  // depth resolves rendered
uint64_t g_texInvalidations = 0;  // cache drops on guest destroy
std::unique_ptr<RenderShader> g_depthResolvePs;
std::unique_ptr<RenderPipeline> g_depthResolvePipeline;

// ---------------------------------------------------------------------------
// Guest EDRAM surfaces -> host render targets.
//
// CreateSurface is fully replaced by the hook, so the guest surface object is
// this backend's own layout: video_native.cpp writes {magic, width, height,
// format, msaa} and it is parsed back here. 0 / unrecognized surface = the
// swapchain backbuffer.
// ---------------------------------------------------------------------------

constexpr uint32_t kSurfaceMagic = 0x564E5346;  // 'VNSF'

struct SurfaceInfo {
  uint32_t width = 0, height = 0;
  uint32_t guest_format = 0;
  uint32_t edram_base = 0;  // tile index, distinct surfaces must not alias
  int32_t exp_bias = 0;     // RB_COLOR_INFO color exponent bias (signed)
  bool valid = false;
};

// Guest passes render up to 4 MRT color outputs (the lighting/AO bakes write
// oC0+oC1 and the game resolves sources 1-3 separately), every cached RT
// carries the full set so those resolves have real backing.
constexpr uint32_t kMrtCount = 4;

struct CachedRenderTarget {
  std::unique_ptr<RenderTexture> color[kMrtCount];
  std::unique_ptr<RenderTexture> depth;
  std::unique_ptr<RenderFramebuffer> framebuffer;
  uint32_t width = 0, height = 0;
  uint32_t edram_base = 0;
  // LRU stamp + approximate VRAM bytes (5 attachments) for the RT-cache
  // watermark. Without a bound, garbage transient surface parses each
  // allocate a multi-hundred-MB RT set and thrash small-VRAM parts into
  // residency paging.
  uint64_t last_bind_frame = 0;
  uint64_t approx_bytes = 0;
  // Lifetime draw count into this RT. A tile resolve from an RT whose count
  // never moves means the content pass never drew, not that the resolve
  // failed.
  uint64_t draws = 0;
  // Global draw-serial stamps for the small-resolve rebase: when did this
  // RT last receive a draw, and at what draw serial was it last used as a
  // rebased resolve source (freshness = last_draw_serial > sourced stamp).
  uint64_t last_draw_serial = 0;
  uint64_t last_sourced_serial = 0;
  // Depth read as R32_FLOAT for the shader resolve path (a DSV resource is
  // legal to sample through an R32 SRV). Created lazily, once per RT.
  std::unique_ptr<RenderTextureView> depth_srv;
  uint32_t depth_descriptor = 0;
  // Per-slot attachment formats. formats[0] doubles as "the RT's format" for
  // slot-agnostic consumers (g_activeRtFormat, size-matched present pick);
  // unbound slots inherit formats[0] so resolves of sources 1-3 keep backing.
  RenderFormat formats[kMrtCount] = {
      RenderFormat::UNKNOWN, RenderFormat::UNKNOWN, RenderFormat::UNKNOWN,
      RenderFormat::UNKNOWN};
  RenderTextureLayout color_layout[kMrtCount] = {
      RenderTextureLayout::UNKNOWN, RenderTextureLayout::UNKNOWN,
      RenderTextureLayout::UNKNOWN, RenderTextureLayout::UNKNOWN};
  RenderTextureLayout depth_layout = RenderTextureLayout::UNKNOWN;
};

// Predicated-tiling extent (BeginTiling..EndTiling): while nonzero, host RTs
// for tile-height EDRAM surfaces are expanded to cover the full frame.
uint32_t g_tilingWidth = 0;
uint32_t g_tilingHeight = 0;
// Sticky copy of the last bracket extent: the band-widen test needs the
// guest's frame size, not the host window's, which changes on resize.
uint32_t g_lastTilingWidth = 0;
uint32_t g_lastTilingHeight = 0;
// Bracket-less tiled passes (another title's per-frame shadow window: no
// BeginTiling, the tile rects live engine-side): growth is keyed off the draw
// viewport instead, SetupDraw stores the wanted extent here and forces a
// re-apply. Scoped to the triggering guest surface object: the lightmap bake
// binds the same 2080x544 shape but uses the slice model (re-render per tile,
// content re-based to row 0) and must keep its short RT + resolve-rect
// re-base.
uint32_t g_growWidth = 0;
uint32_t g_growHeight = 0;
uint32_t g_growSurface = 0;
bool g_activeRtGrown = false;     // active RT came from viewport growth
uint32_t g_activeRtHeight = 0;    // host height of the active RT
uint32_t g_activeRtWidth = 0;     // host width of the active RT
bool g_activeRtBandWidened = false;  // RT widened from a vertical band surface

uint32_t g_passResolves = 0;      // resolves since the last target change
bool g_scissorExplicit = false;   // guest set a scissor since the last bind
uint32_t LoadGuestU32(uint32_t addr);
const uint8_t* GuestPtr(uint32_t addr);

// Keyed by (width, height, host format), EDRAM surfaces are transient
// scratch, so same-shaped surfaces share one host RT.
std::unordered_map<uint64_t, CachedRenderTarget> g_renderTargets;
uint64_t g_activeRtKey = 0;  // 0 = swapchain framebuffer
// The active RT object + a layouts-dirty flag for the lazy resolve path:
// a mid-pass resolve leaves the active RT's surfaces in COPY_SOURCE; the
// next draw (or same-key rebind) repairs the layouts on demand instead of
// every resolve paying the eager round-trip.
CachedRenderTarget* g_activeRt = nullptr;
bool g_activeRtLayoutsDirty = false;


void RepairActiveRtLayouts() {
  if (!g_activeRtLayoutsDirty) return;
  g_activeRtLayoutsDirty = false;
  CachedRenderTarget* rt = g_activeRt;
  if (!rt) return;
  uint32_t n = 0;
  RenderTextureBarrier barriers[kMrtCount + 1];
  for (uint32_t i = 0; i < kMrtCount; i++) {
    if (rt->color_layout[i] != RenderTextureLayout::COLOR_WRITE) {
      barriers[n++] = RenderTextureBarrier(rt->color[i].get(),
                                           RenderTextureLayout::COLOR_WRITE);
      rt->color_layout[i] = RenderTextureLayout::COLOR_WRITE;
    }
  }
  if (rt->depth_layout != RenderTextureLayout::DEPTH_WRITE) {
    barriers[n++] = RenderTextureBarrier(rt->depth.get(),
                                         RenderTextureLayout::DEPTH_WRITE);
    rt->depth_layout = RenderTextureLayout::DEPTH_WRITE;
  }
  if (n) {
    g_commandLists[g_frame]->barriers(RenderBarrierStage::GRAPHICS, barriers,
                                      n);
  }
}
// The RT the most recent draw targeted, resolves read from here (robust
// against binding/parse drift between the pass's draws and its Resolve).
uint64_t g_lastDrawnRtKey = 0;

// Display gamma-ramp LUT (256x3 R16_UNORM: rows = R,G,B curves) sampled by
// the present blit when bound (b4 [0].y). Content mirrors the guest
// D3DGAMMARAMP the game maintains through D3DDevice_SetGammaRamp; the ring
// backend applies the same ramp in its apply-gamma present pass.
std::unique_ptr<plume::RenderTexture> g_gammaRampTexture;
std::unique_ptr<plume::RenderTextureView> g_gammaRampView;
uint32_t g_gammaRampDescriptor = 0;
RenderFormat g_activeRtFormat = kColorFormat;
// Per-slot formats of the active RT set (mirrors CachedRenderTarget::formats;
// all kColorFormat when the swapchain is bound). Consumed by pipeline
// creation so each MRT slot compiles against its true attachment format.
RenderFormat g_activeRtFormats[4] = {kColorFormat, kColorFormat, kColorFormat,
                                     kColorFormat};

uint32_t LoadGuestU32(uint32_t addr) {
  // rq worker: draw records carry a device-block snapshot served through a
  // thread-local window so deferred execution reads call-time state.
  for (const rq::Win& w : rq::t_win) {
    if (w.host && addr - w.base < w.size) {
      uint32_t v;
      std::memcpy(&v, w.host + (addr - w.base), 4);
      return __builtin_bswap32(v);
    }
  }
  return __builtin_bswap32(
      *rex::system::kernel_memory()->TranslateVirtual<const uint32_t*>(addr));
}
void StoreGuestU32(uint32_t addr, uint32_t value) {
  *rex::system::kernel_memory()->TranslateVirtual<uint32_t*>(addr) =
      __builtin_bswap32(value);
}
const uint8_t* GuestPtr(uint32_t addr) {
  for (const rq::Win& w : rq::t_win) {
    if (w.host && addr - w.base < w.size) return w.host + (addr - w.base);
  }
  return rex::system::kernel_memory()->TranslateVirtual<const uint8_t*>(addr);
}

// Commit probe for guest-sourced uploads: reading uncommitted pages is a
// hard AV, so uploads validate the whole source range and skip instead of
// crashing.
// 360 GPU address translation for header-carried virtual bases (the XDK's
// own idiom): physical = (v & 0x1FFFFFFF) + 0x1000 iff v is in the
// 0xE0000000 window, else + 0. Header bases are virtual, bound sampler
// constants physical; comparisons between the two must use this.
static inline uint32_t GuestPhysFromHeaderBase(uint32_t v) {
  return (v & 0x1FFFFFFFu) + (((v >> 20) + 0x200u) & 0x1000u);
}

// The console formula is exact for the 360's fixed windows but not for
// page-table-mapped ranges, so registrations ask the memory system for
// the real mapping; the formula stays as a fallback second key.
struct ResolveDestKeys {
  uint32_t formula;   // GuestPhysFromHeaderBase(raw)
  uint32_t truth;     // page-table physical, or == formula when unavailable
  bool diverged;      // truth resolved and != formula
};

static inline ResolveDestKeys ResolveDestRegKeys(uint32_t raw_base) {
  ResolveDestKeys k;
  k.formula = GuestPhysFromHeaderBase(raw_base);
  k.truth = k.formula;
  k.diverged = false;
  const uint32_t real =
      rex::system::kernel_memory()->GetPhysicalAddress(raw_base & ~0xFFFu);
  if (real != UINT32_MAX) {
    k.truth = real;
    k.diverged = (real != k.formula);
  }
  return k;
}

bool GuestRangeReadable(const uint8_t* p, size_t size) {
  // Known-committed region map: guest commit is grow-only on this title,
  // so verified extents are cached forever and repeat probes cost no
  // syscalls. Negatives are never cached (commit grows).
  static std::map<uint64_t, uint64_t> committed;  // start -> end (host)
  static std::mutex committed_mx;
  const uint64_t lo = reinterpret_cast<uint64_t>(p);
  const uint64_t hi = lo + size;
  {
    std::lock_guard<std::mutex> lk(committed_mx);
    auto it = committed.upper_bound(lo);
    if (it != committed.begin()) {
      --it;
      if (it->first <= lo && it->second >= hi) return true;
    }
  }
  MEMORY_BASIC_INFORMATION mbi;
  const uint8_t* cur = p;
  const uint8_t* const end = p + size;
  uint64_t span_lo = 0, span_hi = 0;
  while (cur < end) {
    if (!VirtualQuery(cur, &mbi, sizeof(mbi))) return false;
    if (mbi.State != MEM_COMMIT ||
        (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) || mbi.Protect == 0) {
      return false;
    }
    const uint64_t region_lo = reinterpret_cast<uint64_t>(mbi.BaseAddress);
    const uint64_t region_hi = region_lo + mbi.RegionSize;
    if (!span_lo) span_lo = region_lo;
    span_hi = region_hi;
    cur = static_cast<const uint8_t*>(mbi.BaseAddress) + mbi.RegionSize;
  }
  if (span_lo) {
    // Record the full committed extent the walk saw (bigger than the asked
    // range, one probe covers the whole region for future callers), merging
    // any overlapping/adjacent entries.
    std::lock_guard<std::mutex> lk(committed_mx);
    auto it = committed.upper_bound(span_lo);
    if (it != committed.begin()) {
      auto prev = std::prev(it);
      if (prev->second >= span_lo) {
        span_lo = prev->first;
        span_hi = std::max(span_hi, prev->second);
        committed.erase(prev);
      }
    }
    while (true) {
      auto next = committed.lower_bound(span_lo);
      if (next == committed.end() || next->first > span_hi) break;
      span_hi = std::max(span_hi, next->second);
      committed.erase(next);
    }
    committed[span_lo] = span_hi;
  }
  return true;
}

// Guest data reads (texture texels, VB/IB bytes, UP vertices): GPU-visible
// addresses are physical on the 360. Titles that allocate identity-mapped
// (virtual == physical) work through the virtual view alone; system-exe
// allocations back only the physical view, and the virtual view there is
// uncommitted and reads AV. Prefer the virtual view, fall back to the
// physical view when the virtual one isn't committed.
const uint8_t* GuestDataPtr(uint32_t addr, size_t bytes) {
  const uint8_t* v = GuestPtr(addr);
  if (v && GuestRangeReadable(v, bytes)) return v;
  const uint8_t* p =
      rex::system::kernel_memory()->TranslatePhysical<const uint8_t*>(addr);
  if (p && GuestRangeReadable(p, bytes)) return p;
  return nullptr;
}

// Endpoint-probed, memoized variant for hot per-draw paths; VirtualQuery
// is far too slow to run per draw, and grow-only commit makes a resolved
// (addr,size)->view verdict permanent.
// cache_negative: stream paths may cache "unreadable"; texture paths must
// pass false (their data lands in freshly-committed memory after first
// bind and re-probes are already rate-limited).
// physical_first: use for texture reads, whose data often lives only in
// the physical view. Do not use for stream/IB reads (those live in the
// virtual pages here) or app-side CPU pointers; the two views are not
// aliased everywhere, and the right view differs per class.
const uint8_t* GuestDataPtrFast(uint32_t addr, size_t bytes,
                                bool cache_negative = true,
                                bool physical_first = false) {
  if (!bytes) return nullptr;
  // rq worker: a draw-record snapshot window covering addr is authoritative
  // for this record and must be served directly, and never memoized. Win
  // pointers are per-record arena blobs; caching one hands every later draw
  // a dangling pointer into recycled arena. Partial coverage falls through
  // to live views.
  for (const rq::Win& w : rq::t_win) {
    if (w.host && addr - w.base < w.size) {
      if (bytes <= size_t(w.size - (addr - w.base))) {
        return w.host + (addr - w.base);
      }
      break;
    }
  }
  struct ProbeEntry {
    const uint8_t* view;
    uint64_t negative_frame;  // frame of the cached nullptr verdict
  };
  static std::unordered_map<uint64_t, ProbeEntry> cache;
  static std::mutex m;
  const uint64_t key = (uint64_t(addr) << 28) ^ bytes ^
                       (physical_first ? 0x8000000000000000ull : 0ull);
  {
    std::lock_guard<std::mutex> lk(m);
    auto it = cache.find(key);
    if (it != cache.end()) {
      if (it->second.view) return it->second.view;
      if (cache_negative &&
          g_frameIndex - it->second.negative_frame < 300) {
        return nullptr;
      }
      cache.erase(it);  // stale/uncached-class negative, re-probe
    }
  }
  // Raw views only below, GuestPtr would resolve through the rq windows
  // and the persistent cache must never store an arena-backed pointer.
  const uint8_t* r = nullptr;
  const uint8_t* first =
      physical_first
          ? rex::system::kernel_memory()->TranslatePhysical<const uint8_t*>(addr)
          : rex::system::kernel_memory()->TranslateVirtual<const uint8_t*>(addr);
  const uint8_t* second =
      physical_first
          ? rex::system::kernel_memory()->TranslateVirtual<const uint8_t*>(addr)
          : rex::system::kernel_memory()->TranslatePhysical<const uint8_t*>(addr);
  if (first && GuestRangeReadable(first, 4) &&
      GuestRangeReadable(first + bytes - 4, 4)) {
    r = first;
  } else if (second && GuestRangeReadable(second, 4) &&
             GuestRangeReadable(second + bytes - 4, 4)) {
    r = second;
  }
  {
    std::lock_guard<std::mutex> lk(m);
    if (cache.size() > 4096) cache.clear();  // unbounded-growth backstop
    if (r || cache_negative) {
      cache[key] = {r, r ? 0 : g_frameIndex};
    }
  }
  return r;
}

// Persistent effective blend/mask registers, the PM4 register file draws
// consume. On hardware every RB_* write (live setter flush or PM4 baked
// into a command buffer) lands in one global register set that persists
// until rewritten; per-draw shadow/record scoping would lose CB-set state
// as soon as the replay ended.
// Fed from CbCall::st_dirty slots on replayed draws and from the XDK's own
// dev+16 dirty bits on live draws (see the draw-state block in SetupDraw).
uint32_t g_fxBlendControl[4] = {0x00010001u, 0x00010001u, 0x00010001u,
                                0x00010001u};
uint32_t g_fxColorMask = 0xFFFFu;

// 360 texture tiling (Xenia's TiledOffset2D functions).
uint32_t TiledOffset2DOuter(uint32_t y, uint32_t width, uint32_t log2_bpp) {
  uint32_t macro = ((y >> 5) * (width >> 5)) << (log2_bpp + 7);
  uint32_t micro = ((y & 6) << 2) << log2_bpp;
  return macro + ((micro & ~0xFu) << 1) + (micro & 0xF) +
         ((y & 8) << (3 + log2_bpp)) + ((y & 1) << 4);
}
uint32_t TiledOffset2DInner(uint32_t x, uint32_t y, uint32_t log2_bpp,
                            uint32_t base_offset) {
  uint32_t macro = (x >> 5) << (log2_bpp + 7);
  uint32_t micro = (x & 7) << log2_bpp;
  uint32_t offset =
      base_offset + (macro + ((micro & ~0xFu) << 1) + (micro & 0xF));
  return ((offset & ~0x1FFu) << 3) + ((offset & 0x1C0) << 2) + (offset & 0x3F) +
         ((y & 16) << 7) + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6);
}

void UntileSurface(uint8_t* dst, const uint8_t* src, uint32_t pitch_blocks,
                   uint32_t width_blocks, uint32_t height_blocks,
                   uint32_t bytes_per_block, uint32_t x0_blocks = 0,
                   uint32_t y0_blocks = 0) {
  uint32_t log2_bpp = 0;
  switch (bytes_per_block) {
    case 1: log2_bpp = 0; break;
    case 2: log2_bpp = 1; break;
    case 4: log2_bpp = 2; break;
    case 8: log2_bpp = 3; break;
    case 16: log2_bpp = 4; break;
    default: return;
  }
  // x0/y0: packed-mip-tail block offset, textures with min dimension
  // <= 16 texels live inside the 32x32 tile at a nonzero origin (ring
  // texture/util.cpp GetPackedMipOffset; the 16x16 base level sits at
  // x=16,y=0). Untiling from the tile origin reads the 4x4/8x8 packing
  // region instead and corrupts the shading ramps.
  for (uint32_t y = 0; y < height_blocks; y++) {
    uint32_t outer = TiledOffset2DOuter(y + y0_blocks, pitch_blocks, log2_bpp);
    for (uint32_t x = 0; x < width_blocks; x++) {
      uint32_t src_off =
          TiledOffset2DInner(x + x0_blocks, y + y0_blocks, log2_bpp, outer);
      std::memcpy(dst + (y * width_blocks + x) * bytes_per_block, src + src_off,
                  bytes_per_block);
    }
  }
}

void SwapBytes16(uint8_t* data, size_t size) {
  auto* p = reinterpret_cast<uint16_t*>(data);
  for (size_t i = 0; i < size / 2; i++) p[i] = __builtin_bswap16(p[i]);
}
void SwapBytes32(uint8_t* data, size_t size) {
  auto* p = reinterpret_cast<uint32_t*>(data);
  for (size_t i = 0; i < size / 4; i++) p[i] = __builtin_bswap32(p[i]);
}

// Parse a guest GPUTEXTURE_FETCH header (6 BE dwords) and realize a host
// texture; returns the cached entry (invalid on unsupported format).
// for_resolve skips the guest-memory upload (content comes from host RT
// copies). override_w/h and override_format force the allocation to match
// the source RT (D3D12 copies cannot resize or convert like hardware
// resolves can). resolve_scale: 0 = global host scale, 1 = 1:1, -1 =
// accept any scale.
CachedTexture* GetOrCreateTexture(uint32_t header_addr,
                                  bool for_resolve = false,
                                  uint32_t override_w = 0,
                                  uint32_t override_h = 0,
                                  RenderFormat override_format =
                                      RenderFormat::UNKNOWN,
                                  const uint32_t* dw_snapshot = nullptr,
                                  int resolve_scale = 0) {
  // Kernel-reported guest rewrites retire overlapping entries before any
  // lookup this draw (lock-free no-op when nothing is pending).
  DrainGuestTextureInvalidates();
  uint32_t dw[6];
  if (dw_snapshot) {
    // Bind-time snapshot (XDK copy semantics), see SetTexture. Never re-read
    // the header here: it may hold a recycled block's dwords by draw time.
    std::memcpy(dw, dw_snapshot, sizeof(dw));
  } else {
    for (int i = 0; i < 6; i++) dw[i] = LoadGuestU32(header_addr + i * 4);
  }
  uint64_t key = XXH3_64bits(dw, sizeof(dw));
  // CPU-owned bases (guest LockRect tell): the guest composes tiles in
  // these pages, so a sampling bind must not serve the raw host resolve
  // product, neither via the redirect nor via a direct key hit on the
  // resolve-dest entry (a tile draw's header can hash identically to the
  // dest entry's). Salting the key diverts the bind to a separate
  // guest-upload entry; the RT entry stays for resolves.
  const bool cpu_owned =
      !for_resolve && BaseIsCpuOwned(dw[1] & kGuestBaseMask);
  if (cpu_owned) {
    key ^= 0x9E3779B97F4A7C15ull;
  }
  // Sampling a resolved base address hits the host-rendered entry even when
  // the sampling header's flag bits differ from the resolve-dest header's.
  if (!for_resolve && !cpu_owned &&
      REXCVAR_GET(native_video_resolve_redirect)) {
    auto resolved = g_resolvedBaseToKey.find(dw[1] & kGuestBaseMask);
    // Packed-mip-tail adjust: for a mipped texture the level-0 image
    // starts at +0x1000 past the packed tail, so samplers ask one page
    // above the registered resolve dest. Probe one page down; serve on
    // exact dims match only.
    if (resolved == g_resolvedBaseToKey.end() &&
        REXCVAR_GET(native_video_redirect_serve_subdim)) {
      auto below = g_resolvedBaseToKey.find((dw[1] & kGuestBaseMask) - 0x1000);
      if (below != g_resolvedBaseToKey.end()) {
        auto bit = g_textureCache.find(below->second);
        if (bit != g_textureCache.end() && bit->second.host_rendered &&
            bit->second.width == (dw[2] & 0x1FFF) + 1 &&
            bit->second.height == ((dw[2] >> 13) & 0x1FFF) + 1) {
          resolved = below;
        }
      }
    }
    // Stale-registration drop: the guest recycles resolve-dest addresses
    // for unrelated textures, and dims cannot discriminate. Recency can:
    // a registration idle on both sample and resolve axes is dead, so
    // drop it and fall through to the guest upload.
    if (resolved != g_resolvedBaseToKey.end()) {
      auto sit = g_textureCache.find(resolved->second);
      const uint64_t redirect_ttl = 120;  // frames
      if (sit == g_textureCache.end() ||
          (g_frameIndex > sit->second.last_resolve_frame + redirect_ttl &&
           g_frameIndex > sit->second.last_bind_frame + redirect_ttl)) {
        g_resolvedBaseToKey.erase(resolved);
        resolved = g_resolvedBaseToKey.end();
      }
    }
    if (resolved != g_resolvedBaseToKey.end()) {
      auto rit = g_textureCache.find(resolved->second);
      if (rit != g_textureCache.end() &&
          rit->second.width >=
              uint32_t(REXCVAR_GET(native_video_redirect_min_dim))) {
        // The redirect must only serve samples of the same surface: match
        // the requesting header's dimensions against the resolved entry.
        // Base-only matching would hijack unrelated textures whose guest
        // allocation reused a resolve destination's base.
        const uint32_t req_w = (dw[2] & 0x1FFF) + 1;
        const uint32_t req_h = ((dw[2] >> 13) & 0x1FFF) + 1;
        CachedTexture& rt = rit->second;
        // Tile-composite idiom: the guest re-samples a resolve product
        // through a smaller texture header. Serving the host entry keeps
        // normalized-UV sampling correct and matches the ring's behavior
        // of serving resolved content for any fetch of the region.
        const bool subdim_fit =
            REXCVAR_GET(native_video_redirect_serve_subdim) &&
            rt.host_rendered && req_w <= rt.width && req_h <= rt.height;
        if ((rt.width == req_w && rt.height == req_h) || subdim_fit ||
            !rt.host_rendered) {
          // LRU stamp: one-time bakes (backdrop layers resolved once at
          // level load) are sampled exclusively through this redirect;
          // without the stamp they look eternally idle and the watermark
          // evicts them on long runs, serving stale guest bytes once the
          // base->key mapping dies with the entry.
          rt.last_bind_frame = g_frameIndex;
          g_redirHit++;
          if (req_w <= 256 && req_h <= 256) g_tileRedirHitCnt++;
          if (rt.host_format == RenderFormat::D32_FLOAT ||
              rt.host_format == RenderFormat::R32_FLOAT) {
            g_redirHitDepth++;
          }
          return &rt;
        }
        g_redirReject++;
        if (req_w <= 256 && req_h <= 256) g_tileRedirRejectCnt++;
      }
    } else {
      // No registration for this base at all: the sample falls through to a
      // guest-memory upload instead of the host texture the resolve
      // rendered.
      const uint32_t req_w = (dw[2] & 0x1FFF) + 1;
      const uint32_t req_h = ((dw[2] >> 13) & 0x1FFF) + 1;
      if (req_w >= 16 && req_w <= 256 && req_h <= 256) g_tileRedirMissCnt++;
      if (req_w >= 256) {
        g_redirMissBig++;
      }
    }
  }
  auto it = g_textureCache.find(key);
  if (it != g_textureCache.end()) {
    it->second.last_bind_frame = g_frameIndex;  // hot, evict sweep skips it
    // Last writer wins: refresh the base->key registration on every
    // resolve, not only at entry creation; whoever wrote a base last owns
    // its content.
    if (for_resolve && it->second.valid) {
      // Refresh both keys (page-table truth + formula).
      const ResolveDestKeys rk = ResolveDestRegKeys(dw[1] & 0xFFFFF000u);
      const uint32_t reg_keys[2] = {rk.truth, rk.formula};
      const int nkeys = rk.diverged ? 2 : 1;
      for (int ki = 0; ki < nkeys; ki++) {
        auto [rit, inserted] = g_resolvedBaseToKey.try_emplace(reg_keys[ki], key);
        if (!inserted && rit->second != key) {
          rit->second = key;
          g_regRefreshChanged++;
        } else if (inserted) {
          g_regRefreshChanged++;
        }
      }
    }
    // An entry refused at upload time (unsupported format) must not block a
    // resolve destination, rebuild it as one. Same when a resolve needs
    // different host dimensions or a different host format (format-converting
    // hardware resolve) than the entry has.
    const bool dims_ok =
        !override_w ||
        (it->second.width == override_w && it->second.height == override_h);
    const bool format_ok = override_format == RenderFormat::UNKNOWN ||
                           it->second.host_format == override_format;
    // A resolve destination must be allocated at the requested host scale.
    const bool scale_ok =
        resolve_scale < 0 ||
        it->second.host_scale ==
            (resolve_scale == 1 ? uint8_t(0x11) : HostScaleCode());
    if (!for_resolve || (it->second.valid && dims_ok && format_ok && scale_ok)) {
      // Re-upload on content change: animated/streamed textures rewrite
      // guest texel data after the first upload (effect frames, avatar face
      // composites). Detect via a sparse content hash (small guest-uploaded
      // textures, every 8th frame, first 2KB) and rebuild the entry through
      // the normal creation path below. The stale entry is retired, not
      // destroyed; in-flight frames may still reference it.
      CachedTexture& e = it->second;
      bool reupload = false;
      // Bounded healing: re-check content only while the entry is young
      // and at most twice. An aged entry is frozen; its guest memory may
      // have been recycled, and a late re-check would heal it into the
      // recycled occupant.
      const uint64_t heal_window =
          uint64_t(std::max(0, REXCVAR_GET(native_video_tex_heal_window)));
      bool header_alive = true;
      if (dw_snapshot) {
        // Full 6-dword liveness: the live header must still equal the bind
        // snapshot. A 1-dword base check would pass for freed-but-intact
        // header blocks.
        for (int i = 0; i < 6 && header_alive; i++) {
          header_alive = LoadGuestU32(header_addr + i * 4) == dw[i];
        }
      }
      // Skipped-upload heal: the source range was uncommitted at create
      // time (see the commit probe); once the guest commits it, re-upload.
      if (!for_resolve && e.valid && e.upload_skipped &&
          e.last_content_check != g_frameIndex && (g_frameIndex & 7) == 0 &&
          !IsGuestRangeFrozen(dw[1] & 0xFFFFF000u, 2048u)) {
        e.last_content_check = g_frameIndex;
        const uint8_t* texels =
            GuestDataPtrFast(dw[1] & 0xFFFFF000u, 2048, false, true);
        if (texels) {
          reupload = true;
        }
      }
      // Per-frame heal budget, not lifetime (see the cvar): a lifetime
      // gate exhausts at boot and freezes every later CPU write out of the
      // cache for the whole session. Checks stop for the frame once the
      // budget is spent; the change re-fires on a later check because the
      // content hash is only stored when a heal runs.
      if (!for_resolve && e.valid && !e.host_rendered && e.width <= 256 &&
          e.height <= 256 && e.last_content_check != g_frameIndex &&
          (g_frameIndex & 7) == 0 &&
          g_healsThisFrame <
              uint32_t(std::max(
                  0, REXCVAR_GET(native_video_tex_heal_frame_budget))) &&
          REXCVAR_GET(native_video_tex_reupload) &&
          g_frameIndex - e.created_frame < heal_window &&
          (REXCVAR_GET(native_video_tex_max_heals) <= 0 ||
           e.content_changes <
               uint32_t(REXCVAR_GET(native_video_tex_max_heals))) &&
          header_alive &&
          !IsGuestRangeFrozen(dw[1] & 0xFFFFF000u,
                              e.approx_bytes ? e.approx_bytes : 2048u)) {
        e.last_content_check = g_frameIndex;
        // Three-window content hash (glyph atlases): CPU glyph
        // rasterization lands anywhere in the page, and a head-only hash
        // sees just the first rows, so most glyph writes never trigger a
        // heal. Sample start/middle/end windows across the whole span.
        const uint32_t span =
            e.approx_bytes ? e.approx_bytes : 2048u;
        const uint8_t* texels =
            GuestDataPtrFast(dw[1] & 0xFFFFF000u, span, false, true);
        if (texels) {
          uint64_t h;
          if (span <= 6144) {
            h = XXH3_64bits(texels, span);
          } else {
            XXH3_state_t* st = XXH3_createState();
            XXH3_64bits_reset(st);
            XXH3_64bits_update(st, texels, 2048);
            XXH3_64bits_update(st, texels + (span / 2 & ~15u), 2048);
            XXH3_64bits_update(st, texels + span - 2048, 2048);
            h = XXH3_64bits_digest(st);
            XXH3_freeState(st);
          }
          if (e.content_hash && h != e.content_hash) {
            reupload = true;
            g_healsThisFrame++;
            ++e.content_changes;
          }
          e.content_hash = h;
        }
      }
      if (!reupload) return &it->second;
      g_retiredTextureTotal++;
      // Drop any last-resolve pointers into the entry before retiring it,
      // then erase once and recreate below; `it` must not be touched after
      // the erase.
      for (auto lit = g_lastResolvedBySize.begin();
           lit != g_lastResolvedBySize.end();) {
        if (lit->second == &it->second) {
          lit = g_lastResolvedBySize.erase(lit);
        } else {
          ++lit;
        }
      }
      RetireEntry(it->second, "hit-reupload");
      g_textureCache.erase(it);
      // fall through to full recreation + upload with the fresh guest data
    } else {
      for (auto lit = g_lastResolvedBySize.begin();
           lit != g_lastResolvedBySize.end();) {
        if (lit->second == &it->second) {
          lit = g_lastResolvedBySize.erase(lit);
        } else {
          ++lit;
        }
      }
      // Retire rather than destroy: in-flight frames may still sample it.
      RetireEntry(it->second, "hit-recreate");
      g_textureCache.erase(it);
    }
  }

  CachedTexture entry;
  entry.created_frame = g_frameIndex;  // bounds the healing window
  // xe_gpu_texture_fetch_t dword0: type:2 .. pitch:9@22. Conservative read:
  // pitch (in 32-texel tiles) at bits 22..30, tiled at 31.
  const uint32_t pitch_tiles = (dw[0] >> 22) & 0x1FF;
  const bool tiled = (dw[0] >> 31) & 1;
  const uint32_t format = dw[1] & 0x3F;
  const uint32_t endian = (dw[1] >> 6) & 0x3;
  const uint32_t base_address = (dw[1] & 0xFFFFF000);
  const uint32_t width = (dw[2] & 0x1FFF) + 1;
  const uint32_t height = ((dw[2] >> 13) & 0x1FFF) + 1;
  // DataDimension at dword5 bits 9-10 (xenos.h): 0=1D, 1=2D/stacked, 2=3D,
  // 3=cube. Cube maps (level backdrops / sky) are 6 consecutive tiled 2D
  // faces in guest memory and need a cube SRV; sampling them through the 2D
  // array view is undefined.
  const uint32_t dimension = (dw[5] >> 9) & 3;
  const bool cube = dimension == 3;
  if (dimension == 2) {
    static uint64_t vol_logs = 4;
    if (vol_logs) {
      vol_logs--;
      REXGPU_WARN("videonative: 3D texture {}x{} fmt={} unsupported (treated "
                  "as 2D)",
                  width, height, format);
    }
  }

  RenderFormat host_format = RenderFormat::UNKNOWN;
  uint32_t block_dim = 1, bytes_per_block = 0;
  switch (format) {
    case 6:  // k_8_8_8_8
      host_format = RenderFormat::R8G8B8A8_UNORM;
      bytes_per_block = 4;
      break;
    case 7:   // k_2_10_10_10 (packed layout matches DXGI after 8-in-32 swap)
    case 54:  // k_2_10_10_10_AS_16_16_16_16, the resolve alias of 2_10_10_10
              // (light-accum RT textures). A 10-bit format: the XDK's
              // D3D_SurfaceInitFields normalizes 54 -> 7, and mapping it to
              // RGBA8 bands the light accum.
      host_format = RenderFormat::R10G10B10A2_UNORM;
      bytes_per_block = 4;
      break;
    case 32:  // k_16_16_16_16_FLOAT (A16B16G16R16F, bloom chain, RTT type 3)
      host_format = RenderFormat::R16G16B16A16_FLOAT;
      bytes_per_block = 8;
      break;
    case 2:  // k_8
      host_format = RenderFormat::R8_UNORM;
      bytes_per_block = 1;
      break;
    case 10:  // k_8_8 (two-channel 8-bit; avatar item textures use it for
              // small maps).
      host_format = RenderFormat::R8G8_UNORM;
      bytes_per_block = 2;
      break;
    case 18:  // k_DXT1
      host_format = RenderFormat::BC1_UNORM;
      block_dim = 4;
      bytes_per_block = 8;
      break;
    case 19:  // k_DXT2_3
      host_format = RenderFormat::BC2_UNORM;
      block_dim = 4;
      bytes_per_block = 16;
      break;
    case 20:  // k_DXT4_5
      host_format = RenderFormat::BC3_UNORM;
      block_dim = 4;
      bytes_per_block = 16;
      break;
    case 22:  // k_24_8 (depth-as-texture)
    case 23:  // k_24_8_FLOAT, depth resolve destinations; host depth copies
              // land here as R32_FLOAT (value semantics match: 0..1 depth).
      host_format = RenderFormat::R32_FLOAT;
      bytes_per_block = 4;
      break;
    default:
      REXGPU_WARN("videonative: unsupported texture format {} ({}x{})", format,
                  width, height);
      entry.valid = false;
      {
    auto prior = g_textureCache.find(key);
    if (prior != g_textureCache.end()) RetireEntry(prior->second);
  }
  entry.last_bind_frame = g_frameIndex;
  entry.approx_bytes = entry.texture ? ApproxTexBytes(entry.width, entry.height, entry.host_format) : 0;
  g_texCacheBytes += entry.approx_bytes;
  entry.header_addr = header_addr;
  std::memcpy(entry.header_dw, dw, sizeof(entry.header_dw));
  return &(g_textureCache[key] = std::move(entry));
  }

  if (for_resolve) {
    // Host-rendered: allocate only; resolves copy content in. The host
    // format follows the source RT when the caller says so (hardware
    // resolves convert formats; host copies cannot).
    if (override_format != RenderFormat::UNKNOWN) {
      host_format = override_format;
    }
    const uint32_t alloc_w = override_w ? override_w : width;
    const uint32_t alloc_h = override_h ? override_h : height;
    const bool alloc_full = resolve_scale == 1;
    RenderTextureDesc rdesc = RenderTextureDesc::Texture2D(
        alloc_full ? alloc_w : HostDim(alloc_w),
        alloc_full ? alloc_h : HostDim(alloc_h), 1, host_format);
    // Depth resolve destinations are render targets under the shader resolve
    // path: it renders the region in instead of staging the whole depth plane
    // through a buffer (D3D12 forbids partial copies out of a depth
    // resource). R32_FLOAT dests only, this is the depth-resolve format.
    if (host_format == RenderFormat::R32_FLOAT &&
        REXCVAR_GET(native_video_depth_resolve_shader)) {
      rdesc.flags |= RenderTextureFlag::RENDER_TARGET;
      entry.rt_capable = true;
    }
    entry.texture = g_device->createTexture(rdesc);
    if (!entry.texture) {
      REXGPU_WARN(
          "videonative: resolve tex creation FAILED ({}x{} fmt={}), entry "
          "invalid",
          alloc_w, alloc_h, uint32_t(host_format));
      entry.valid = false;
      auto prior2 = g_textureCache.find(key);
      if (prior2 != g_textureCache.end()) RetireEntry(prior2->second);
      entry.last_bind_frame = g_frameIndex;
  entry.approx_bytes = entry.texture ? ApproxTexBytes(entry.width, entry.height, entry.host_format) : 0;
  g_texCacheBytes += entry.approx_bytes;
  entry.header_addr = header_addr;
  std::memcpy(entry.header_dw, dw, sizeof(entry.header_dw));
  return &(g_textureCache[key] = std::move(entry));
    }
    entry.host_scale = alloc_full ? uint8_t(0x11) : HostScaleCode();
    entry.texture->setName(
        fmt::format("resolve tex {}x{} (sc {:#x}) base={:#x}", alloc_w,
                    alloc_h, entry.host_scale, base_address));
    // 2D-array view: the pack contract declares xe_textures_2d as
    // Texture2DArray (Xenos binds all 2D textures as arrays); a plain 2D SRV
    // through an array declaration is undefined in D3D12.
    entry.view = entry.texture->createTextureView(
        RenderTextureViewDesc::Texture2DArray(host_format));
    entry.width = alloc_w;
    entry.height = alloc_h;
    entry.host_format = host_format;
    entry.host_rendered = true;
    entry.layout = RenderTextureLayout::UNKNOWN;
    entry.descriptor_index = AllocTextureDescriptor();
    if (entry.descriptor_index) {
      g_textureDescriptorSet->setTexture(entry.descriptor_index,
                                         entry.texture.get(),
                                         RenderTextureLayout::SHADER_READ,
                                         entry.view.get());
    }
    entry.valid = true;
    // Register under the physical page every sampler constant will
    // carry, keyed both by the page-table truth and the console formula.
    // CPU-owned bases never register; the suppression must happen here
    // because per-frame re-bakes re-register before the tile draw.
    // Registration is size-agnostic: a >=256 gate would hide the tile class.
    if (!BaseIsCpuOwned(base_address)) {
      const ResolveDestKeys rk = ResolveDestRegKeys(base_address);
      g_resolvedBaseToKey[rk.truth] = key;
      if (rk.diverged) {
        g_resolvedBaseToKey[rk.formula] = key;
        g_regTruthDiverged++;
      }
    }
    g_redirRegistered++;
    if (host_format == RenderFormat::D32_FLOAT ||
        host_format == RenderFormat::R32_FLOAT) {
      g_redirRegDepth++;
    }
    {
    auto prior = g_textureCache.find(key);
    if (prior != g_textureCache.end()) RetireEntry(prior->second);
  }
  entry.last_bind_frame = g_frameIndex;
  entry.approx_bytes = entry.texture ? ApproxTexBytes(entry.width, entry.height, entry.host_format) : 0;
  g_texCacheBytes += entry.approx_bytes;
  entry.header_addr = header_addr;
  std::memcpy(entry.header_dw, dw, sizeof(entry.header_dw));
  return &(g_textureCache[key] = std::move(entry));
  }

  const uint32_t width_blocks = (width + block_dim - 1) / block_dim;
  const uint32_t height_blocks = (height + block_dim - 1) / block_dim;
  // Tiled textures have a hard minimum pitch of one tile (32 texels);
  // the pitch field can read 0 for sub-tile textures and a narrower
  // fallback garbles macro-tile addressing. Clamp to the tile width.
  uint32_t pitch_blocks =
      pitch_tiles ? (pitch_tiles * 32 / block_dim) : width_blocks;
  if (tiled && REXCVAR_GET(native_video_untile_packed)) {
    pitch_blocks = std::max(pitch_blocks, 32u / block_dim);
  }
  const uint32_t data_size = width_blocks * height_blocks * bytes_per_block;
  // Cube faces are stored consecutively, each laid out like a full 2D
  // texture; tiled faces are padded to the 32-texel tile grid vertically.
  const uint32_t tile_blocks = 32 / block_dim;
  const uint32_t aligned_height_blocks =
      tiled ? ((height_blocks + tile_blocks - 1) & ~(tile_blocks - 1))
            : height_blocks;
  const uint32_t face_stride =
      pitch_blocks * aligned_height_blocks * bytes_per_block;
  const uint32_t face_count = cube ? 6 : 1;

  // D3D12 rejects a BC resource whose dimensions are not a multiple of
  // the 4x4 block (some avatar items carry 34x34 DXT5). Allocate the
  // block-rounded size; keep the logical dims for cache identity.
  const uint32_t alloc_w =
      block_dim > 1 ? width_blocks * block_dim : width;
  const uint32_t alloc_h =
      block_dim > 1 ? height_blocks * block_dim : height;
  RenderTextureDesc desc =
      cube ? RenderTextureDesc::Texture(RenderTextureDimension::TEXTURE_2D,
                                        alloc_w, alloc_h, 1, 1, 6, host_format)
           : RenderTextureDesc::Texture2D(alloc_w, alloc_h, 1, host_format);
  entry.texture = g_device->createTexture(desc);
  if (!entry.texture) {
    // Never dereference an unvalidated resource: a refused allocation must
    // degrade to "no texture" (blank), not take the process down.
    static uint64_t create_fail_logs = 12;
    if (create_fail_logs) {
      create_fail_logs--;
      REXGPU_WARN(
          "videonative: texture creation REFUSED {}x{} (alloc {}x{}) fmt={} "
          "host_fmt={} cube={}, entry left blank",
          width, height, alloc_w, alloc_h, format, uint32_t(host_format),
          cube);
    }
    entry.valid = false;
    auto prior_fail = g_textureCache.find(key);
    if (prior_fail != g_textureCache.end()) RetireEntry(prior_fail->second);
    entry.last_bind_frame = g_frameIndex;
    entry.header_addr = header_addr;
    std::memcpy(entry.header_dw, dw, sizeof(entry.header_dw));
    return &(g_textureCache[key] = std::move(entry));
  }
  entry.texture->setName(
      fmt::format("guest tex {}x{} fmt={}", width, height, format));
  // Fetch-constant destination swizzle (dword3 bits 1-12, 3 bits per
  // output: 0-3 = data X..W, 4/6 = zero, 5/7 = one). Applied through the
  // SRV component mapping; the swizzle dwords are part of the cache key.
  RenderComponentMapping component_mapping;
  // Swizzle policy: honor pure channel-reorder swizzles; keep identity
  // whenever any selector forces 0/1. Replace with the full ring
  // composition once the per-endian upload convention is understood.
  const uint32_t guest_swizzle_raw = (dw[3] >> 1) & 0xFFF;
  bool has_const_selector = false;
  for (uint32_t i = 0; i < 4; i++) {
    if (((guest_swizzle_raw >> (3 * i)) & 0b100) != 0) has_const_selector = true;
  }
  if (REXCVAR_GET(native_video_tex_swizzle_full)) {
    // Faithful ring composition: data selectors (0-3) index the format's
    // host base swizzle, 4/6 -> zero, 5/7 -> one. Base swizzles mirror
    // the d3d12 texture cache's host_formats_[].swizzle:
    static constexpr uint16_t kRGBA = 0x688, kRRRR = 0x000, kRGGG = 0x248,
                              kRGBB = 0x488, kRBGG = 0x250, k0000 = 0x924;
    static constexpr uint16_t kHostFormatSwizzle[64] = {
        kRRRR, kRRRR, kRRRR, kRGBA, kRGBB, kRBGG, kRGBA, kRGBA,  // 0-7
        kRRRR, kRRRR, kRGGG, kRGBB, kRGBB, kRGGG, kRGBA, kRGBA,  // 8-15
        kRGBB, kRGBB, kRGBA, kRGBA, kRGBA, kRGBA, kRRRR, kRRRR,  // 16-23
        kRRRR, kRGGG, kRGBA, kRRRR, kRGGG, kRGBA, kRRRR, kRGGG,  // 24-31
        kRGBA, kRRRR, kRGGG, kRGBA, kRRRR, kRGGG, kRGBA, kRRRR,  // 32-39
        kRGGG, kRRRR, kRGGG, kRRRR, kRRRR, kRGGG, kRRRR, kRRRR,  // 40-47
        kRGGG, kRGGG, kRGBA, kRGBA, kRGBA, kRGBA, kRGBA, kRGBB,  // 48-55
        kRGBB, kRGBB, kRRRR, kRRRR, kRGGG, kRGBA, kRGBA, kRGBA,  // 56-63
    };
    const uint32_t base = kHostFormatSwizzle[format & 63];
    RenderSwizzle* out[4] = {&component_mapping.r, &component_mapping.g,
                             &component_mapping.b, &component_mapping.a};
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t sel = (guest_swizzle_raw >> (3 * i)) & 0x7;
      uint32_t host;
      if (sel & 0b100) {
        host = sel & 0b101;  // 4/6 -> 4 (zero), 5/7 -> 5 (one)
      } else {
        host = (base >> (3 * sel)) & 0x7;
      }
      *out[i] = host == 4 ? RenderSwizzle::ZERO
                : host == 5
                    ? RenderSwizzle::ONE
                    : RenderSwizzle(uint32_t(RenderSwizzle::R) + host);
    }
  } else if (REXCVAR_GET(native_video_tex_swizzle) && !has_const_selector) {
    const bool host_rrrr = (format == 2 || format == 22 || format == 23);
    const uint32_t guest_swizzle = guest_swizzle_raw;
    RenderSwizzle* out[4] = {&component_mapping.r, &component_mapping.g,
                             &component_mapping.b, &component_mapping.a};
    for (uint32_t i = 0; i < 4; i++) {
      const uint32_t sel = (guest_swizzle >> (3 * i)) & 0x7;
      if (sel & 0b100) {
        *out[i] = (sel & 1) ? RenderSwizzle::ONE : RenderSwizzle::ZERO;
      } else {
        const uint32_t host_component = host_rrrr ? 0 : sel;
        *out[i] =
            RenderSwizzle(uint32_t(RenderSwizzle::R) + host_component);
      }
    }
  }
  RenderTextureViewDesc view_desc =
      cube ? RenderTextureViewDesc::TextureCube(host_format)
           : RenderTextureViewDesc::Texture2DArray(host_format);
  view_desc.componentMapping = component_mapping;
  entry.view = entry.texture->createTextureView(view_desc);
  entry.view_dim = cube ? 3 : 1;
  entry.width = width;
  entry.height = height;
  entry.host_format = host_format;

  // Upload through the frame's ring + inline copy on the direct queue.
  // D3D12 placed footprints require a 256-byte-aligned row pitch, small
  // textures (e.g. 64x64 DXT1 = 128-byte rows) must be re-strided into the
  // staging copy, or the copy is illegal (device removal on AMD).
  const uint32_t row_bytes = width_blocks * bytes_per_block;
  const uint32_t padded_row_bytes = (row_bytes + 255u) & ~255u;
  const uint32_t staging_size = padded_row_bytes * height_blocks;
  auto& commandList = g_commandLists[g_frame];
  const bool zero_depth =
      REXCVAR_GET(native_video_unresolved_depth_zero) && !for_resolve &&
      (format == 22 || format == 23) &&
      !g_resolvedBaseToKey.count(base_address & kGuestBaseMask);
  commandList->barriers(RenderBarrierStage::COPY,
                        RenderTextureBarrier(entry.texture.get(),
                                             RenderTextureLayout::COPY_DEST));
  std::vector<uint8_t> linear(data_size);
  for (uint32_t face = 0; face < face_count; face++) {
    const size_t face_bytes = size_t(pitch_blocks) *
                              (tiled ? aligned_height_blocks : height_blocks) *
                              bytes_per_block;
    // Memoized endpoint probe, not the full VirtualQuery walk: the walk is
    // far too slow to run once per upload, and navigation is an upload storm.
    const uint8_t* src =
        GuestDataPtrFast(base_address + face * face_stride, face_bytes, false,
                         true);
    // Frozen-range create hole: the freeze bracket gates heals, but a
    // fresh bind during a frozen window creates a new entry and would
    // upload the half-rewritten buffer. Defer exactly like an uncommitted
    // source: zero texture now; the commit recheck re-uploads once the
    // range thaws.
    if (src && IsGuestRangeFrozen(base_address + face * face_stride,
                                  uint32_t(face_bytes))) {
      entry.upload_skipped = true;
      continue;
    }
    if (!src) {
      entry.upload_skipped = true;
      static std::set<uint32_t> warned;
      if (warned.size() < 16 && warned.insert(base_address).second) {
        REXGPU_WARN(
            "videonative: [texguard] source not committed, upload skipped "
            "base={:#x} {}x{} fmt={} (zero texture)",
            base_address, width, height, format);
      }
      continue;
    }
    if (zero_depth) {
      std::memset(linear.data(), 0, linear.size());
    } else if (tiled) {
      // Packed mip tail (ring GetPackedMipOffset, mip 0): min dimension
      // <= 16 texels means the base level is packed inside the 32x32 tile,
      // wider-than-tall at y=16, else (incl. square) at x=16, texel
      // units scaled to blocks.
      uint32_t px = 0, py = 0;
      if (REXCVAR_GET(native_video_untile_packed) &&
          std::min(width, height) <= 16) {
        if (width > height) {
          py = 16 / block_dim;
        } else {
          px = 16 / block_dim;
        }
      }
      UntileSurface(linear.data(), src, pitch_blocks, width_blocks,
                    height_blocks, bytes_per_block, px, py);
    } else {
      for (uint32_t y = 0; y < height_blocks; y++) {
        std::memcpy(linear.data() + y * width_blocks * bytes_per_block,
                    src + y * pitch_blocks * bytes_per_block,
                    width_blocks * bytes_per_block);
      }
    }
    // Endianness: 1 = 8-in-16, 2 = 8-in-32.
    if (endian == 1) SwapBytes16(linear.data(), linear.size());
    if (endian == 2) SwapBytes32(linear.data(), linear.size());


      UploadAllocation staging =
        g_upload[g_frame].Allocate((staging_size + 511) & ~511u, 512);
    if (padded_row_bytes == row_bytes) {
      std::memcpy(staging.memory, linear.data(), data_size);
    } else {
      for (uint32_t y = 0; y < height_blocks; y++) {
        std::memcpy(staging.memory + y * padded_row_bytes,
                    linear.data() + y * row_bytes, row_bytes);
      }
    }
    // Footprint extents must be block-aligned for compressed formats (and
    // must match the block-rounded allocation), a 34x34 DXT5 copies as
    // 36x36, which is exactly the 9x9 blocks the guest data provides.
    commandList->copyTextureRegion(
        RenderTextureCopyLocation::Subresource(entry.texture.get(), 0, face),
        RenderTextureCopyLocation::PlacedFootprint(
            staging.buffer, host_format, alloc_w, alloc_h, 1,
            (padded_row_bytes / bytes_per_block) * block_dim, staging.offset));
  }
  commandList->barriers(RenderBarrierStage::GRAPHICS,
                        RenderTextureBarrier(entry.texture.get(),
                                             RenderTextureLayout::SHADER_READ));
  entry.layout = RenderTextureLayout::SHADER_READ;

  entry.descriptor_index = AllocTextureDescriptor();
  if (entry.descriptor_index) {
    g_textureDescriptorSet->setTexture(entry.descriptor_index,
                                       entry.texture.get(),
                                       RenderTextureLayout::SHADER_READ,
                                       entry.view.get());
  }
  entry.valid = true;
  // Record the uploaded content's fingerprint so later content checks can
  // tell whether an entry still matches its guest bytes.
  if (!entry.host_rendered && width <= 256 && height <= 256) {
    const uint8_t* texels = GuestDataPtrFast(base_address, 2048, false, true);
    if (texels) {
      entry.content_hash = XXH3_64bits(texels, 2048);
    }
  }
  {
    auto prior = g_textureCache.find(key);
    if (prior != g_textureCache.end()) RetireEntry(prior->second);
  }
  entry.last_bind_frame = g_frameIndex;
  entry.approx_bytes = entry.texture ? ApproxTexBytes(entry.width, entry.height, entry.host_format) : 0;
  g_texCacheBytes += entry.approx_bytes;
  entry.header_addr = header_addr;
  std::memcpy(entry.header_dw, dw, sizeof(entry.header_dw));
  return &(g_textureCache[key] = std::move(entry));
}

// ---------------------------------------------------------------------------
// Render-target retargeting
// ---------------------------------------------------------------------------

// Registry of surfaces created through the CreateSurface hook (unused when
// the recompiled XDK CreateSurface runs, the common case).
std::unordered_map<uint32_t, SurfaceInfo> g_surfaces;

// Parse a real XDK D3DSurface object (CreateSurface 0x8252BB30, 48 bytes):
// +20 = 0xFFFF0000 stamp, +36 = ((w-1)<<18)|((h-1)<<3), +40 = format dword.
SurfaceInfo ParseSurface(uint32_t surface_obj) {
  SurfaceInfo info;
  if (!surface_obj) return info;
  auto it = g_surfaces.find(surface_obj);
  if (it != g_surfaces.end()) return it->second;
  // Only surface pointers reach SetRenderTarget, so validate by plausibility
  // (the +20 fence field is 0xFFFF0000 only at init, so it cannot be tested).
  const uint32_t size_dword = LoadGuestU32(surface_obj + 36);
  info.width = (size_dword >> 18) + 1;
  info.height = ((size_dword >> 3) & 0x7FFF) + 1;
  info.guest_format = LoadGuestU32(surface_obj + 40);
  // Surface+28 = the RB_COLOR_INFO dword the XDK computed for this surface
  // (D3D_SurfaceInitFields 0x8252B540): bits 0-11 EDRAM base tile, 16-19 hw
  // color format, 20-25 signed color exponent bias (D3DSURFACE_PARAMETERS.
  // ColorExpBias), the ring backend multiplies every PS color export into
  // this RT by 2^bias; ignoring it renders biased RTs 2^bias too dim.
  const uint32_t edram_info = LoadGuestU32(surface_obj + 28);
  info.exp_bias = int32_t(edram_info << 6) >> 26;  // sign-extend bits 20-25
  // RB_COLOR_INFO's low 12 bits are the EDRAM base tile. Left out of the RT
  // key: keying on it separates passes the game expects to share a surface,
  // which drops scene resolves.
  info.edram_base = 0;
  info.valid = (size_dword & 7) == 0 && info.width > 1 &&
               info.width <= 8192 && info.height > 1 && info.height <= 8192;
  return info;
}

RenderFormat TranslateSurfaceFormat(uint32_t guest_format) {
  // D3DFORMAT-packed like fetch dword1: GPUTEXTUREFORMAT in the low 6 bits.
  // Ground truth for the mapping is the XDK D3D_SurfaceInitFields
  // (0x8252B540): the EDRAM hardware format comes from a per-DataFormat
  // table, with 54 normalized to 7 first, but the raw dword is what's
  // stored at surface+40, so both aliases arrive here.
  switch (guest_format & 0x3F) {
    case 6: return RenderFormat::R8G8B8A8_UNORM;  // k_8_8_8_8
    case 7:   // k_2_10_10_10
    case 54:  // k_2_10_10_10_AS_16_16_16_16. 10-bit RTs must land in
      // R10G10B10A2 hosts: as RGBA8 the light accum bands and aliases
      // with same-size 8888 scene RTs (native RTs are keyed by
      // (w, h, host format)).
      return RenderFormat::R10G10B10A2_UNORM;
    case 32:
      // k_16_16_16_16_FLOAT (A16B16G16R16F): the bloom chain. As RGBA8
      // UNORM this clamps HDR highlights at 1.0, and the ping-pong blur
      // pair aliases into one host RT and washes out the composite.
      return RenderFormat::R16G16B16A16_FLOAT;
    default: {
      static std::set<uint32_t> logged;
      if (logged.insert(guest_format).second) {
        REXGPU_WARN(
            "videonative: [rtlog] unhandled surface D3DFMT {:#010x} "
            "(data_format={} signs={:#x} numfmt={}) -> RGBA8 fallback",
            guest_format, guest_format & 0x3F, (guest_format >> 9) & 0xFF,
            (guest_format >> 17) & 1);
      }
      return RenderFormat::R8G8B8A8_UNORM;  // conservative default
    }
  }
}

uint64_t RtKeyOf(uint32_t width, uint32_t height,
                 const RenderFormat formats[kMrtCount], uint32_t edram_base) {
  // Slot formats folded into the low 16 bits (weighted so slot order
  // matters); the practical surface format set here is tiny (8888 /
  // 10.10.10.2 / 16F families), so distinct weights cannot collide.
  const uint64_t fmix =
      (uint64_t(formats[0]) * 131 + uint64_t(formats[1]) * 31 +
       uint64_t(formats[2]) * 17 + uint64_t(formats[3]) * 7) &
      0xFFFFu;
  return (uint64_t(edram_base) << 48) | (uint64_t(width) << 32) |
         (uint64_t(height) << 16) | fmix;
}

// RT-cache VRAM accounting (bounded by the sweep in the stats block).
uint64_t g_rtCacheBytes = 0;
std::deque<std::pair<uint64_t, CachedRenderTarget>> g_retiredRts;

CachedRenderTarget* GetOrCreateRenderTarget(uint32_t width, uint32_t height,
                                            const RenderFormat formats[kMrtCount],
                                            uint32_t edram_base) {
  // Reject implausible dimensions: surface objects read mid-init parse
  // as garbage shapes that each allocate a fresh multi-hundred-MB RT set.
  // The cap sits at the largest legitimate shape (the 2080x2048 grown
  // shadow/bake RT), below the shapes those parses produce.
  if (uint64_t(width) * height > 2080ull * 2048ull) {
    static uint64_t whale_logs = 8;
    if (whale_logs) {
      whale_logs--;
      REXGPU_WARN(
          "videonative: [rtlog] REJECTED implausible RT request {}x{} "
          "(garbage surface parse), draws at this binding are skipped",
          width, height);
    }
    return nullptr;
  }
  const uint64_t key = RtKeyOf(width, height, formats, edram_base);
  auto it = g_renderTargets.find(key);
  if (it != g_renderTargets.end()) {
    it->second.last_bind_frame = g_frameIndex;
    return &it->second;
  }

  CachedRenderTarget rt;
  const uint32_t host_w = HostDim(width);
  const uint32_t host_h = HostDim(height);
  const RenderTexture* colors[kMrtCount];
  for (uint32_t i = 0; i < kMrtCount; i++) {
    rt.color[i] = g_device->createTexture(
        RenderTextureDesc::ColorTarget(host_w, host_h, formats[i]));
    rt.color[i]->setName(fmt::format("rt color{} {}x{} (host {}x{})", i,
                                     width, height, host_w, host_h));
    colors[i] = rt.color[i].get();
  }
  // Depth resource format: D32S8 maps to R32G8X24_TYPELESS in plume's D3D12
  // backend, so the depth plane stays SRV-readable for the shader resolve
  // path; D3D12 forbids an SRV over a typed depth resource. The DSV and
  // every pipeline use kDepthFormat; plume specializes the views.
  rt.depth = g_device->createTexture(
      RenderTextureDesc::DepthTarget(host_w, host_h, kDepthFormat));
  rt.depth->setName(
      fmt::format("rt depth {}x{} (host {}x{})", width, height, host_w, host_h));
  RenderFramebufferDesc desc;
  desc.colorAttachments = colors;
  desc.colorAttachmentsCount = kMrtCount;
  desc.depthAttachment = rt.depth.get();
  rt.framebuffer = g_device->createFramebuffer(desc);
  rt.width = width;
  rt.height = height;
  rt.edram_base = edram_base;
  for (uint32_t i = 0; i < kMrtCount; i++) rt.formats[i] = formats[i];
  rt.last_bind_frame = g_frameIndex;
  rt.approx_bytes = 0;
  for (uint32_t i = 0; i < kMrtCount; i++) {
    rt.approx_bytes += ApproxTexBytes(host_w, host_h, formats[i]);
  }
  rt.approx_bytes += uint64_t(host_w) * host_h * 8;  // depth+stencil planes
  g_rtCacheBytes += rt.approx_bytes;
  CachedRenderTarget* stored = &(g_renderTargets[key] = std::move(rt));
  // Zero-init all attachments: passes that never write RT1-3 still get their
  // content resolved by the game (EDRAM leftovers on hardware), creation
  // garbage (white on AMD) would leak into the composites.
  if (g_frameOpen) {
    auto& cl = g_commandLists[g_frame];
    RenderTextureBarrier init_barriers[kMrtCount + 1];
    for (uint32_t i = 0; i < kMrtCount; i++) {
      init_barriers[i] = RenderTextureBarrier(stored->color[i].get(),
                                              RenderTextureLayout::COLOR_WRITE);
      stored->color_layout[i] = RenderTextureLayout::COLOR_WRITE;
    }
    init_barriers[kMrtCount] = RenderTextureBarrier(
        stored->depth.get(), RenderTextureLayout::DEPTH_WRITE);
    stored->depth_layout = RenderTextureLayout::DEPTH_WRITE;
    cl->barriers(RenderBarrierStage::GRAPHICS, init_barriers, kMrtCount + 1);
    cl->setFramebuffer(stored->framebuffer.get());
    for (uint32_t i = 0; i < kMrtCount; i++) {
      cl->clearColor(i, RenderColor(0.0f, 0.0f, 0.0f, 0.0f), nullptr, 0);
    }
    cl->clearDepthStencil(true, true, 0.0f, 0, nullptr, 0);
    // The caller (ApplyRenderTargetState) re-binds the framebuffer it wants
    // right after; force it to do so even if the key matches.
    g_activeRtKey = ~0ull;
  }
  return stored;
}

uint64_t RtKey(const CachedRenderTarget* rt) {
  return RtKeyOf(rt->width, rt->height, rt->formats, rt->edram_base);
}

// The active pass is depth-only (shadow map): its output is sampled via
// analytic light-space UVs computed by the game's shaders, which assume the
// guest (ring) texture row order, so such passes render Y-flipped (see
// SetupDraw) and the resolved depth texture matches the guest layout.
bool g_depthOnlyTarget = false;

// Live RTT note (see the native_video_sem_rtt cvar). Set by
// NoteRttBegin (rq-ordered with the surface binds that follow it), applied
// per bind only when the bound surface EA equals one of the note's own
// EDRAM surfaces, cleared at frame end. A lingering note is harmless: it
// cannot match a foreign pass's surfaces.
struct SemRttNote {
  bool active = false;
  uint32_t texbase = 0, width = 0, height = 0;
  uint32_t color_surf = 0, depth_surf = 0, dest_tex = 0;
  uint32_t format = 0, msaa = 0, tiling = 0;
};
SemRttNote g_semRtt;
uint64_t g_semRttApplied = 0;  // binds served with note dims/identity

// Points the command list at the framebuffer for the currently bound guest
// surfaces (swapchain when none). Returns false if the frame isn't open.
// Device EA of the draw being set up (SetupDraw entry), the RB shadow
// fold below reads register-level RT identity from the guest device block.
uint32_t g_currentDevice = 0;
// The raw D3D device (the setter's a1, not the draw device), recorded by
// the real-SetRenderTarget hook (its r3), cached for the session.
uint32_t g_rawDeviceEA = 0;

void ApplyRenderTargetState() {
  auto& commandList = g_commandLists[g_frame];
  g_depthOnlyTarget = false;
  SurfaceInfo info = ParseSurface(g_state.color_surface[0]);
  if (!info.valid) {
    // Depth-only pass (no color surface bound, the sun shadow map): bind
    // a host RT sized to the depth surface so z lands in a resolvable
    // depth texture instead of the swapchain depth buffer.
    const SurfaceInfo dinfo = ParseSurface(g_state.depth_surface);
    if (dinfo.valid) {
      info = dinfo;
      info.guest_format = 6;  // color attachment unused (PS-less, mask 0)
      g_depthOnlyTarget = true;
    } else {
      if (g_activeRtKey != 0) {
        // Back to the swapchain framebuffer.
        commandList->setFramebuffer(g_framebuffers[g_backBufferIndex].get());
        g_activeRtKey = 0;
        g_activeRtFormat = kColorFormat;
        for (uint32_t i = 0; i < kMrtCount; i++) {
          g_activeRtFormats[i] = kColorFormat;
        }
        g_activeRtGrown = false;
        g_activeRtHeight = 720;
        g_activeRtWidth = 1280;
        g_activeRtBandWidened = false;
        g_passResolves = 0;
        if (REXCVAR_GET(native_video_bind_scissor_reset) &&
            !g_scissorExplicit) {
          g_state.scissor[0] = 0;
          g_state.scissor[1] = 0;
          g_state.scissor[2] = 1280;
          g_state.scissor[3] = 720;
        }
        g_scissorExplicit = false;
      }
      return;
    }
  }

  // Tiled scene passes bind a tile-height EDRAM surface but draw a
  // full-frame viewport, so the host RT must cover the whole frame for
  // per-tile resolves to cut bands back out. Only genuine tile bands
  // (full tiling width, shorter height) expand, with pitch-padded width
  // tolerance. The extent comes from a live bracket, else from the
  // viewport-derived growth request. When the engine-authoritative RTT
  // note's surface is bound, its identity replaces EDRAM inference.
  bool sem_rtt_applied = false;
  if (REXCVAR_GET(native_video_sem_rtt) && g_semRtt.active &&
      g_semRtt.width && g_semRtt.height) {
    const bool match =
        (g_state.color_surface[0] &&
         g_state.color_surface[0] == g_semRtt.color_surf) ||
        (g_depthOnlyTarget && g_state.depth_surface &&
         g_state.depth_surface == g_semRtt.depth_surf);
    if (match) {
      info.width = g_semRtt.width;
      info.height = g_semRtt.height;
      info.edram_base =
          ((g_semRtt.texbase >> 4) ^ (g_semRtt.texbase >> 16)) & 0xFFFFu;
      sem_rtt_applied = true;
      g_semRttApplied++;
    }
  }
  bool grown = false;
  if (sem_rtt_applied) {
    // Note dims are authoritative: no tiling/viewport growth heuristics.
  } else if (g_tilingWidth && g_tilingHeight && info.width >= g_tilingWidth &&
      info.width <= g_tilingWidth + 64 && info.height < g_tilingHeight) {
    info.height = g_tilingHeight;
  } else if (g_growWidth && g_growHeight && g_growSurface &&
             (g_state.color_surface[0] == g_growSurface ||
              g_state.depth_surface == g_growSurface) &&
             info.width >= g_growWidth && info.width <= g_growWidth + 64 &&
             info.height < g_growHeight) {
    info.height = g_growHeight;
    grown = true;
  }
  // Vertical band surfaces: the guest renders the full frame once into a
  // band-width surface and per-band resolves arrive in full-frame
  // coordinates, so the host RT must span the whole frame. Genuine band
  // shapes only (integer width multiple, tile-aligned height).
  g_activeRtBandWidened = false;
  if (REXCVAR_GET(native_video_band_rt_widen) && !sem_rtt_applied &&
      g_swapChain) {
    // Guest frame dims, never the host window: live bracket extent
    // first, then the sticky last bracket extent, and only then the
    // swapchain (right solely while the window matches the guest frame).
    const uint32_t sw =
        g_tilingWidth ? g_tilingWidth
                      : (g_lastTilingWidth ? g_lastTilingWidth
                                           : g_swapChain->getWidth());
    const uint32_t sh =
        g_tilingHeight ? g_tilingHeight
                       : (g_lastTilingHeight ? g_lastTilingHeight
                                             : g_swapChain->getHeight());
    // Height: >= the swapchain, up to 4096. The banded scene surface
    // parses 320x2048 (tall EDRAM allocation), not just the 320x736
    // align32 shape, so the height cap cannot sit at the swapchain height.
    if (info.width && sw > info.width && sw % info.width == 0 &&
        sw / info.width >= 2 && sw / info.width <= 8 && info.height >= sh &&
        info.height <= 4096) {
      info.width = sw;
      g_activeRtBandWidened = true;
    }
  }
  const RenderFormat format = TranslateSurfaceFormat(info.guest_format);
  // Per-slot attachment formats: slot 0 from the primary surface; slots 1-3
  // from their own bound surfaces when present (Xenos MRT surfaces share
  // dimensions but not formats, LightingPass = 10-bit diffuse + 8888
  // specular). Unbound slots inherit slot 0's format, so resolves of
  // unwritten slots still have backing.
  RenderFormat formats[kMrtCount];
  formats[0] = format;
  for (uint32_t i = 1; i < kMrtCount; i++) {
    formats[i] = format;
    if (!g_depthOnlyTarget && g_state.color_surface[i]) {
      const SurfaceInfo si = ParseSurface(g_state.color_surface[i]);
      if (si.valid) formats[i] = TranslateSurfaceFormat(si.guest_format);
    }
  }
  // Register-level RT identity fold (see the cvar comment): the RB shadow
  // words on the raw device distinguish EDRAM retargets the surface object
  // never shows.
  if (!sem_rtt_applied && REXCVAR_GET(native_video_rt_key_rbinfo) &&
      g_rawDeviceEA) {
    const uint32_t rb0 = LoadGuestU32(g_rawDeviceEA + 4u * 2593u);
    const uint32_t rbd = LoadGuestU32(g_rawDeviceEA + 4u * 2594u);
    const uint32_t salt =
        (rb0 ^ (rb0 >> 16) ^ (rbd << 8) ^ (rbd >> 8)) & 0xFFFFu;
    info.edram_base ^= salt;
  }
  // Fold the other bound surfaces (color 1-3 + depth) into the RT identity
  // so distinct guest passes get distinct host RT sets, matching the ring's
  // (surface, C0-C3, depth) pass key. Only the spare key bits above the
  // 12-bit EDRAM tile index are used (a 4-bit fold separates the handful
  // of concurrent pass tuples).
  if (!sem_rtt_applied && REXCVAR_GET(native_video_pass_separation) &&
      !g_depthOnlyTarget) {
    uint32_t fold = 0;
    for (uint32_t i = 1; i < kMrtCount; i++) {
      if (g_state.color_surface[i]) {
        const SurfaceInfo si = ParseSurface(g_state.color_surface[i]);
        if (si.valid) fold ^= (si.edram_base + 0x9E37u * i);
      }
    }
    const SurfaceInfo di = ParseSurface(g_state.depth_surface);
    if (di.valid) fold ^= di.edram_base * 0x85EBu;
    fold = (fold ^ (fold >> 4) ^ (fold >> 8) ^ (fold >> 12)) & 0xFu;
    info.edram_base ^= fold << 12;
  }
  CachedRenderTarget* rt =
      GetOrCreateRenderTarget(info.width, info.height, formats, info.edram_base);
  if (!rt) {
    // Implausible-dimension rejection: bind the swapchain so state stays
    // sane; draws land there (and the real surface binds moments later).
    if (g_activeRtKey != 0) {
      commandList->setFramebuffer(g_framebuffers[g_backBufferIndex].get());
      g_activeRtKey = 0;
      g_activeRtFormat = kColorFormat;
      for (uint32_t i = 0; i < kMrtCount; i++) g_activeRtFormats[i] = kColorFormat;
      g_activeRtGrown = false;
      g_activeRtHeight = 720;
      g_activeRtWidth = 1280;
      g_passResolves = 0;
    }
    return;
  }
  const uint64_t key = RtKey(rt);
  g_activeRtHeight = info.height;
  g_activeRtWidth = info.width;
  // This runs on every render-target bind; keep it free of memory sweeps.
  if (key == g_activeRtKey) {
    // Same target: nothing rebinds, but a lazy mid-pass resolve may have
    // left surfaces in COPY_SOURCE, repair on demand.
    RepairActiveRtLayouts();
    return;
  }

  uint32_t barrier_count = 0;
  RenderTextureBarrier barriers[kMrtCount + 1];
  for (uint32_t i = 0; i < kMrtCount; i++) {
    if (rt->color_layout[i] != RenderTextureLayout::COLOR_WRITE) {
      barriers[barrier_count++] = RenderTextureBarrier(
          rt->color[i].get(), RenderTextureLayout::COLOR_WRITE);
      rt->color_layout[i] = RenderTextureLayout::COLOR_WRITE;
    }
  }
  if (rt->depth_layout != RenderTextureLayout::DEPTH_WRITE) {
    barriers[barrier_count++] =
        RenderTextureBarrier(rt->depth.get(), RenderTextureLayout::DEPTH_WRITE);
    rt->depth_layout = RenderTextureLayout::DEPTH_WRITE;
  }
  if (barrier_count) {
    commandList->barriers(RenderBarrierStage::GRAPHICS, barriers,
                          barrier_count);
  }
  commandList->setFramebuffer(rt->framebuffer.get());
  g_activeRtKey = key;
  g_activeRt = rt;
  g_activeRtLayoutsDirty = false;  // the block above restored write states
  g_activeRtFormat = format;
  for (uint32_t i = 0; i < kMrtCount; i++) g_activeRtFormats[i] = rt->formats[i];
  g_activeRtGrown = grown;
  g_passResolves = 0;
  // XDK window-scissor semantics: SetRenderTarget resets the scissor to the
  // full target (the ring path reads it back from PA_SC_WINDOW_SCISSOR).
  // Without this, RTs larger than the screen inherit the stale screen
  // scissor: a 2048 shadow map clips to the top-left 1280x720.
  if (REXCVAR_GET(native_video_bind_scissor_reset) && !g_scissorExplicit) {
    g_state.scissor[0] = 0;
    g_state.scissor[1] = 0;
    g_state.scissor[2] = int32_t(info.width);
    g_state.scissor[3] = int32_t(info.height);
  }
  g_scissorExplicit = false;
}

// ---------------------------------------------------------------------------
// Pipeline cache
// ---------------------------------------------------------------------------

struct PipelineKey {
  const void* vs_dxil;
  const void* ps_dxil;
  uint32_t blend_control[kMrtCount];  // RB_BLENDCONTROL0-3 (per-RT)
  uint32_t depth_control;
  uint32_t cull_bits;
  uint32_t topology;
  uint32_t rect_gs;     // RECTLIST expansion GS attached
  uint32_t point_gs;    // POINTLIST expansion GS attached
  uint32_t rt_formats[kMrtCount];  // per-slot color target formats
  uint32_t rt_count;    // color attachment count (4 for cached RTs, 1 swap)
  uint32_t color_mask;  // RB_COLOR_MASK (4 nibbles = RT0-3 write masks)
  // RB_STENCILREFMASK (ref 0:7 / testmask 8:15 / writemask 16:23), plume
  // bakes the ref into the PSO. Zeroed while stencil is disabled so plain
  // draws never fork pipelines on a stale refmask register.
  uint32_t stencil_ref_mask;
  bool operator==(const PipelineKey& o) const {
    return !std::memcmp(this, &o, sizeof(o));
  }
};
struct PipelineKeyHash {
  size_t operator()(const PipelineKey& k) const {
    return size_t(XXH3_64bits(&k, sizeof(k)));
  }
};

struct CachedPipeline {
  std::unique_ptr<RenderShader> vs;
  std::unique_ptr<RenderShader> ps;
  std::unique_ptr<RenderPipeline> pipeline;
};
std::unordered_map<PipelineKey, CachedPipeline, PipelineKeyHash> g_pipelines;

// Rectangle-list expansion GS (created on first rect pipeline; one for all).
std::unique_ptr<RenderShader> g_rectGs;
bool g_rectGsTried = false;

// Point-sprite expansion GS (POINTLIST -> quads sized by the VS oPts export).
std::unique_ptr<RenderShader> g_pointGs;
bool g_pointGsTried = false;

// FPS overlay resources (drawn by DrawFpsOverlay at the tail of the Swap
// composite), created on first visible tick.
std::unique_ptr<RenderTexture> g_fpsTexture;
std::unique_ptr<RenderTextureView> g_fpsView;
uint32_t g_fpsDescriptor = 0;
// Catalog-search overlay strip (same machinery, right-aligned).

// Fullscreen frontbuffer blit (Swap composite), created on first use.
std::unique_ptr<RenderShader> g_blitVs, g_blitPs;
std::unique_ptr<RenderPipeline> g_blitPipeline;
// FPS overlay variant of the blit pipeline with alpha blending (the counter
// has no backdrop box; glyph outlines composite over the scene).
std::unique_ptr<RenderPipeline> g_fpsPipeline;
std::unique_ptr<RenderShader> g_fpsPs;  // alpha-preserving blit PS variant
// Image rect of the last frontbuffer blit (x, y, w, h) in swapchain pixels;
// overlays anchor to the game image, not the window (letterbox bars).
float g_presentRect[4] = {0.0f, 0.0f, 0.0f, 0.0f};
bool g_blitTried = false;
// Frontbuffer-flip fallback note: the game may Swap the texture of the pair
// that has not received a resolve yet, g_lastResolvedBySize (declared with
// the texture cache above) tracks the last resolve per guest size.
uint64_t SizeKeyOf(uint32_t width, uint32_t height) {
  return (uint64_t(width) << 32) | height;
}

RenderBlend TranslateBlend(uint32_t f) {
  switch (f) {
    case 0: return RenderBlend::ZERO;
    case 1: return RenderBlend::ONE;
    case 4: return RenderBlend::SRC_COLOR;
    case 5: return RenderBlend::INV_SRC_COLOR;
    case 6: return RenderBlend::SRC_ALPHA;
    case 7: return RenderBlend::INV_SRC_ALPHA;
    case 8: return RenderBlend::DEST_COLOR;
    case 9: return RenderBlend::INV_DEST_COLOR;
    case 10: return RenderBlend::DEST_ALPHA;
    case 11: return RenderBlend::INV_DEST_ALPHA;
    default: return RenderBlend::ONE;
  }
}
RenderBlendOperation TranslateBlendOp(uint32_t op) {
  switch (op) {
    case 0: return RenderBlendOperation::ADD;
    case 1: return RenderBlendOperation::SUBTRACT;
    case 2: return RenderBlendOperation::MIN;
    case 3: return RenderBlendOperation::MAX;
    case 4: return RenderBlendOperation::REV_SUBTRACT;
    default: return RenderBlendOperation::ADD;
  }
}
RenderComparisonFunction TranslateCompare(uint32_t f) {
  switch (f & 7) {
    case 0: return RenderComparisonFunction::NEVER;
    case 1: return RenderComparisonFunction::LESS;
    case 2: return RenderComparisonFunction::EQUAL;
    case 3: return RenderComparisonFunction::LESS_EQUAL;
    case 4: return RenderComparisonFunction::GREATER;
    case 5: return RenderComparisonFunction::NOT_EQUAL;
    case 6: return RenderComparisonFunction::GREATER_EQUAL;
    default: return RenderComparisonFunction::ALWAYS;
  }
}

// Xenos StencilOp encoding (RB_DEPTHCONTROL op fields).
RenderStencilOp TranslateStencilOp(uint32_t op) {
  switch (op) {
    case 0: return RenderStencilOp::KEEP;
    case 1: return RenderStencilOp::ZERO;
    case 2: return RenderStencilOp::REPLACE;
    case 3: return RenderStencilOp::INCREMENT_AND_CLAMP;
    case 4: return RenderStencilOp::DECREMENT_AND_CLAMP;
    case 5: return RenderStencilOp::INVERT;
    case 6: return RenderStencilOp::INCREMENT_AND_WRAP;
    default: return RenderStencilOp::DECREMENT_AND_WRAP;
  }
}

const CachedPipeline* GetOrCreatePipeline(const ResolvedShader* vs,
                                          const ResolvedShader* ps,
                                          const uint32_t* blend_controls,
                                          uint32_t depth_control,
                                          uint32_t cull_bits,
                                          RenderPrimitiveTopology topology,
                                          bool rect, bool point,
                                          uint32_t color_mask,
                                          uint32_t stencil_ref_mask) {
  if (rect) {
    if (!g_rectGsTried) {
      g_rectGsTried = true;
      const std::vector<uint8_t>& dxil = detail::RectExpandGsDxil();
      if (!dxil.empty()) {
        g_rectGs = g_device->createShader(dxil.data(), dxil.size(), "main",
                                          RenderShaderFormat::DXIL);
      }
    }
    if (!g_rectGs) return nullptr;  // pack predates the rect GS
  }
  if (point) {
    if (!g_pointGsTried) {
      g_pointGsTried = true;
      const std::vector<uint8_t>& dxil = detail::PointExpandGsDxil();
      if (!dxil.empty()) {
        g_pointGs = g_device->createShader(dxil.data(), dxil.size(), "main",
                                           RenderShaderFormat::DXIL);
      }
    }
    // Without the GS (pack predates it), draw plain 1px points rather than
    // skipping, the deferred sprite lights are at least visible as specks.
    if (!g_pointGs) point = false;
  }

  // Prefer the trimmed-signature VS variant: fewer declared
  // outputs = less parameter-cache/attribute cost per vertex on every GPU.
  // The expansion GS links against the full oVar0-15+oPts signature, so GS
  // pipelines keep the full variant. The chosen blob's pointer keys the
  // pipeline, so both variants of one shader cache as distinct entries.
  const bool wants_gs = rect || point;
  const int32_t trim_mode = REXCVAR_GET(native_video_vs_trim);
  const bool trim_enabled =
      trim_mode == 1 || (trim_mode == 2 && g_adapterIsIntel);
  const std::vector<uint8_t>& vs_blob =
      (!wants_gs && trim_enabled && !vs->pack->dxil_trim.empty())
          ? vs->pack->dxil_trim
          : vs->pack->dxil;

  PipelineKey key{};
  key.vs_dxil = vs_blob.data();
  // ps == nullptr: deliberate PS-less depth-only draw (a depth restore:
  // fullscreen rectlist, VS writes oPos.z from the resolved depth texture,
  // no color output).
  key.ps_dxil = ps ? ps->pack->dxil.data() : nullptr;
  for (uint32_t i = 0; i < kMrtCount; i++) {
    key.blend_control[i] = blend_controls[i];
  }
  key.depth_control = depth_control;
  // bit3 = y-flipped pass (winding inverts); bit4 = inverted viewport-Z
  // pass (depth compare inverts, see the revz block in SetupDraw).
  key.cull_bits = cull_bits & 0x1F;
  key.topology = uint32_t(topology);
  key.rect_gs = rect ? 1 : 0;
  key.point_gs = point ? 1 : 0;
  for (uint32_t i = 0; i < kMrtCount; i++) {
    key.rt_formats[i] = uint32_t(g_activeRtFormats[i]);
  }
  key.rt_count = g_activeRtKey ? kMrtCount : 1;
  key.color_mask = color_mask & 0xFFFF;
  key.stencil_ref_mask =
      (depth_control & 1u) ? (stencil_ref_mask & 0xFFFFFFu) : 0u;
  auto it = g_pipelines.find(key);
  if (it != g_pipelines.end()) return &it->second;

  CachedPipeline entry;
  entry.vs = g_device->createShader(vs_blob.data(), vs_blob.size(), "main",
                                    RenderShaderFormat::DXIL);
  if (ps) {
    entry.ps = g_device->createShader(ps->pack->dxil.data(),
                                      ps->pack->dxil.size(), "main",
                                      RenderShaderFormat::DXIL);
  }

  RenderGraphicsPipelineDesc desc;
  desc.pipelineLayout = g_pipelineLayout.get();
  desc.vertexShader = entry.vs.get();
  desc.pixelShader = entry.ps ? entry.ps.get() : nullptr;
  if (rect) desc.geometryShader = g_rectGs.get();
  if (point) desc.geometryShader = g_pointGs.get();
  for (uint32_t i = 0; i < key.rt_count; i++) {
    desc.renderTargetFormat[i] = g_activeRtFormats[i];
  }
  desc.renderTargetCount = key.rt_count;
  desc.depthTargetFormat = kDepthFormat;
  desc.primitiveTopology = topology;

  // RB_DEPTHCONTROL: bit1 z_enable, bit2 z_write, func bits 4..6.
  desc.depthEnabled = (depth_control >> 1) & 1;
  desc.depthWriteEnabled = (depth_control >> 2) & 1;
  {
    // Inverted-viewport-Z pass (cull_bits bit4): the stored depth values
    // are flipped (z' = 1-z), so LESS<->GREATER and LEQUAL<->GEQUAL swap.
    uint32_t func = (depth_control >> 4) & 7;
    if (cull_bits & 16) {
      static const uint32_t kZInv[8] = {0, 4, 2, 6, 1, 5, 3, 7};
      func = kZInv[func];
    }
    desc.depthFunction = TranslateCompare(func);
  }
  desc.depthClipEnabled = true;

  // RB_DEPTHCONTROL stencil: bit0 enable, bit7 backface_enable, FRONT
  // func 8:10 / fail 11:13 / zpass 14:16 / zfail 17:19, BACK 20:22 /
  // 23:25 / 26:28 / 29:31. The AE mirror pass: silhouette punch = func
  // Always + zpass REPLACE ref 0xFF (dctl 0x708731), reflected avatar =
  // func LESSEQUAL ref 0xFF (dctl 0x708337), stencil-masked to the glass.
  desc.stencilEnabled = (depth_control & 1u) != 0;
  if (desc.stencilEnabled) {
    desc.stencilReference = key.stencil_ref_mask & 0xFF;
    desc.stencilReadMask = (key.stencil_ref_mask >> 8) & 0xFF;
    desc.stencilWriteMask = (key.stencil_ref_mask >> 16) & 0xFF;
    const auto face = [&](uint32_t shift) {
      RenderStencilFaceDesc f;
      f.compareFunction = TranslateCompare((depth_control >> shift) & 7);
      f.failOp = TranslateStencilOp((depth_control >> (shift + 3)) & 7);
      f.passOp = TranslateStencilOp((depth_control >> (shift + 6)) & 7);
      f.depthFailOp = TranslateStencilOp((depth_control >> (shift + 9)) & 7);
      return f;
    };
    desc.stencilFrontFace = face(8);
    desc.stencilBackFace =
        (depth_control & 0x80u) ? face(20) : desc.stencilFrontFace;
  }

  // PA_SU_SC_MODE_CNTL: bit0 cull front, bit1 cull back, bit2 face.
  // bit3 (runtime-defined): Y-flipped pass (depth-only shadow rendering);
  // flipping the projection inverts the winding, so the front-face sense
  // flips with it.
  // Rectangles are never culled on Xenos (and the GS reorders vertices).
  if (rect) {
    desc.cullMode = RenderCullMode::NONE;
  } else if (cull_bits & 1) {
    desc.cullMode = RenderCullMode::FRONT;
  } else if (cull_bits & 2) {
    desc.cullMode = RenderCullMode::BACK;
  } else {
    desc.cullMode = RenderCullMode::NONE;
  }
  // front_counter_clockwise = (face == 0), matching the ring path.
  const bool front_ccw = ((cull_bits & 4) == 0) ^ ((cull_bits & 8) != 0);
  desc.frontFace = front_ccw ? RenderFrontFace::COUNTER_CLOCKWISE
                             : RenderFrontFace::CLOCKWISE;

  // RB_BLENDCONTROL0-3 (one per RT): srcblend 0:4, comb 5:7, dstblend 8:12;
  // alpha at 16+. RB_COLOR_MASK nibble i = RT i's RGBA write mask (the
  // z-prepass draws with mask 0, writing their color would paint flat
  // prepass output over the scene).
  for (uint32_t i = 0; i < key.rt_count; i++) {
    const uint32_t bc = blend_controls[i];
    RenderBlendDesc rt_blend;
    const uint32_t src = bc & 0x1F;
    const uint32_t op = (bc >> 5) & 0x7;
    const uint32_t dst = (bc >> 8) & 0x1F;
    const uint32_t src_a = (bc >> 16) & 0x1F;
    const uint32_t op_a = (bc >> 21) & 0x7;
    const uint32_t dst_a = (bc >> 24) & 0x1F;
    rt_blend.blendEnabled = !(src == 1 && dst == 0 && op == 0 && src_a == 1 &&
                              dst_a == 0 && op_a == 0);
    rt_blend.srcBlend = TranslateBlend(src);
    rt_blend.dstBlend = TranslateBlend(dst);
    rt_blend.blendOp = TranslateBlendOp(op);
    rt_blend.srcBlendAlpha = TranslateBlend(src_a);
    rt_blend.dstBlendAlpha = TranslateBlend(dst_a);
    rt_blend.blendOpAlpha = TranslateBlendOp(op_a);
    rt_blend.renderTargetWriteMask = uint8_t((color_mask >> (4 * i)) & 0xF);
    if (!ps) {
      // Depth-only: no PS means undefined color output, mask all writes.
      rt_blend.blendEnabled = false;
      rt_blend.renderTargetWriteMask = 0;
    }
    desc.renderTargetBlend[i] = rt_blend;
  }

  // PSO-compile cost accounting: driver shader compilation is CPU-side work
  // with the GPU idle.
  {
    const auto pso_t0 = std::chrono::steady_clock::now();
    entry.pipeline = g_device->createGraphicsPipeline(desc);
    g_psoCreateMs += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - pso_t0)
                         .count();
    g_psoCreateCount++;
  }
  // A PS reading an interpolant its VS never writes fails linkage against
  // the trimmed signature. Retry once with the full-signature VS and log
  // the pair: it is a translator-contract violation.
  if (!entry.pipeline && vs_blob.data() == vs->pack->dxil_trim.data() &&
      !vs->pack->dxil.empty()) {
    entry.vs = g_device->createShader(vs->pack->dxil.data(),
                                      vs->pack->dxil.size(), "main",
                                      RenderShaderFormat::DXIL);
    desc.vertexShader = entry.vs.get();
    entry.pipeline = g_device->createGraphicsPipeline(desc);
    static uint64_t trim_retry_log = 8;
    if (trim_retry_log) {
      trim_retry_log--;
      REXGPU_WARN(
          "videonative: trimmed VS {:016X} failed linkage with ps={:016X}, "
          "retried with the full signature ({})",
          vs->ucode_hash, ps ? ps->ucode_hash : 0,
          entry.pipeline ? "ok" : "STILL FAILED");
    }
  }
  // A failed PSO would otherwise be silent (cached null, draws skip via
  // g_skipPipeline only). Name the pair so the failing shader/feature can
  // be identified from the log.
  if (!entry.pipeline) {
    static uint64_t pso_fail_log = 16;
    if (pso_fail_log) {
      pso_fail_log--;
      REXGPU_WARN(
          "videonative: pipeline creation FAILED vs={:016X} ps={:016X} "
          "topo={} rect={} point={} mask={:#x}, draws with this pair skip",
          vs ? vs->ucode_hash : 0, ps ? ps->ucode_hash : 0,
          uint32_t(topology), rect, point, color_mask);
    }
  }
  return &(g_pipelines[key] = std::move(entry));
}

// ---------------------------------------------------------------------------
// Frame flow
// ---------------------------------------------------------------------------

void CreateFramebuffers() {
  g_framebuffers.clear();
  // A minimized window reports a zero client rect, and a zero-extent depth
  // texture removes the device. Stay down; the per-frame resize in
  // EnsureFrameOpen brings the chain up once the window has a size.
  if (g_swapChain->getWidth() == 0 || g_swapChain->getHeight() == 0) {
    g_swapChainValid = false;
    return;
  }
  RenderTextureDesc depthDesc = RenderTextureDesc::DepthTarget(
      g_swapChain->getWidth(), g_swapChain->getHeight(), kDepthFormat);
  g_depthTexture = g_device->createTexture(depthDesc);
  for (uint32_t i = 0; i < g_swapChain->getTextureCount(); i++) {
    const RenderTexture* color = g_swapChain->getTexture(i);
    const RenderTexture* depth = g_depthTexture.get();
    RenderFramebufferDesc desc;
    desc.colorAttachments = &color;
    desc.colorAttachmentsCount = 1;
    desc.depthAttachment = depth;
    g_framebuffers.push_back(g_device->createFramebuffer(desc));
  }
}

}  // namespace

// Deferred writeback delivery hook for the frame-open slot-reuse wait
// (definition lives with the writeback machinery below).
static void DeliverSlotWritebacks();

bool Init() {
  g_device = detail::Device();
  g_queue = detail::Queue();
  if (!g_device || !g_queue) return false;

  for (auto& cl : g_commandLists) cl = g_queue->createCommandList();
  for (auto& f : g_commandFences) f = g_device->createCommandFence();
  for (auto& s : g_acquireSemaphores) s = g_device->createCommandSemaphore();
  for (auto& s : g_renderSemaphores) s = g_device->createCommandSemaphore();

  void* hwnd = nullptr;
  if (auto* runtime = rex::Runtime::instance()) {
    if (auto* window = runtime->display_window()) {
      hwnd = window->GetNativeWindowHandle();
    }
  }
  g_inspectHwnd = hwnd;
  if (hwnd) {
    REXGPU_INFO("videonative: creating swapchain on hwnd {}",
                reinterpret_cast<uintptr_t>(hwnd));
    RenderSwapChainDesc swapDesc(static_cast<RenderWindow>(hwnd), kColorFormat,
                                 kSwapChainBufferCount);
    g_swapChain = g_queue->createSwapChain(swapDesc);
    // D3D/DXGI teardown must not run from CRT exit: the swapchain
    // destructor, invoked via the static's atexit slot, deadlocks in DXGI's
    // critical section under overlay hooks. This handler registers after the
    // graphics globals exist, so it runs before their destructors (LIFO) and
    // defuses them by leaking; the process is exiting and the OS reclaims
    // everything.
    static bool exit_guard_registered = false;
    if (!exit_guard_registered) {
      exit_guard_registered = true;
      std::atexit([] {
        g_swapChain.release();
        for (auto& cl : g_commandLists) cl.release();
        for (auto& f : g_commandFences) f.release();
        for (auto& s2 : g_acquireSemaphores) s2.release();
        for (auto& s2 : g_renderSemaphores) s2.release();
        // Segment pools hold the same D3D object classes, same leak-defusal.
        for (auto& sl : g_segSpareLists) sl.release();
        for (auto& sf : g_segSpareFences) sf.release();
        for (auto& fr : g_segInFlight) {
          for (auto& seg : fr) {
            seg.list.release();
            seg.fence.release();
          }
        }
      });
    }
    if (g_swapChain) {
      g_swapChainValid = !g_swapChain->needsResize();
      if (g_swapChainValid) CreateFramebuffers();
    }
  }
  if (!g_swapChain) {
    REXGPU_WARN("videonative: no swapchain (hwnd {}), headless", hwnd);
  }

  // Pipeline layout per the native shader contract. b1 (float constants) and
  // b4 (descriptor indices) are per-stage: the pack VS and PS each read their
  // own buffer at the same register, so two root params alias b1/b4 with
  // vertex/pixel visibility, same scheme as the ring backend's bindless root
  // signature. b0/b2/b3 are stage-shared.
  RenderPipelineLayoutBuilder layoutBuilder;
  layoutBuilder.begin(false, true);
  layoutBuilder.addRootDescriptor(  // kRootB0System
      0, 0, RenderRootDescriptorType::CONSTANT_BUFFER);
  layoutBuilder.addRootDescriptor(  // kRootB1FloatsVs
      1, 0, RenderRootDescriptorType::CONSTANT_BUFFER,
      RenderShaderVisibility::VERTEX);
  layoutBuilder.addRootDescriptor(  // kRootB1FloatsPs
      1, 0, RenderRootDescriptorType::CONSTANT_BUFFER,
      RenderShaderVisibility::PIXEL);
  layoutBuilder.addRootDescriptor(  // kRootB2BoolLoop
      2, 0, RenderRootDescriptorType::CONSTANT_BUFFER);
  layoutBuilder.addRootDescriptor(  // kRootB3Fetch
      3, 0, RenderRootDescriptorType::CONSTANT_BUFFER);
  layoutBuilder.addRootDescriptor(  // kRootB4IndicesVs
      4, 0, RenderRootDescriptorType::CONSTANT_BUFFER,
      RenderShaderVisibility::VERTEX);
  layoutBuilder.addRootDescriptor(  // kRootB4IndicesPs
      4, 0, RenderRootDescriptorType::CONSTANT_BUFFER,
      RenderShaderVisibility::PIXEL);
  layoutBuilder.addRootDescriptor(  // kRootSharedMemSrv
      0, 0, RenderRootDescriptorType::SHADER_RESOURCE);
  layoutBuilder.addRootDescriptor(  // kRootDummyUav
      0, 0, RenderRootDescriptorType::UNORDERED_ACCESS);

  RenderDescriptorSetBuilder samplerSetBuilder;
  samplerSetBuilder.begin();
  samplerSetBuilder.addSampler(0, kSamplerDescriptorCount);
  samplerSetBuilder.end(true, kSamplerDescriptorCount);
  layoutBuilder.addDescriptorSet(samplerSetBuilder);  // set 0 -> space0

  RenderDescriptorSetBuilder textureSetBuilder;
  textureSetBuilder.begin();
  textureSetBuilder.addTexture(0, kTextureDescriptorCount);
  textureSetBuilder.end(true, kTextureDescriptorCount);
  layoutBuilder.addDescriptorSet(textureSetBuilder);  // set 1 -> space1 (2D)
  layoutBuilder.addDescriptorSet(textureSetBuilder);  // set 2 -> space2 (3D)
  layoutBuilder.addDescriptorSet(textureSetBuilder);  // set 3 -> space3 (cube)
  layoutBuilder.end();
  g_pipelineLayout = layoutBuilder.create(g_device);

  g_samplerDescriptorSet = samplerSetBuilder.create(g_device);
  g_textureDescriptorSet = textureSetBuilder.create(g_device);

  // Samplers: 0 = linear wrap, 1 = linear clamp, 2 = point wrap, 3 = point clamp.
  for (uint32_t i = 0; i < 4; i++) {
    RenderSamplerDesc samplerDesc;
    const bool linear = i < 2;
    const bool wrap = (i & 1) == 0;
    samplerDesc.minFilter = linear ? RenderFilter::LINEAR : RenderFilter::NEAREST;
    samplerDesc.magFilter = samplerDesc.minFilter;
    samplerDesc.addressU = wrap ? RenderTextureAddressMode::WRAP
                                : RenderTextureAddressMode::CLAMP;
    samplerDesc.addressV = samplerDesc.addressU;
    samplerDesc.addressW = samplerDesc.addressU;
    auto sampler = g_device->createSampler(samplerDesc);
    g_samplerDescriptorSet->setSampler(i, sampler.get());
    g_samplers.push_back(std::move(sampler));
  }
  // Samplers 4-12: linear with per-axis address modes, index = 4 + u*3 + v,
  // mode order {WRAP, MIRROR, CLAMP}. Xenos fetch constants carry clamp_x
  // and clamp_y independently (scrolling textures bind clampX=CLAMP with
  // clampY=WRAP), so both axes cannot collapse to clamp_x.
  {
    static const RenderTextureAddressMode kAxisModes[3] = {
        RenderTextureAddressMode::WRAP, RenderTextureAddressMode::MIRROR,
        RenderTextureAddressMode::CLAMP};
    for (uint32_t u = 0; u < 3; u++) {
      for (uint32_t v = 0; v < 3; v++) {
        RenderSamplerDesc samplerDesc;
        samplerDesc.minFilter = RenderFilter::LINEAR;
        samplerDesc.magFilter = RenderFilter::LINEAR;
        samplerDesc.addressU = kAxisModes[u];
        samplerDesc.addressV = kAxisModes[v];
        samplerDesc.addressW = kAxisModes[v];
        auto sampler = g_device->createSampler(samplerDesc);
        g_samplerDescriptorSet->setSampler(4 + u * 3 + v, sampler.get());
        g_samplers.push_back(std::move(sampler));
      }
    }
  }
  // Slots 13-15: fill with sampler 0 so every slot the shader-side clamp
  // (xe_sampler_index, min 15) can produce is a valid descriptor. The
  // sampler semantics layer re-purposes them as write-once dynamic slots for
  // specs the static set cannot express (aniso/border/mirror-once/mixed).
  for (uint32_t i = 13; i < kSamplerDescriptorCount; i++) {
    g_samplerDescriptorSet->setSampler(i, g_samplers.front().get());
  }
  semantics::InitSamplerArm(g_device, g_samplerDescriptorSet.get());

  // Blank texture in every descriptor so unset slots are valid.
  g_blankTexture = g_device->createTexture(
      RenderTextureDesc::Texture2D(1, 1, 1, RenderFormat::R8G8B8A8_UNORM));
  g_blankTextureView = g_blankTexture->createTextureView(
      RenderTextureViewDesc::Texture2DArray(RenderFormat::R8G8B8A8_UNORM));
  for (uint32_t i = 0; i < kTextureDescriptorCount; i++) {
    g_textureDescriptorSet->setTexture(i, g_blankTexture.get(),
                                       RenderTextureLayout::SHADER_READ,
                                       g_blankTextureView.get());
  }
  // Dimension-matched blanks (Intel TDR fix): TextureCube/Texture3D
  // declarations must never dereference a 2D-array descriptor, reserved
  // slots serve a real cube / real 3D resource instead.
  g_blankTextureCube = g_device->createTexture(RenderTextureDesc::Texture(
      RenderTextureDimension::TEXTURE_2D, 1, 1, 1, 1, 6,
      RenderFormat::R8G8B8A8_UNORM));
  g_blankTextureCubeView = g_blankTextureCube->createTextureView(
      RenderTextureViewDesc::TextureCube(RenderFormat::R8G8B8A8_UNORM));
  g_textureDescriptorSet->setTexture(kBlankCubeDescriptor,
                                     g_blankTextureCube.get(),
                                     RenderTextureLayout::SHADER_READ,
                                     g_blankTextureCubeView.get());
  g_blankTexture3d = g_device->createTexture(RenderTextureDesc::Texture3D(
      1, 1, 1, 1, RenderFormat::R8G8B8A8_UNORM));
  g_blankTexture3dView = g_blankTexture3d->createTextureView(
      RenderTextureViewDesc::Texture3D(RenderFormat::R8G8B8A8_UNORM));
  g_textureDescriptorSet->setTexture(kBlank3dDescriptor,
                                     g_blankTexture3d.get(),
                                     RenderTextureLayout::SHADER_READ,
                                     g_blankTexture3dView.get());

  // Zero-initialize the blank textures: committed resources are not
  // guaranteed cleared, so an unwritten "blank" could sample as garbage.
  {
    // Row pitch must be 256-byte aligned for buffer->texture copies: use a
    // 64-texel row (64 * 4 bytes) even though only texel 0 matters.
    auto staging = g_device->createBuffer(RenderBufferDesc::UploadBuffer(256));
    std::memset(staging->map(), 0, 256);
    staging->unmap();
    auto& initList = g_commandLists[0];
    initList->begin();
    const RenderTextureBarrier to_copy[] = {
        RenderTextureBarrier(g_blankTexture.get(),
                             RenderTextureLayout::COPY_DEST),
        RenderTextureBarrier(g_blankTextureCube.get(),
                             RenderTextureLayout::COPY_DEST),
        RenderTextureBarrier(g_blankTexture3d.get(),
                             RenderTextureLayout::COPY_DEST)};
    initList->barriers(RenderBarrierStage::COPY, nullptr, 0, to_copy, 3);
    initList->copyTextureRegion(
        RenderTextureCopyLocation::Subresource(g_blankTexture.get(), 0),
        RenderTextureCopyLocation::PlacedFootprint(
            staging.get(), RenderFormat::R8G8B8A8_UNORM, 1, 1, 1, 64, 0));
    for (uint32_t face = 0; face < 6; face++) {
      initList->copyTextureRegion(
          RenderTextureCopyLocation::Subresource(g_blankTextureCube.get(),
                                                 face),
          RenderTextureCopyLocation::PlacedFootprint(
              staging.get(), RenderFormat::R8G8B8A8_UNORM, 1, 1, 1, 64, 0));
    }
    initList->copyTextureRegion(
        RenderTextureCopyLocation::Subresource(g_blankTexture3d.get(), 0),
        RenderTextureCopyLocation::PlacedFootprint(
            staging.get(), RenderFormat::R8G8B8A8_UNORM, 1, 1, 1, 64, 0));
    const RenderTextureBarrier to_read[] = {
        RenderTextureBarrier(g_blankTexture.get(),
                             RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(g_blankTextureCube.get(),
                             RenderTextureLayout::SHADER_READ),
        RenderTextureBarrier(g_blankTexture3d.get(),
                             RenderTextureLayout::SHADER_READ)};
    initList->barriers(RenderBarrierStage::GRAPHICS, nullptr, 0, to_read, 3);
    initList->end();
    const RenderCommandList* initLists[] = {initList.get()};
    g_queue->executeCommandLists(initLists, 1, nullptr, 0, nullptr, 0,
                                 g_commandFences[0].get());
    g_queue->waitForCommandFence(g_commandFences[0].get());
  }

  g_dummyUav = g_device->createBuffer(RenderBufferDesc::DefaultBuffer(
      256, RenderBufferFlag::UNORDERED_ACCESS));
  REXGPU_INFO("videonative: renderer initialized (swapchain {}x{})",
              g_swapChain ? g_swapChain->getWidth() : 0,
              g_swapChain ? g_swapChain->getHeight() : 0);
  return true;
}

void Shutdown() {
  rq::StopAndJoin();
  for (uint32_t i = 0; i < kNumFrames; i++) {
    if (g_commandListPending[i]) {
      g_queue->waitForCommandFence(g_commandFences[i].get());
      g_commandListPending[i] = false;
    }
  }
  g_pipelines.clear();
  g_rectGs.reset();
  g_rectGsTried = false;
  g_pointGs.reset();
  g_pointGsTried = false;
  g_blitPipeline.reset();
  g_fpsPipeline.reset();
  g_fpsPs.reset();
  g_blitVs.reset();
  g_blitPs.reset();
  g_blitTried = false;
  g_fpsTexture.reset();
  g_fpsView.reset();
  g_fpsDescriptor = 0;
  g_renderTargets.clear();
  g_lastResolvedBySize.clear();  // points into g_textureCache
  g_textureCache.clear();
  for (auto& slot : g_depthStaging) {
    slot.buffer.reset();
    slot.size = 0;
  }
  g_depthStagingRot = 0;
  g_depthStageMemo = {};  // never leave the memo pointing at a freed buffer
  g_framebuffers.clear();
  g_depthTexture.reset();
  g_swapChain.reset();
}

void EnsureFrameOpen() {
  if (g_frameOpen || !g_swapChain) return;
  if (g_swapChain->needsResize() || !g_swapChainValid) {
    for (uint32_t i = 0; i < kNumFrames; i++) {
      if (g_commandListPending[i]) {
        g_queue->waitForCommandFence(g_commandFences[i].get());
        g_commandListPending[i] = false;
      }
    }
    g_framebuffers.clear();
    g_swapChainValid = g_swapChain->resize();
    if (g_swapChainValid) CreateFramebuffers();
    // A resize that keeps failing is permanent black (device removed / TDR),
    // so log it instead of retrying silently. Reason details land in
    // plume_present_dbg.txt (hr + DRED breadcrumbs).
    if (!g_swapChainValid) {
      static uint64_t resize_fail_log = 8;
      if (resize_fail_log) {
        resize_fail_log--;
        REXGPU_WARN(
            "videonative: swapchain resize FAILED, rendering is DOWN "
            "(device removed/TDR likely; see plume_present_dbg.txt)");
      }
    }
  }
  if (!g_swapChainValid) return;

  if (g_commandListPending[g_frame]) {
    g_queue->waitForCommandFence(g_commandFences[g_frame].get());
    g_commandListPending[g_frame] = false;
  }
  // Deferred writeback delivery: this slot's fence
  // was just waited for reuse, its bucket's snapshot copies are executed.
  // Deliver now (pure map+memcpy) and recycle the slot's arena segment.
  DeliverSlotWritebacks();
  // Recycle this slot's early-submitted segments (their fences are FIFO
  // before the present fence just waited; the extra wait is a no-op on a
  // presented frame and covers the swapchain-loss path).
  if (!g_segInFlight[g_frame].empty()) {
    g_queue->waitForCommandFence(g_segInFlight[g_frame].back().fence.get());
    for (auto& seg : g_segInFlight[g_frame]) {
      g_segSpareLists.push_back(std::move(seg.list));
      g_segSpareFences.push_back(std::move(seg.fence));
    }
    g_segInFlight[g_frame].clear();
  }
  // Persistent-region reclamation: the region has no per-entry free, so a
  // long session eventually fills it and every bind falls back to per-frame
  // uploads. On region-full, flush everything at this idle point: wait the
  // other parity's fence too (all prior submissions then complete on the
  // single queue, so nothing references the region), clear, re-warm.
  // Rate-limited, since re-warming costs a burst of uploads.
  if (g_vbPersistFlushWanted) {
    static uint64_t last_flush_tick = 0;
    const uint64_t now = rex::chrono::Clock::QueryHostTickCount();
    const uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
    if (!last_flush_tick || now - last_flush_tick > 60 * freq) {
      last_flush_tick = now;
      for (uint32_t i = 0; i < kNumFrames; i++) {
        if (g_commandListPending[i]) {
          g_queue->waitForCommandFence(g_commandFences[i].get());
          g_commandListPending[i] = false;
        }
      }
      g_vbPersistCache.clear();
      g_vbPersistByGuest.clear();
      g_vbPersistWatermark = 0;
      g_vbPersistOrphans = 0;
    }
    g_vbPersistFlushWanted = false;
  }
  g_upload[g_frame].Reset();
  g_vertexRing[g_frame].Reset(kVbPersistCap);
  g_vbUploadCache.clear();

  if (!g_swapChain->acquireTexture(g_acquireSemaphores[g_frame].get(),
                                   &g_backBufferIndex)) {
    g_swapChainValid = false;
    return;
  }
  // The first submission of this frame (an early segment or the present
  // execute) consumes the acquire semaphore.
  g_segAcquirePending[g_frame] = true;

  auto& commandList = g_commandLists[g_frame];
  commandList->begin();
  const RenderTextureBarrier frameBarriers[] = {
      RenderTextureBarrier(g_swapChain->getTexture(g_backBufferIndex),
                           RenderTextureLayout::COLOR_WRITE),
      RenderTextureBarrier(g_depthTexture.get(),
                           RenderTextureLayout::DEPTH_WRITE)};
  commandList->barriers(RenderBarrierStage::GRAPHICS, frameBarriers, 2);
  commandList->setFramebuffer(g_framebuffers[g_backBufferIndex].get());
  commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.05f, 1.0f));
  commandList->clearDepthStencil(true, true, 1.0f, 0);
  commandList->setGraphicsPipelineLayout(g_pipelineLayout.get());
  commandList->setGraphicsDescriptorSet(g_samplerDescriptorSet.get(), 0);
  commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 1);
  commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 2);
  commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 3);
  g_frameOpen = true;
  g_drawsThisFrame = 0;
  // The frame opens on the swapchain framebuffer.
  g_activeRtKey = 0;
  g_activeRtFormat = kColorFormat;
}

// Resolve write-back: the 360 resolve writes guest RAM and some engines
// CPU-read the resolved bytes. Queue each color resolve; at the guest's
// fence wait, read the rect back and write it to guest memory with the
// dest header's pitch/tiling/endian. Entries never cross frames.
struct ResolveWriteback {
  RenderFormat host_format;
  uint32_t base_address;   // guest bytes (header dw1 & ~0xFFF)
  uint32_t pitch_texels;   // header dw0 pitch (32-texel units) * 32
  uint32_t endian;         // header dw1 bits 6-7 (2 = 8-in-32)
  bool tiled;              // header dw0 bit 31
  uint32_t tex_w, tex_h;   // guest texture dims
  uint32_t x, y, w, h;     // resolved dest rect (guest texels)
  // Content is snapshotted into the readback arena at resolve time; the
  // flush only delivers bytes. A deferred flush can never copy a later
  // frame's RT content into an earlier frame's ping-pong buffer.
  uint32_t snap_offset;    // byte offset into g_wbSnapBuffer
  uint32_t snap_pitch;     // 256-aligned row pitch in the arena
  // Shadow depth resolves: host D32 float converted to guest D24S8
  // (depth<<8, stencil 0) at delivery.
  bool depth_convert;
  // A dest texture with Xenos gamma signs stores PWL-encoded bytes in
  // guest RAM; the host renders linear, so encode at delivery.
  bool gamma_encode;
  // Liveness guard: deferred delivery lands about two frames after the
  // resolve, and the guest may have freed the dest texture by then
  // (menu exit destroys plate textures; the recycled heap block gets new
  // tenants). Delivery re-reads this header: if dw1 no longer names
  // base_address, the guest freed or rebuilt the texture and the record
  // is dropped.
  uint32_t header_addr = 0;
  // Allocation identity of the dest pages at resolve time: delivery
  // re-queries the physical heap, and a dest whose allocation changed was
  // freed (and possibly reused), so the record is dropped.
  uint32_t alloc_base = 0;
  uint32_t alloc_size = 0;
  uint32_t alloc_state = 0;
};
static bool QueryDestAllocation(uint32_t guest_addr, uint32_t* base, uint32_t* size,
                                uint32_t* state) {
  auto* mem = rex::system::kernel_memory();
  auto* heap = mem ? mem->LookupHeap(guest_addr) : nullptr;
  rex::memory::HeapAllocationInfo info{};
  if (!heap || !heap->QueryRegionInfo(guest_addr, &info)) return false;
  *base = info.allocation_base;
  *size = info.allocation_size;
  *state = info.state;
  return true;
}

// Xenos piecewise-linear gamma encode (linear -> PWL), 8-bit LUT.
const uint8_t* PwlGammaLut() {
  static uint8_t lut[256];
  static bool init = false;
  if (!init) {
    init = true;
    for (int i = 0; i < 256; i++) {
      const float x = float(i) / 255.0f;
      const float y = x < 0.0625f ? x * 4.0f
                      : x < 0.125f ? x * 2.0f + 0.125f
                      : x < 0.5f   ? x + 0.25f
                                   : x * 0.5f + 0.5f;
      lut[i] = uint8_t(y * 255.0f + 0.5f);
    }
  }
  return lut;
}

uint32_t PwlEncodeRgba(uint32_t rgba) {
  const uint8_t* lut = PwlGammaLut();
  return (rgba & 0xFF000000u) | (uint32_t(lut[(rgba >> 16) & 0xFF]) << 16) |
         (uint32_t(lut[(rgba >> 8) & 0xFF]) << 8) | lut[rgba & 0xFF];
}
std::vector<ResolveWriteback> g_resolveWritebacks;
std::atomic<uint32_t> g_resolveWritebacksPending{0};
// Snapshot arena: grown on demand; grow flushes first so live records
// never reference a replaced buffer.
std::unique_ptr<RenderBuffer> g_wbSnapBuffer;
uint32_t g_wbSnapCap = 0;
// Deferred delivery: records ride per-slot buckets and deliver at the
// slot's reuse fence wait, which the frame ring pays anyway. The arena is
// segmented per slot so snapshot bytes survive until delivery; the sync
// flush still delivers everything immediately.
std::vector<ResolveWriteback> g_wbReadyBuckets[kNumFrames];
uint32_t g_wbSnapUsedSlot[kNumFrames] = {};

// Pending-record bases for the guest LockRect sync hook: a lock on a
// texture whose pages have an undelivered writeback flushes inline before
// the guest's CPU read. Approximate set, false positives only cost an
// extra flush; cleared whenever the pending count hits zero.
static std::mutex g_wbBaseMutex;
static std::vector<uint32_t> g_wbPendingBases;

bool WritebackPendingForBase(uint32_t base) {
  if (g_resolveWritebacksPending.load(std::memory_order_acquire) == 0) {
    return false;
  }
  const uint32_t b = base & 0x1FFFF000u;  // page + physical-alias mask
  std::lock_guard<std::mutex> lock(g_wbBaseMutex);
  for (uint32_t v : g_wbPendingBases) {
    if (v == b) return true;
  }
  return false;
}

static void NoteWritebackBase(uint32_t base_address) {
  const uint32_t b = base_address & 0x1FFFF000u;
  std::lock_guard<std::mutex> lock(g_wbBaseMutex);
  if (std::find(g_wbPendingBases.begin(), g_wbPendingBases.end(), b) ==
      g_wbPendingBases.end()) {
    if (g_wbPendingBases.size() >= 64) {
      g_wbPendingBases.erase(g_wbPendingBases.begin());
    }
    g_wbPendingBases.push_back(b);
  }
}

static void UpdateWritebackPending() {
  uint32_t n = uint32_t(g_resolveWritebacks.size());
  for (uint32_t i = 0; i < kNumFrames; i++) {
    n += uint32_t(g_wbReadyBuckets[i].size());
  }
  g_resolveWritebacksPending.store(n, std::memory_order_release);
  if (n == 0) {
    std::lock_guard<std::mutex> lock(g_wbBaseMutex);
    g_wbPendingBases.clear();
  }
}

uint32_t ResolveWritebacksPending() {
  return g_resolveWritebacksPending.load(std::memory_order_acquire);
}

bool ResolveWritebackEnabled() {
  return REXCVAR_GET(native_video_resolve_writeback);
}

bool WritebackSyncEnabled() {
  return REXCVAR_GET(native_video_writeback_sync);
}

void NoteRawDevice(uint32_t device_ea) {
  if (!device_ea || g_rawDeviceEA == device_ea) return;
  g_rawDeviceEA = device_ea;
}

// Delivery core: map the snapshot arena and write every record's texels
// into guest memory. Caller guarantees the arena's recorded GPU copies
// have executed (frame fence passed). Never touches command lists.
static void DeliverWritebackRecords(std::vector<ResolveWriteback>& records);
static void DeliverAllWritebacks();
void DrainDeferredCopies();  // defined below
void QueueGuestTextureInvalidate(uint32_t guest_address, uint32_t size);

// Per-frame wait attribution: the mid-frame sync flush, the post-present
// writeback fence wait, and the present call itself.
static double g_wbInlineMsFrame = 0.0;
static double g_wbPostMsLast = 0.0;
static double g_presentMsLast = 0.0;
static uint32_t g_wbRecsLast = 0;
namespace {
struct WbInlineTimer {
  uint64_t t0 = rex::chrono::Clock::QueryHostTickCount();
  ~WbInlineTimer() {
    g_wbInlineMsFrame +=
        double(rex::chrono::Clock::QueryHostTickCount() - t0) * 1000.0 /
        double(rex::chrono::Clock::QueryHostTickFrequency());
  }
};
}  // namespace

// Mid-frame reopen: restore the framework binds after a segment submit
// (identical to the inline flush's reopen; per-draw state re-emits).
static void ReopenCurrentListMidFrame() {
  auto& commandList = g_commandLists[g_frame];
  commandList->begin();
  commandList->setGraphicsPipelineLayout(g_pipelineLayout.get());
  commandList->setGraphicsDescriptorSet(g_samplerDescriptorSet.get(), 0);
  commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 1);
  commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 2);
  commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 3);
  if (g_activeRtKey) {
    auto it = g_renderTargets.find(g_activeRtKey);
    commandList->setFramebuffer(it != g_renderTargets.end()
                                    ? it->second.framebuffer.get()
                                    : g_framebuffers[g_backBufferIndex].get());
  } else {
    commandList->setFramebuffer(g_framebuffers[g_backBufferIndex].get());
  }
}

// Submit the current segment (no wait) and continue recording on a fresh
// list: the GPU executes in parallel with further CPU recording, so a later
// sync flush waits only the in-flight tail instead of executing the whole
// frame from scratch.
void EarlySubmitSegment() {
  if (!g_frameOpen || !g_device || !g_swapChainValid) return;
  if (g_segInFlight[g_frame].size() >= 12) return;  // runaway guard
  // Deferred VB/IB copies must land before the GPU consumes them (same
  // contract as the inline flush).
  DrainDeferredCopies();
  auto& commandList = g_commandLists[g_frame];
  commandList->end();
  std::unique_ptr<RenderCommandFence> fence;
  if (!g_segSpareFences.empty()) {
    fence = std::move(g_segSpareFences.back());
    g_segSpareFences.pop_back();
  } else {
    fence = g_device->createCommandFence();
  }
  const RenderCommandList* lists[] = {commandList.get()};
  RenderCommandSemaphore* waitSems[] = {g_acquireSemaphores[g_frame].get()};
  const bool take_acquire = g_segAcquirePending[g_frame];
  g_queue->executeCommandLists(lists, 1, take_acquire ? waitSems : nullptr,
                               take_acquire ? 1u : 0u, nullptr, 0,
                               fence.get());
  g_segAcquirePending[g_frame] = false;
  SegInFlight seg;
  seg.list = std::move(commandList);
  seg.fence = std::move(fence);
  g_segInFlight[g_frame].push_back(std::move(seg));
  if (!g_segSpareLists.empty()) {
    commandList = std::move(g_segSpareLists.back());
    g_segSpareLists.pop_back();
  } else {
    commandList = g_queue->createCommandList();
  }
  ReopenCurrentListMidFrame();
}

void FlushResolveWritebacksInline() {
  WbInlineTimer wb_inline_timer;
  if (g_resolveWritebacksPending.load(std::memory_order_acquire) == 0) {
    return;
  }
  // Fast path: with segments in flight, the flush only needs the small
  // tail since the last early submit; submit it and wait the newest
  // segment fence (single-queue FIFO: everything earlier, including every
  // snapshot copy, has executed). The full path below handles frames with
  // no early submits.
  if (REXCVAR_GET(native_video_early_submit) && g_frameOpen &&
      g_frameIndex > 0 && g_device && g_swapChainValid &&
      g_wbSnapBuffer && !Scaled()) {
    EarlySubmitSegment();
    if (!g_segInFlight[g_frame].empty()) {
      g_queue->waitForCommandFence(g_segInFlight[g_frame].back().fence.get());
    }
    DeliverAllWritebacks();
    return;
  }
  auto drop_all = [] {
    g_resolveWritebacks.clear();
    for (uint32_t i = 0; i < kNumFrames; i++) {
      g_wbReadyBuckets[i].clear();
      g_wbSnapUsedSlot[i] = 0;
    }
    g_resolveWritebacksPending.store(0, std::memory_order_release);
  };
  if (!g_device || !g_swapChainValid) {
    drop_all();
    return;
  }
  // Mid-frame delivery is required: the guest CPU-copies resolve pages
  // into its display arrays right after the resolve call, and bakes run
  // once, so present-time delivery is a frame too late.
  if (Scaled()) {
    static uint64_t scale_logs = 2;
    if (scale_logs) {
      scale_logs--;
      REXGPU_WARN(
          "videonative: [writeback] skipped under resolution_scale "
          "(host rects are scaled; guest bytes would be wrong)");
    }
    drop_all();
    return;
  }
  auto& commandList = g_commandLists[g_frame];
  const bool frame_open = g_frameOpen;
  if (!frame_open) {
    if (g_commandListPending[g_frame]) {
      g_queue->waitForCommandFence(g_commandFences[g_frame].get());
      g_commandListPending[g_frame] = false;
    }
    commandList->begin();
  }
  // Records already carry their content in the snapshot arena (copied at
  // resolve time); this flush only needs those recorded copies executed,
  // then maps and delivers.
  if (!g_wbSnapBuffer) {
    drop_all();
    if (!frame_open) {
      commandList->end();
      const RenderCommandList* empty_lists[] = {commandList.get()};
      g_queue->executeCommandLists(empty_lists, 1, nullptr, 0, nullptr, 0,
                                   g_commandFences[g_frame].get());
      g_queue->waitForCommandFence(g_commandFences[g_frame].get());
    }
    return;
  }
  // This mid-frame execute runs every draw recorded so far, and the
  // deferred VB/IB copies normally drain at present; drain them now or
  // those draws run on unwritten ring memory.
  DrainDeferredCopies();
  commandList->end();
  const RenderCommandList* lists[] = {commandList.get()};
  g_queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0,
                               g_commandFences[g_frame].get());
  g_queue->waitForCommandFence(g_commandFences[g_frame].get());
  g_commandListPending[g_frame] = false;
  DeliverAllWritebacks();

  if (frame_open) {
    // Reopen mid-frame: restore the framework binds; per-draw state
    // (pipeline/viewport/scissor/root descriptors) re-emits every draw.
    commandList->begin();
    commandList->setGraphicsPipelineLayout(g_pipelineLayout.get());
    commandList->setGraphicsDescriptorSet(g_samplerDescriptorSet.get(), 0);
    commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 1);
    commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 2);
    commandList->setGraphicsDescriptorSet(g_textureDescriptorSet.get(), 3);
    if (g_activeRtKey) {
      auto it = g_renderTargets.find(g_activeRtKey);
      commandList->setFramebuffer(
          it != g_renderTargets.end()
              ? it->second.framebuffer.get()
              : g_framebuffers[g_backBufferIndex].get());
    } else {
      commandList->setFramebuffer(g_framebuffers[g_backBufferIndex].get());
    }
  }
}

static void DeliverWritebackRecords(std::vector<ResolveWriteback>& records) {
  if (records.empty() || !g_wbSnapBuffer) {
    records.clear();
    UpdateWritebackPending();
    return;
  }
  const uint8_t* bytes = static_cast<const uint8_t*>(g_wbSnapBuffer->map());
  if (!bytes) {
    // Device pressure / removed-device: map can fail, so never dereference
    // the result.
    static uint64_t mapfail_logs = 4;
    if (mapfail_logs) {
      mapfail_logs--;
      REXGPU_WARN("videonative: [writeback] arena map FAILED, {} records "
                  "dropped this frame",
                  records.size());
    }
    records.clear();
    UpdateWritebackPending();
    return;
  }
  uint32_t written = 0;
  for (size_t i = 0; i < records.size(); i++) {
    const ResolveWriteback& wb = records[i];
    // Header staleness is not a drop gate: plate resolve-dest headers are
    // transient (rebuilt between the resolve and the deferred delivery), so
    // a mismatch does not mean the destination died.
    // Allocation liveness (the real drop gate, see ResolveWriteback): the
    // dest's physical allocation must still be the one resolved into.
    if (wb.alloc_size) {
      uint32_t now_base = 0, now_size = 0, now_state = 0;
      const bool ok = QueryDestAllocation(wb.base_address, &now_base, &now_size, &now_state);
      if (!ok || now_base != wb.alloc_base || now_size != wb.alloc_size ||
          now_state != wb.alloc_state) {
        static uint64_t freed_drop_logs = 24;
        if (freed_drop_logs) {
          freed_drop_logs--;
          REXGPU_WARN(
              "videonative: [writeback] DROPPED {}x{} rect for base={:#x}: dest "
              "allocation changed ({:#x}/{:#x}/{} -> {:#x}/{:#x}/{}), freed or reused",
              wb.w, wb.h, wb.base_address, wb.alloc_base, wb.alloc_size, wb.alloc_state,
              now_base, now_size, now_state);
        }
        continue;
      }
    }
    const uint32_t pitch = wb.snap_pitch;
    const uint32_t pitch_texels =
        wb.pitch_texels ? wb.pitch_texels : ((wb.tex_w + 31u) & ~31u);
    // Direct translation, not GuestPtr (that consults the worker's
    // snapshot windows; this is a write to live guest memory).
    uint8_t* guest =
        rex::system::kernel_memory()->TranslateVirtual<uint8_t*>(
            wb.base_address);
    if (!guest) continue;
    static uint64_t endian_logs = 2;
    if (wb.endian != 2 && wb.endian != 0 && endian_logs) {
      endian_logs--;
      REXGPU_WARN("videonative: [writeback] endian {} unsupported (raw copy)",
                  wb.endian);
    }
    // Guest channel order: the guest's native 8888 memory layout is reversed
    // relative to host RGBA, and every sampler carries the (2,1,0,3) dest
    // swizzle to compensate (the "identity = ZYXW-order" convention in the
    // swizzle table). The resolve must write that native order, so color
    // texels swap R<->B before the endian encode.
    const auto rb_swap = [](uint32_t v) -> uint32_t {
      return (v & 0xFF00FF00u) | ((v >> 16) & 0xFFu) | ((v & 0xFFu) << 16);
    };
    // Content-change probe: sample 16 texels of the mid row before and after
    // the write. A delivery that changed the pages must force the texture
    // over them to re-upload (the budget-limited heal scan starves under
    // scroll churn); steady-state canvases delivering the same bytes stay
    // cached, with no retire/create churn.
    uint32_t pre_sample[16];
    uint8_t* sample_ptrs[16];
    uint32_t sample_count = 0;
    {
      const uint32_t mid_gy = wb.y + wb.h / 2;
      const uint32_t step = wb.w > 16 ? wb.w / 16 : 1;
      const uint32_t outer =
          wb.tiled ? TiledOffset2DOuter(mid_gy, pitch_texels, 2) : 0;
      for (uint32_t t = 0; t < wb.w && sample_count < 16; t += step) {
        uint8_t* p =
            wb.tiled
                ? guest + TiledOffset2DInner(wb.x + t, mid_gy, 2, outer)
                : guest + (uint64_t(mid_gy) * pitch_texels + wb.x + t) * 4;
        sample_ptrs[sample_count] = p;
        std::memcpy(&pre_sample[sample_count], p, 4);
        sample_count++;
      }
    }
    for (uint32_t row = 0; row < wb.h; row++) {
      const uint32_t* src = reinterpret_cast<const uint32_t*>(
          bytes + wb.snap_offset + row * pitch);
      const uint32_t gy = wb.y + row;
      if (!wb.tiled) {
        uint32_t* dst = reinterpret_cast<uint32_t*>(
            guest + (uint64_t(gy) * pitch_texels + wb.x) * 4);
        if (wb.depth_convert) {
          for (uint32_t t = 0; t < wb.w; t++) {
            float z;
            std::memcpy(&z, &src[t], 4);
            z = z < 0.f ? 0.f : (z > 1.f ? 1.f : z);
            const uint32_t d = uint32_t(z * 16777215.0f + 0.5f) << 8;
            dst[t] = wb.endian == 2 ? __builtin_bswap32(d) : d;
          }
        } else if (wb.gamma_encode) {
          for (uint32_t t = 0; t < wb.w; t++) {
            const uint32_t e = rb_swap(PwlEncodeRgba(src[t]));
            dst[t] = wb.endian == 2 ? __builtin_bswap32(e) : e;
          }
        } else if (wb.endian == 2) {
          for (uint32_t t = 0; t < wb.w; t++)
            dst[t] = __builtin_bswap32(rb_swap(src[t]));
        } else {
          for (uint32_t t = 0; t < wb.w; t++) dst[t] = rb_swap(src[t]);
        }
      } else {
        const uint32_t outer = TiledOffset2DOuter(gy, pitch_texels, 2);
        for (uint32_t t = 0; t < wb.w; t++) {
          const uint32_t off = TiledOffset2DInner(wb.x + t, gy, 2, outer);
          uint32_t raw = src[t];
          if (wb.depth_convert) {
            float z;
            std::memcpy(&z, &raw, 4);
            z = z < 0.f ? 0.f : (z > 1.f ? 1.f : z);
            raw = uint32_t(z * 16777215.0f + 0.5f) << 8;
          } else if (wb.gamma_encode) {
            raw = rb_swap(PwlEncodeRgba(raw));
          } else {
            raw = rb_swap(raw);
          }
          const uint32_t v = wb.endian == 2 ? __builtin_bswap32(raw) : raw;
          std::memcpy(guest + off, &v, 4);
        }
      }
    }
    // Delivery makes the guest pages authoritative for this base: the guest
    // composes tiles in these pages after reading them, so the host resolve
    // product and the pages diverge from here on. Retire the redirect
    // registration so a sampler bind falls through to the guest-upload path;
    // the next resolve of this dest re-registers.
    {
      const ResolveDestKeys rk = ResolveDestRegKeys(wb.base_address);
      g_resolvedBaseToKey.erase(rk.truth);
      if (rk.diverged) {
        g_resolvedBaseToKey.erase(rk.formula);
      }
    }
    {
      bool content_changed = false;
      for (uint32_t s = 0; s < sample_count; s++) {
        uint32_t post;
        std::memcpy(&post, sample_ptrs[s], 4);
        if (post != pre_sample[s]) {
          content_changed = true;
          break;
        }
      }
      if (content_changed) {
        const size_t span = size_t(pitch_texels) * wb.tex_h * 4;
        QueueGuestTextureInvalidate(
            wb.base_address, uint32_t(std::min<size_t>(span, 0x400000)));
      }
    }
    written++;
  }
  g_wbSnapBuffer->unmap();
  records.clear();
  UpdateWritebackPending();
}

// Deliver everything now, ready buckets (oldest first) then the current
// frame's records, and free every arena segment. Caller guarantees all
// recorded GPU copies executed (single queue: waiting the just-submitted
// list's fence implies every earlier submission completed).
static void DeliverAllWritebacks() {
  // Oldest bucket first so a newer resolve of the same dest wins. After a
  // present g_frame has already advanced, so bucket[g_frame] (i=kNumFrames)
  // is the oldest and bucket[g_frame+1] (i=kNumFrames-1... 1) newer.
  for (uint32_t i = kNumFrames; i >= 1; i--) {
    DeliverWritebackRecords(g_wbReadyBuckets[(g_frame + i) % kNumFrames]);
  }
  DeliverWritebackRecords(g_resolveWritebacks);
  for (uint32_t i = 0; i < kNumFrames; i++) {
    g_wbSnapUsedSlot[i] = 0;
  }
}

// Frame-open delivery: called right after the slot's reuse fence wait,
// this slot's bucket copies are GPU-complete, so the delivery is a pure
// map+memcpy. A frame that recorded writebacks but
// never presented (swapchain loss) leaves records in g_resolveWritebacks
// whose GPU copies may never have executed, drop them before their
// segment recycles.
static void DeliverSlotWritebacks() {
  if (!g_resolveWritebacks.empty()) {
    g_resolveWritebacks.clear();
    UpdateWritebackPending();
  }
  if (!g_wbReadyBuckets[g_frame].empty()) {
    DeliverWritebackRecords(g_wbReadyBuckets[g_frame]);
  }
  g_wbSnapUsedSlot[g_frame] = 0;
}

// Deferred guest->upload copies: queued at draw time, executed
// at EndFrameAndPresent before submission, so the GPU sees the guest
// buffers' end-of-frame state exactly like real hardware (which fetches at
// execution time), not the mid-frame state at Run-call time. swap: 0 = raw
// memcpy, 2 = bswap16 words, 4 = bswap32 words.
struct DeferredCopy {
  const void* src;
  void* dst;
  uint32_t size;
  uint8_t swap;
};
std::vector<DeferredCopy> g_deferredCopies;

void DrainDeferredCopies() {
  for (const DeferredCopy& c : g_deferredCopies) {
    if (c.swap == 2) {
      const uint16_t* src = reinterpret_cast<const uint16_t*>(c.src);
      uint16_t* dst = reinterpret_cast<uint16_t*>(c.dst);
      for (uint32_t i = 0; i < c.size / 2; i++) {
        dst[i] = __builtin_bswap16(src[i]);
      }
    } else if (c.swap == 4) {
      const uint32_t* src = reinterpret_cast<const uint32_t*>(c.src);
      uint32_t* dst = reinterpret_cast<uint32_t*>(c.dst);
      for (uint32_t i = 0; i < c.size / 4; i++) {
        dst[i] = __builtin_bswap32(src[i]);
      }
    } else {
      std::memcpy(c.dst, c.src, c.size);
    }
  }
  g_deferredCopies.clear();
}

void EndFrameAndPresent() {
  if (rq::Active()) { rq::EnqEndFrameAndPresent(); return; }
  // Execute the deferred guest->upload copies first: every draw recorded
  // this frame now samples the guest VB/IB state as of end-of-frame, the
  // moment real hardware would fetch it.
  DrainDeferredCopies();
  // Boot-race compensation (see the cvar): hold early presents so async
  // loads land before the guest's one-shot bakes consume their sources.
  if (g_frameIndex <
          uint64_t(std::max(0, REXCVAR_GET(native_video_boot_delay_frames))) &&
      REXCVAR_GET(native_video_boot_frame_delay_ms) > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        REXCVAR_GET(native_video_boot_frame_delay_ms)));
  }
  g_frameIndex++;
  g_healsThisFrame = 0;  // per-frame heal budget (see tex_heal_frame_budget)
  // Per-tile-loop clear suppression is frame-scoped: without this reset
  // the next frame's legitimate pass-opening clear (same RT, no key change)
  // would still be suppressed by last frame's resolve count. (Known
  // limitation: a second shadow window on the same surface within one
  // frame would have its opening clear suppressed; revisit if a title
  // shows two.)
  g_passResolves = 0;
  // Destroy retired (re-uploaded) texture entries once every frame that
  // could reference the old host texture has left the GPU. The window is
  // frame-count based, not fence-based: a slow GPU with deep queueing can
  // lag more than N frames behind this counter and sample a destroyed
  // texture (Intel iGPUs page-fault on it).
  g_rendererThreadHash.store(ThisThreadHash(), std::memory_order_release);
  while (!g_retiredTextures.empty() &&
         g_frameIndex - g_retiredTextures.front().first >
             uint64_t(REXCVAR_GET(native_video_texture_retire_frames))) {
    // Fence-safe point (window >> kNumFrames): scrub the dead entry's
    // descriptor slot back to blank before the texture is destroyed, so no
    // slot ever points at a freed resource.
    CachedTexture& dead = g_retiredTextures.front().second;
    if (dead.texture) {
      g_retiredPtrs.erase(dead.texture.get());
      for (auto& kv : g_textureCache) {
        if (kv.second.texture.get() == dead.texture.get()) {
          static uint64_t dbl2_logs = 32;
          if (dbl2_logs) {
            dbl2_logs--;
            REXGPU_WARN("videonative: [texdouble] drain: a live cache entry still owns "
                        "retired texture base={:#x} {}x{} (duplicate owner leaked)",
                        dead.header_dw[1] & 0xFFFFF000u, dead.width, dead.height);
          }
          (void)dead.texture.release();
          (void)dead.view.release();
          break;
        }
      }
    }
    if (dead.descriptor_index > 0 &&
        dead.descriptor_index < kFirstReservedDescriptor) {
      // Scrub with the blank of the dead texture's own view dimension: a
      // stale reference through the matching declaration stays defined.
      const bool was_cube = dead.view_dim == 3;
      g_textureDescriptorSet->setTexture(
          dead.descriptor_index,
          was_cube ? g_blankTextureCube.get() : g_blankTexture.get(),
          RenderTextureLayout::SHADER_READ,
          was_cube ? g_blankTextureCubeView.get() : g_blankTextureView.get());
      // Fence-safe: nothing in flight references the slot anymore, recycle
      // it (see AllocTextureDescriptor).
      g_freeDescriptors.push_back(dead.descriptor_index);
    }
    g_retiredTextures.pop_front();
  }
  // Watermark eviction (every 64 frames): only above
  // native_video_tex_cache_max do the coldest entries retire through the
  // drain. Idle eviction is deliberately avoided: slow-cycling bound sets
  // would re-upload forever, and upload-once textures whose guest source
  // was freed come back as garbage.
  const int32_t cache_max = REXCVAR_GET(native_video_tex_cache_max);
  const uint64_t bytes_max =
      uint64_t(std::max(0, REXCVAR_GET(native_video_tex_cache_mb))) << 20;
  const bool over_count =
      cache_max > 0 && g_textureCache.size() > size_t(cache_max);
  const bool over_bytes = bytes_max && g_texCacheBytes > bytes_max;
  if ((over_count || over_bytes) && (g_frameIndex & 63) == 0) {
    const uint64_t min_age =
        uint64_t(std::max(0, REXCVAR_GET(native_video_tex_evict_frames)));
    // Eviction safety: host-rendered entries re-resolve correctly. A
    // guest-uploaded entry is safe only when unreachable, meaning its live
    // header no longer matches the creation snapshot, so its key (the header
    // hash) can never be looked up again. Evicting a reachable upload risks
    // re-uploading recycled guest memory.
    const auto evict_safe = [min_age](const CachedTexture& e) {
      if (e.host_rendered) {
        // One-time bakes (backdrop sheets resolved once at boot or level
        // load) are sampled forever; their content is host-side only and
        // irrecoverable after eviction. Only entries in the active resolve
        // chain (re-resolved recently: scene/post/frontbuffer targets)
        // re-materialize and may go.
        if (!e.last_resolve_frame) return true;  // never held content
        return g_frameIndex - e.last_resolve_frame <= min_age * 2;
      }
      if (!e.header_addr) return true;  // untracked -> unreachable
      for (int i = 0; i < 6; i++) {
        if (LoadGuestU32(e.header_addr + i * 4) != e.header_dw[i]) {
          return true;  // header dead/recycled: entry unreachable
        }
      }
      return false;
    };
    // Collect eviction candidates (old enough + safe), coldest first.
    std::vector<std::pair<uint64_t, uint64_t>> cand;  // (last_bind, key)
    cand.reserve(g_textureCache.size());
    for (auto& [k, e] : g_textureCache) {
      if (e.texture && g_frameIndex - e.last_bind_frame > min_age &&
          evict_safe(e)) {
        cand.emplace_back(e.last_bind_frame, k);
      }
    }
    // Trim to ~87% of whichever watermark tripped, coldest-first, so
    // sweeps stay rare.
    const size_t target_count =
        cache_max > 0 ? size_t(cache_max) - size_t(cache_max) / 8
                      : size_t(-1);
    const uint64_t target_bytes =
        bytes_max ? bytes_max - bytes_max / 8 : ~0ull;
    size_t take = 0;
    if (!cand.empty()) {
      std::sort(cand.begin(), cand.end());
      size_t evicted_i = 0;
      for (size_t i = 0; i < cand.size(); i++) {
        if (g_textureCache.size() - evicted_i <= target_count &&
            g_texCacheBytes <= target_bytes) {
          break;
        }
        auto tit = g_textureCache.find(cand[i].second);
        if (tit == g_textureCache.end()) continue;
        CachedTexture& e = tit->second;
        // Detach every raw-pointer reference before the entry moves.
        for (auto sit = g_lastResolvedBySize.begin();
             sit != g_lastResolvedBySize.end();) {
          if (sit->second == &e) {
            sit = g_lastResolvedBySize.erase(sit);
          } else {
            ++sit;
          }
        }
        for (auto rit = g_resolvedBaseToKey.begin();
             rit != g_resolvedBaseToKey.end();) {
          if (rit->second == tit->first) {
            rit = g_resolvedBaseToKey.erase(rit);
          } else {
            ++rit;
          }
        }
        RetireEntry(e);
        g_textureCache.erase(tit);
        evicted_i++;
      }
      take = evicted_i;
    }
    if (!take) {
      static uint64_t hot_warn = 4;
      if (hot_warn) {
        hot_warn--;
        REXGPU_WARN(
            "videonative: [texevict] cache over watermark ({} entries, "
            "{} MB) but no entries older than the min age, pressure ahead",
            g_textureCache.size(), g_texCacheBytes >> 20);
      }
    }
  }
  // Retired RT sets destroy at the same fence-safe point as textures.
  while (!g_retiredRts.empty() &&
         g_frameIndex - g_retiredRts.front().first >
             uint64_t(REXCVAR_GET(native_video_texture_retire_frames))) {
    g_retiredRts.pop_front();
  }
  // RT-cache watermark sweep: g_renderTargets accumulated per unique
  // (dims, format-set) forever, with garbage-parse whales rejected above,
  // legitimate diversity still grows across a long session; bound it like
  // the texture cache (LRU, min-age, fence-safe; the active and last-drawn
  // sets are never touched).
  {
    const uint64_t rt_bytes_max =
        uint64_t(std::max(0, REXCVAR_GET(native_video_rt_cache_mb))) << 20;
    if (rt_bytes_max && g_rtCacheBytes > rt_bytes_max &&
        (g_frameIndex & 63) == 0) {
      const uint64_t min_age =
          uint64_t(std::max(0, REXCVAR_GET(native_video_tex_evict_frames)));
      std::vector<std::pair<uint64_t, uint64_t>> cand;
      for (auto& [k, e] : g_renderTargets) {
        if (k != g_activeRtKey && k != g_lastDrawnRtKey &&
            g_frameIndex - e.last_bind_frame > min_age) {
          cand.emplace_back(e.last_bind_frame, k);
        }
      }
      std::sort(cand.begin(), cand.end());
      const uint64_t target = rt_bytes_max - rt_bytes_max / 8;
      for (size_t i = 0; i < cand.size() && g_rtCacheBytes > target; i++) {
        auto rit = g_renderTargets.find(cand[i].second);
        if (rit == g_renderTargets.end()) continue;
        g_rtCacheBytes -=
            std::min<uint64_t>(g_rtCacheBytes, rit->second.approx_bytes);
        g_retiredRts.emplace_back(g_frameIndex, std::move(rit->second));
        g_renderTargets.erase(rit);
      }
    }
  }
  if (!g_swapChain) return;
  if (!g_frameOpen) {
    // Nothing was drawn this frame; open it just to clear and present.
    EnsureFrameOpen();
    if (!g_frameOpen) return;
  }

  auto& commandList = g_commandLists[g_frame];

  commandList->barriers(
      RenderBarrierStage::GRAPHICS,
      RenderTextureBarrier(g_swapChain->getTexture(g_backBufferIndex),
                           RenderTextureLayout::PRESENT));
  commandList->end();

  const RenderCommandList* commandLists[] = {commandList.get()};
  RenderCommandSemaphore* waitSemaphores[] = {
      g_acquireSemaphores[g_frame].get()};
  RenderCommandSemaphore* signalSemaphores[] = {
      g_renderSemaphores[g_frame].get()};
  // An early-submitted segment may already have consumed the acquire
  // semaphore this frame; waiting it twice deadlocks the queue.
  const bool take_acquire = g_segAcquirePending[g_frame];
  g_segAcquirePending[g_frame] = false;
  g_queue->executeCommandLists(commandLists, 1,
                               take_acquire ? waitSemaphores : nullptr,
                               take_acquire ? 1u : 0u, signalSemaphores, 1,
                               g_commandFences[g_frame].get());
  g_commandListPending[g_frame] = true;
  // Host-side frame cap (framerate_limit), same contract as the ring CP's
  // swap throttle: hold the present until the minimum interval since the
  // previous one; time from the target so wake-up jitter does not lower the
  // average; resync after stalls instead of bursting.
  {
    const int32_t limit = REXCVAR_GET(framerate_limit);
    static uint64_t throttle_last_tick = 0;
    if (limit > 0) {
      const uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
      const uint64_t interval = std::max<uint64_t>(1, freq / uint64_t(limit));
      uint64_t now = rex::chrono::Clock::QueryHostTickCount();
      if (throttle_last_tick) {
        const uint64_t target = throttle_last_tick + interval;
        while (now < target) {
          const uint64_t remaining_us = (target - now) * 1000000 / freq;
          if (remaining_us > 1000) {
            SleepPrecise(std::chrono::microseconds(remaining_us - 500));
          } else {
            rex::thread::MaybeYield();
          }
          now = rex::chrono::Clock::QueryHostTickCount();
        }
        throttle_last_tick = (now - target < interval) ? target : now;
      } else {
        throttle_last_tick = now;
      }
    } else {
      throttle_last_tick = 0;
    }
  }
  // Forward the vsync cvar (the ring path honors it; the native swapchain
  // otherwise stays at its sync-interval-1 default, capping the renderer at
  // the monitor refresh no matter what framerate_limit asks for). vsync off
  // presents with tearing allowed; framerate_limit then paces exactly.
  g_swapChain->setVsyncEnabled(QueryVsync());
  {
    const uint64_t pt0 = rex::chrono::Clock::QueryHostTickCount();
    g_swapChainValid =
        g_swapChain->present(g_backBufferIndex, signalSemaphores, 1);
    g_presentMsLast =
        double(rex::chrono::Clock::QueryHostTickCount() - pt0) * 1000.0 /
        double(rex::chrono::Clock::QueryHostTickFrequency());
  }
  // Writeback hand-off, no fence wait: the frame's records move to this
  // slot's ready bucket and deliver at the slot's reuse wait, which the
  // frame ring pays anyway. First-copy-correct delivery remains the sync
  // path's job.
  if (!g_resolveWritebacks.empty()) {
    g_wbRecsLast = uint32_t(g_resolveWritebacks.size());
    auto& bucket = g_wbReadyBuckets[g_frame];
    if (bucket.empty()) {
      bucket.swap(g_resolveWritebacks);
    } else {
      bucket.insert(bucket.end(), g_resolveWritebacks.begin(),
                    g_resolveWritebacks.end());
      g_resolveWritebacks.clear();
    }
    UpdateWritebackPending();
    g_wbPostMsLast = 0.0;
  } else {
    g_wbPostMsLast = 0.0;
    g_wbRecsLast = 0;
  }
  static uint64_t present_failures = 0;
  if (!g_swapChainValid && (present_failures++ % 256) == 0) {
    REXGPU_WARN(
        "videonative: present FAILED ({} so far, hr/DRED breadcrumbs in "
        "plume_present_dbg.txt next to the exe)",
        present_failures);
  }
  g_frame = (g_frame + 1) % kNumFrames;
  g_frameOpen = false;

}

// ---------------------------------------------------------------------------
// State capture
// ---------------------------------------------------------------------------

void SetViewport(uint32_t x, uint32_t y, uint32_t w, uint32_t h, float min_z,
                 float max_z) {
  if (rq::Active()) { rq::EnqSetViewport(x, y, w, h, min_z, max_z); return; }
  g_state.viewport_x = x;
  g_state.viewport_y = y;
  g_state.viewport_w = w ? w : 1;
  g_state.viewport_h = h ? h : 1;
  g_state.viewport_min_z = min_z;
  g_state.viewport_max_z = max_z;
}
// Guest texture destroyed: drop its cached upload, or the game's reuse
// of the header serves stale content later. Only guest-uploaded entries
// drop; host-rendered entries are host-owned and the redirect needs them.
static void InvalidateSmallAlphaTexturesNow();

// Locked guest texture object -> texel base + generous byte span. The object
// carries its 6-dword fetch constant at +0x1C (dw1 = base, dw2 = dims); the
// D3D unlock path is shared with vertex buffers, whose fetch constant has
// type 3, so gate on the texture fetch type (2). Span = w*h*4 (32bpp upper
// bound, clamped) so mip-tail-adjusted (+0x1000) entries overlap as well.
static bool LockedTextureRange(uint32_t obj, uint32_t* base, uint32_t* span,
                               uint32_t* dw0_out = nullptr) {
  const auto rd = [](uint32_t ea) -> uint32_t {
    const uint8_t* p = GuestPtr(ea);
    if (!p || !GuestRangeReadable(p, 4)) return 0;
    return __builtin_bswap32(*reinterpret_cast<const uint32_t*>(p));
  };
  const uint32_t dw0 = rd(obj + 0x1C);
  if (dw0_out) *dw0_out = dw0;
  if ((dw0 & 3u) != 2u) return false;  // not a texture fetch constant
  const uint32_t b = rd(obj + 0x20) & 0xFFFFF000u;
  if (!b) return false;
  const uint32_t dw2 = rd(obj + 0x24);
  const uint64_t w = (dw2 & 0x1FFFu) + 1u, h = ((dw2 >> 13) & 0x1FFFu) + 1u;
  *base = b;
  *span = uint32_t(std::clamp<uint64_t>(w * h * 4u, 4096u, 4u << 20));
  return true;
}
// Any-thread, mutex-queued retire of the guest-uploaded entries over a
// locked texture's range (drained on the render thread before its next
// texture lookup, DrainGuestTextureInvalidates).
static void QueueLockedTextureInvalidate(uint32_t obj) {
  uint32_t base = 0, span = 0;
  if (!LockedTextureRange(obj, &base, &span)) return;
  QueueGuestTextureInvalidate(base & 0x1FFFFFFFu, span);
}
void InvalidateTextureByHeader(uint32_t header_addr) {
  if (!header_addr) return;
  if (rq::Active()) {
    if (rq::OnProducerThread()) { rq::EnqInvalidateTexture(header_addr); return; }
    // Off-producer guest thread (XUI's image worker decoding PNG icons locks
    // textures too): the record ring is single-producer, so enqueueing here
    // would race the main thread's draw records and the invalidate could be
    // lost, leaving a recycled texture base serving its previous occupant.
    // Mutex-queued range retire instead; the atlas-class sentinel is a
    // main-thread (glyph) concern.
    if (header_addr == 0xFFFFFFFFu) return;
    QueueLockedTextureInvalidate(header_addr);
    return;
  }
  if (!OnRendererThread()) {
    // No offload: the lock hook runs on the locking thread. Worker-thread
    // locks (booth readbacks, icon decodes) must not walk/erase the cache.
    if (header_addr == 0xFFFFFFFFu) return;
    QueueLockedTextureInvalidate(header_addr);
    return;
  }
  if (header_addr == 0xFFFFFFFFu) {  // atlas-class sentinel (rq route)
    InvalidateSmallAlphaTexturesNow();
    return;
  }
  // The lock hook hands a texture object pointer while entries are keyed
  // by fetch addresses, so address equality alone matches nothing. Match
  // by guest texel base too, reading the header under both "object" and
  // "already the header" interpretations.
  uint32_t bases[2] = {0, 0};
  {
    const auto rd = [](uint32_t ea) -> uint32_t {
      const uint8_t* p = GuestPtr(ea);
      if (!p || !GuestRangeReadable(p, 4)) return 0;
      return __builtin_bswap32(*reinterpret_cast<const uint32_t*>(p));
    };
    bases[0] = rd(header_addr + 0x1C + 4) & 0xFFFFF000u;
    bases[1] = rd(header_addr + 4) & 0xFFFFF000u;
  }
  for (auto it = g_textureCache.begin(); it != g_textureCache.end();) {
    CachedTexture& e = it->second;
    const uint32_t ebase = e.header_dw[1] & 0xFFFFF000u;
    const bool addr_match = e.header_addr == header_addr;
    const bool base_match =
        ebase && ((bases[0] && ebase == bases[0]) ||
                  (bases[1] && ebase == bases[1]));
    if ((addr_match || base_match) && !e.host_rendered) {
      RetireEntry(e, "inval-header");  // fence-safe: descriptor scrubbed on drain
      it = g_textureCache.erase(it);
      g_texInvalidations++;
    } else {
      ++it;
    }
  }
}

// Atlas-class invalidation: header matching misses the atlas objects,
// and XUI renders the label strip the same frame it rasterizes a glyph.
// Locks only happen when glyphs rasterize, so retiring the whole
// small-R8 class on any lock is cheap and closes the race.
static void InvalidateSmallAlphaTexturesNow() {
  for (auto it = g_textureCache.begin(); it != g_textureCache.end();) {
    CachedTexture& e = it->second;
    // R8 only (glyph atlases). The small RGBA8 class covers textures that
    // are still in use, so retiring it wholesale only churns
    // retire/re-upload; icon staleness needs the specific icon entries.
    if (!e.host_rendered && e.valid &&
        e.host_format == RenderFormat::R8_UNORM && e.width <= 256 &&
        e.height <= 256) {
      RetireEntry(e);
      it = g_textureCache.erase(it);
      g_texInvalidations++;
    } else {
      ++it;
    }
  }
}

void InvalidateSmallAlphaTextures() {
  if (rq::Active()) {
    if (rq::OnProducerThread()) rq::EnqInvalidateTexture(0xFFFFFFFFu);
    return;  // off-producer threads: see InvalidateTextureByHeader
  }
  if (!OnRendererThread()) return;  // worker-thread lock: not a glyph raster
  InvalidateSmallAlphaTexturesNow();
}
// Core D3D unlock: the CPU write is complete, and an upload taken
// between lock and unlock captured the old texels; retire the range now
// so the next bind uploads the finished ones. Any thread (mutex queue).
void InvalidateTextureAfterUnlock(uint32_t header_addr) {
  if (!header_addr) return;
  uint32_t base = 0, span = 0, dw0 = 0;
  const bool tex = LockedTextureRange(header_addr, &base, &span, &dw0);
  if (!tex) return;
  QueueGuestTextureInvalidate(base & 0x1FFFFFFFu, span);
}

void NoteRttBegin(uint32_t texbase, uint32_t width, uint32_t height,
                  uint32_t color_surf, uint32_t depth_surf, uint32_t dest_tex,
                  uint32_t format, uint32_t msaa, uint32_t tiling) {
  if (rq::Active()) {
    rq::EnqNoteRtt(texbase, width, height, color_surf, depth_surf, dest_tex,
                   format, msaa, tiling);
    return;
  }
  // Sanity gate: implausible dims mean the title read drifted TU offsets,
  // drop the note (inference continues) and say so once.
  if (width < 16 || width > 4096 || height < 16 || height > 4096) {
    static uint64_t bad_logs = 4;
    if (bad_logs) {
      bad_logs--;
      REXGPU_WARN(
          "videonative: [semrtt] note REJECTED texbase={:#x} dims {}x{} "
          "(offset drift?)",
          texbase, width, height);
    }
    g_semRtt.active = false;
    return;
  }
  g_semRtt.active = true;
  g_semRtt.texbase = texbase;
  g_semRtt.width = width;
  g_semRtt.height = height;
  g_semRtt.color_surf = color_surf;
  g_semRtt.depth_surf = depth_surf;
  g_semRtt.dest_tex = dest_tex;
  g_semRtt.format = format;
  g_semRtt.msaa = msaa;
  g_semRtt.tiling = tiling;
}

void SetScissor(int32_t left, int32_t top, int32_t right, int32_t bottom) {
  if (rq::Active()) { rq::EnqSetScissor(left, top, right, bottom); return; }
  g_state.scissor[0] = left;
  g_state.scissor[1] = top;
  g_state.scissor[2] = right;
  g_state.scissor[3] = bottom;
  g_scissorExplicit = true;  // guest scissor wins over the bind-time reset
}
void SetTexture(uint32_t sampler, uint32_t header) {
  if (sampler >= 32) return;
  uint32_t dw[6] = {};
  if (header) {
    for (int i = 0; i < 6; i++) dw[i] = LoadGuestU32(header + i * 4);
  }
  if (rq::Active()) {
    rq::EnqSetTexture(sampler, header, dw);
    return;
  }
  SetTextureWithSnapshot(sampler, header, dw);
}

void SetTextureWithSnapshot(uint32_t sampler, uint32_t header,
                            const uint32_t* dw6) {
  if (sampler >= 32) return;
  g_state.textures[sampler] = header;
  // Bind-time fetch-dword snapshot: the XDK copies the header's fetch
  // dwords into the device shadow at SetTexture time; re-reading the
  // header at draw time sees recycled memory after a Release. Draws key
  // the cache on the original dwords.
  if (header) {
    std::memcpy(g_state.texture_dw[sampler], dw6, 24);
  } else {
    std::memset(g_state.texture_dw[sampler], 0,
                sizeof(g_state.texture_dw[sampler]));
  }
}
void SetAdapterIsIntel(bool is_intel) { g_adapterIsIntel = is_intel; }

void SetStream(uint32_t stream, uint32_t vb_object, uint32_t offset_bytes,
               uint32_t stride_bytes) {
  if (rq::Active()) { rq::EnqSetStream(stream, vb_object, offset_bytes, stride_bytes); return; }
  if (stream >= 16) return;
  StreamState& ss = g_state.streams[stream];
  ss.vb_object = vb_object;
  ss.offset_bytes = offset_bytes;
  ss.stride_bytes = stride_bytes;
  // Capture the VB object's fields now (stack-temporary headers, see
  // ib_snap_*): the deferred draw must never dereference the object later.
  ss.snap_valid = false;
  ss.snap_data = 0;
  ss.snap_size = 0;
  if (vb_object) {
    const uint32_t data = LoadGuestU32(vb_object + 24);
    const uint32_t size = LoadGuestU32(vb_object + 28);
    if (data) {
      ss.snap_valid = true;
      ss.snap_data = data;
      ss.snap_size = size;
    }
  }
}
void SetIndices(uint32_t ib_object) {
  if (rq::Active()) { rq::EnqSetIndices(ib_object); return; }
  g_state.index_buffer_object = ib_object;
  // Capture the object fields now (stack-temporary IB headers, see the
  // ib_snap_* comment in RenderState).
  g_state.ib_snap_valid = false;
  if (ib_object) {
    const uint32_t dw0 = LoadGuestU32(ib_object);
    const uint32_t data = LoadGuestU32(ib_object + 24);
    const uint32_t size = LoadGuestU32(ib_object + 28);
    if (data) {
      g_state.ib_snap_valid = true;
      g_state.ib_snap_data = data;
      g_state.ib_snap_size = size;
      g_state.ib_snap_idx32 = (dw0 & 0x80000000u) != 0;
    }
  }
}

void SetShaderConstantsF(bool pixel, uint32_t start_reg, uint32_t guest_data,
                         uint32_t vec4_count) {
  if (rq::Active()) { rq::EnqSetShaderConstantsF(pixel, start_reg, guest_data, vec4_count); return; }
  if (start_reg >= 256) return;
  vec4_count = std::min(vec4_count, 256 - start_reg);
  float* dst = (pixel ? g_state.ps_floats : g_state.vs_floats) + start_reg * 4;
  const uint32_t* src =
      rex::system::kernel_memory()->TranslateVirtual<const uint32_t*>(
          guest_data);
  for (uint32_t i = 0; i < vec4_count * 4; i++) {
    const uint32_t v = __builtin_bswap32(src[i]);
    std::memcpy(&dst[i], &v, 4);
  }
}

// Host-order data variant: replayed command-buffer constant sets carry the
// values captured at record time (the guest source pointer is long gone).
void SetShaderConstantsFHost(bool pixel, uint32_t start_reg,
                             const uint32_t* host_dwords, uint32_t vec4_count) {
  if (rq::Active()) { rq::EnqSetShaderConstantsFHost(pixel, start_reg, host_dwords, vec4_count); return; }
  if (start_reg >= 256) return;
  vec4_count = std::min(vec4_count, 256 - start_reg);
  float* dst = (pixel ? g_state.ps_floats : g_state.vs_floats) + start_reg * 4;
  std::memcpy(dst, host_dwords, size_t(vec4_count) * 16);
}

void PatchShaderFloatDwordsHost(bool pixel, uint32_t dword_index,
                                const uint32_t* host_dwords, uint32_t count) {
  if (rq::Active()) { rq::EnqPatchShaderFloatDwordsHost(pixel, dword_index, host_dwords, count); return; }
  if (dword_index >= 256 * 4) return;
  count = std::min(count, 256 * 4 - dword_index);
  float* dst = (pixel ? g_state.ps_floats : g_state.vs_floats) + dword_index;
  std::memcpy(dst, host_dwords, size_t(count) * 4);
}
void MergeShaderFloatDwordHost(bool pixel, uint32_t dword_index, uint32_t mask,
                               uint32_t orv) {
  if (rq::Active()) { rq::EnqMergeShaderFloatDwordHost(pixel, dword_index, mask, orv); return; }
  if (dword_index >= 256 * 4) return;
  float* dst = (pixel ? g_state.ps_floats : g_state.vs_floats) + dword_index;
  uint32_t v;
  std::memcpy(&v, dst, 4);
  v = (v & mask) | orv;
  std::memcpy(dst, &v, 4);
}

void Clear(uint32_t, uint32_t flags, uint32_t color, float z,
           const int32_t* rects, uint32_t rect_count, uint32_t stencil) {
  if (rq::Active()) {
    rq::EnqClear(flags, color, z, rects, rect_count, stencil);
    return;
  }
  EnsureFrameOpen();
  if (!g_frameOpen) return;
  ApplyRenderTargetState();  // clears apply to the bound guest surfaces
  // Per-tile loops interleave full-surface clears between tile resolves;
  // in the native single-surface model those clears would wipe later
  // tiles' source before they resolve, so they are dropped for grown RTs
  // of the wide bake class. Rect-scoped clears always pass.
  if (REXCVAR_GET(native_video_vp_grow) &&
      (g_activeRtGrown || (g_tilingWidth && g_activeRtWidth >= 2048)) &&
      g_passResolves > 0 && rect_count == 0) {
    return;
  }
  // Only a clear that executes invalidates the depth-staging plane memo:
  // the suppressed mid-tile-loop clears above return before touching the
  // RT, and the memo exists precisely because those bands share one render.
  g_rtMutationSeq++;
  auto& commandList = g_commandLists[g_frame];
  // XDK pRects scope the clear (per-tile command buffers clear only their
  // band, a full clear here would wipe the other tiles' content).
  RenderRect clear_rects[4];
  const uint32_t n = std::min(rect_count, 4u);
  // Swapchain-targeted clears (no active RT) stay 1:1.
  const auto cc = [&](int32_t v) { return g_activeRtKey ? HostCoord(v) : v; };
  for (uint32_t i = 0; i < n; i++) {
    clear_rects[i] =
        RenderRect(cc(rects[i * 4 + 0]), cc(rects[i * 4 + 1]),
                   cc(rects[i * 4 + 2]), cc(rects[i * 4 + 3]));
  }
  // D3DCLEAR: target bits low, zbuffer 0x10-ish, clear color on any target
  // bit, depth on the z bit (Xenos flag values used by another title: 0x1F all, 0x10 z).
  if (flags & 0xF) {
    const RenderColor clear_color(
        ((color >> 16) & 0xFF) / 255.0f, ((color >> 8) & 0xFF) / 255.0f,
        (color & 0xFF) / 255.0f, ((color >> 24) & 0xFF) / 255.0f);
    // Bits 0-3 = color targets 0-3; the swapchain framebuffer only has one.
    const uint32_t attachment_count = g_activeRtKey ? kMrtCount : 1;
    for (uint32_t i = 0; i < attachment_count; i++) {
      if (flags & (1u << i)) {
        commandList->clearColor(i, clear_color, n ? clear_rects : nullptr, n);
      }
    }
  }
  if (flags & 0x30) {
    // Inverted-viewport-Z pass (Xenos revz idiom, AE avatar passes): the
    // stored depth convention is flipped (z' = 1-z, see SetupDraw), so a
    // clear issued under an inverted viewport carries the guest's reversed
    // value (far = 0.0) and must flip with it.
    float zc = z;
    if (g_state.viewport_min_z > g_state.viewport_max_z) zc = 1.0f - zc;
    // D3DCLEAR: 0x10 = ZBUFFER, 0x20 = STENCIL, clear each plane only
    // when its flag is set (the AE mirror pass clears stencil to 0 before
    // the silhouette punch).
    commandList->clearDepthStencil((flags & 0x10) != 0, (flags & 0x20) != 0,
                                   zc, stencil & 0xFF,
                                   n ? clear_rects : nullptr, n);
  }
}

void SetTilingExtent(uint32_t width, uint32_t height) {
  if (rq::Active()) { rq::EnqSetTilingExtent(width, height); return; }
  g_tilingWidth = width;
  g_tilingHeight = height;
  if (width && height) {
    g_lastTilingWidth = width;
    g_lastTilingHeight = height;
  }
}

void RegisterSurface(uint32_t surface_obj, uint32_t width, uint32_t height,
                     uint32_t guest_format) {
  if (rq::Active()) { rq::EnqRegisterSurface(surface_obj, width, height, guest_format); return; }
  SurfaceInfo info;
  info.width = width;
  info.height = height;
  info.guest_format = guest_format;
  info.valid = width > 0 && width <= 8192 && height > 0 && height <= 8192;
  g_surfaces[surface_obj] = info;
}

void SetRenderTargetSurface(uint32_t index, uint32_t surface_obj) {
  if (rq::Active()) { rq::EnqSetRenderTargetSurface(index, surface_obj); return; }
  g_state.color_surface[index & 3u] = surface_obj;
}
void SetDepthSurface(uint32_t surface_obj) {
  if (rq::Active()) { rq::EnqSetDepthSurface(surface_obj); return; }
  g_state.depth_surface = surface_obj;
}

// Snapshot/restore of the captured guest state around command-buffer replay
// (the XDK re-emits device shadow state after Run, CB binds must not leak).
static std::vector<GuestState> g_stateStack;
void PushState() {
  if (rq::Active()) { rq::EnqPushState(); return; } g_stateStack.push_back(g_state); }
void PopState() {
  if (rq::Active()) { rq::EnqPopState(); return; }
  if (g_stateStack.empty()) return;
  g_state = g_stateStack.back();
  g_stateStack.pop_back();
}

// XDK Run contract: a command buffer inherits the device's current constants
// (the game zeroes them during recording and re-sets before every Run, often
// through recompiled paths no hook sees). Seed the replay mirror from the
// live device float shadow; recorded in-CB sets overlay afterwards. Without
// it, per-frame constants that only ever reach the device shadow leave
// CB-replayed draws running on stale record-time values.
void SeedReplayFloatConstants(uint32_t device, uint32_t recording_device) {
  if (rq::Active()) { rq::EnqSeedReplayFloatConstants(device, recording_device); return; }
  const uint32_t* vs_src =
      reinterpret_cast<const uint32_t*>(GuestPtr(device + 1920));
  const uint32_t* ps_src =
      reinterpret_cast<const uint32_t*>(GuestPtr(device + 6016));
  uint32_t* vdst = reinterpret_cast<uint32_t*>(g_state.vs_floats);
  uint32_t* pdst = reinterpret_cast<uint32_t*>(g_state.ps_floats);
  for (uint32_t i = 0; i < 256 * 4; i++) {
    vdst[i] = __builtin_bswap32(vs_src[i]);
    pdst[i] = __builtin_bswap32(ps_src[i]);
  }
  if (!recording_device || recording_device == device) return;
  // Registers the run device never touched but the recording device holds:
  // hardware executes the CB with those values baked in (see renderer.h).
  const uint32_t* rec[2] = {
      reinterpret_cast<const uint32_t*>(GuestPtr(recording_device + 1920)),
      reinterpret_cast<const uint32_t*>(GuestPtr(recording_device + 6016))};
  uint32_t* dst[2] = {vdst, pdst};
  for (int stage = 0; stage < 2; stage++) {
    for (uint32_t r = 0; r < 256; r++) {
      uint32_t* d = dst[stage] + r * 4;
      if (d[0] | d[1] | d[2] | d[3]) continue;  // live value wins
      const uint32_t* s = rec[stage] + r * 4;
      const uint32_t v0 = __builtin_bswap32(s[0]);
      const uint32_t v1 = __builtin_bswap32(s[1]);
      const uint32_t v2 = __builtin_bswap32(s[2]);
      const uint32_t v3 = __builtin_bswap32(s[3]);
      if (!(v0 | v1 | v2 | v3)) continue;
      d[0] = v0;
      d[1] = v1;
      d[2] = v2;
      d[3] = v3;
    }
  }
}

// Render-state override for replayed draws (see renderer.h).
static uint32_t g_replayDrawState[10];
static uint32_t g_replayDrawStateDirty = 0;
static bool g_replayDrawStateValid = false;
void ApplyReplayStatePersistent(const uint32_t* state10,
                                uint32_t dirty_mask) {
  if (rq::Active()) { rq::EnqApplyReplayStatePersistent(state10, dirty_mask); return; }
  if (!state10) return;
  if (dirty_mask & (1u << 1)) g_fxBlendControl[0] = state10[1];
  if (dirty_mask & (1u << 4)) g_fxColorMask = state10[4];
  if (dirty_mask & (1u << 7)) g_fxBlendControl[1] = state10[7];
  if (dirty_mask & (1u << 8)) g_fxBlendControl[2] = state10[8];
  if (dirty_mask & (1u << 9)) g_fxBlendControl[3] = state10[9];
}

void ApplyBlendControlDirect(uint32_t rt, uint32_t value) {
  if (rq::Active()) { rq::EnqApplyBlendControlDirect(rt, value); return; }
  if (rt < kMrtCount) g_fxBlendControl[rt] = value;
}

// Fold the run device's live blend/mask programming (dev+16 dirty bits +
// register shadows) into the effective register file and drain the bits,
// models the XDK Run pre-flush (live writes reach hardware before the CB's
// PM4). Called at RunCommandBuffer start and at live draws; not at replayed
// draws, where it would invert the order against recorded direct blend
// writes (see SetupDraw).
void ConsumeLiveStateDirty(uint32_t device) {
  // Read + clear the guest dirty bits at call time (the game re-sets them
  // between calls); the captured values apply on the worker in order.
  const uint64_t dirty_hi = LoadGuestU32(device + 16);
  const uint64_t dirty_lo = LoadGuestU32(device + 20);
  uint64_t dirty = (dirty_hi << 32) | dirty_lo;
  const uint64_t consumed =
      dirty & ((1ull << 10) | (1ull << 3) | (1ull << 2) | (1ull << 1) | 1ull);
  uint32_t vals[5] = {0, 0, 0, 0, 0};
  if (dirty & (1ull << 10)) vals[0] = LoadGuestU32(device + 10552);
  if (dirty & (1ull << 2)) vals[1] = LoadGuestU32(device + 10584);
  if (dirty & (1ull << 1)) vals[2] = LoadGuestU32(device + 10588);
  if (dirty & (1ull << 0)) vals[3] = LoadGuestU32(device + 10592);
  if (dirty & (1ull << 3)) vals[4] = LoadGuestU32(device + 10580);
  if (consumed) {
    dirty &= ~consumed;
    StoreGuestU32(device + 16, uint32_t(dirty >> 32));
    StoreGuestU32(device + 20, uint32_t(dirty));
  }
  if (rq::Active()) {
    rq::EnqApplyLiveStateDirty(uint32_t(consumed), vals);
    return;
  }
  ApplyLiveStateDirtyHost(uint32_t(consumed), vals);
}

void ApplyLiveStateDirtyHost(uint32_t consumed_mask, const uint32_t vals[5]) {
  if (consumed_mask & (1u << 10)) g_fxBlendControl[0] = vals[0];
  if (consumed_mask & (1u << 2)) g_fxBlendControl[1] = vals[1];
  if (consumed_mask & (1u << 1)) g_fxBlendControl[2] = vals[2];
  if (consumed_mask & (1u << 0)) g_fxBlendControl[3] = vals[3];
  if (consumed_mask & (1u << 3)) g_fxColorMask = vals[4];
}

// Exported resolver for the rq enqueue capture (render_queue.cpp): probes
// and memoizes like every other stream read (VirtualQuery once per span,
// negatives cached), the enqueue side must never take a fault on junk
// fetch-shadow pairs, and the runtime's VEH makes SEH guards unreliable.
const uint8_t* GuestDataPtrProbe(uint32_t addr, size_t bytes) {
  return GuestDataPtrFast(addr, bytes);
}

void QueueGuestTextureFreeze(uint32_t guest_address, uint32_t size) {
  if (!size) return;
  // Offload write barrier: queued draws may still worker-read big VBs
  // living in this range (deferred > kSnapVbMax). Let them finish before
  // the caller starts rewriting, or the bake geometry reads torn.
  rq::DrainForGuestRewrite();
  std::lock_guard<std::mutex> lk(g_invalidateMx);
  g_frozenRanges.emplace_back(guest_address & 0x1FFFFFFFu, size, g_frameIndex);
  g_hasFrozenRanges.store(true, std::memory_order_release);
}

void QueueGuestTextureInvalidate(uint32_t guest_address, uint32_t size) {
  if (!size) return;
  std::lock_guard<std::mutex> lk(g_invalidateMx);
  g_pendingInvalidates.emplace_back(guest_address & 0x1FFFFFFFu, size);
  g_hasPendingInvalidates.store(true, std::memory_order_release);
}

// Guest-thread entry for the LockRect hook: queue the CPU-authority note;
// the render thread drains it (DrainGuestCpuLocks) before texture lookup.
void NoteGuestTextureCpuLock(uint32_t base) {
  std::lock_guard<std::mutex> lk(g_invalidateMx);
  g_pendingCpuLocks.push_back(base & 0x1FFFF000u);
  g_hasPendingCpuLocks.store(true, std::memory_order_release);
}

void SetReplayDrawState(const uint32_t* state10, uint32_t dirty_mask) {
  if (rq::Active()) { rq::EnqSetReplayDrawState(state10, dirty_mask); return; }
  if (state10) {
    std::memcpy(g_replayDrawState, state10, sizeof(g_replayDrawState));
    g_replayDrawStateDirty = dirty_mask;
    g_replayDrawStateValid = true;
  } else {
    g_replayDrawStateDirty = 0;
    g_replayDrawStateValid = false;
  }
}

// Bool/loop constants captured from the recording device at record time
// (host-order 40 dwords), the unhooked recompiled SetShaderConstantB/I
// setters write the recording device's shadow block, which the live draw
// device never sees (the avatar bool uploader runs during Avatar_Draw
// recording, and the bake passes set their branch bools while recording).
static uint32_t g_replayBoolLoop[40];
static bool g_replayBoolLoopValid = false;
static uint32_t g_replayFetch[32][6];
static uint32_t g_replayFetchValid = 0;  // bit per slot
void SetReplayFetchConstants(const uint32_t* records, uint32_t count) {
  if (rq::Active()) { rq::EnqSetReplayFetchConstants(records, count); return; }
  g_replayFetchValid = 0;
  for (uint32_t r = 0; r < count; r++) {
    const uint32_t slot = records[r * 7];
    if (slot >= 32) continue;
    std::memcpy(g_replayFetch[slot], &records[r * 7 + 1], 24);
    g_replayFetchValid |= 1u << slot;
  }
}

void SetReplayBoolConstants(const uint32_t* dwords40) {
  if (rq::Active()) { rq::EnqSetReplayBoolConstants(dwords40); return; }
  if (dwords40) {
    std::memcpy(g_replayBoolLoop, dwords40, sizeof(g_replayBoolLoop));
    g_replayBoolLoopValid = true;
  } else {
    g_replayBoolLoopValid = false;
  }
}

// FPS overlay (renderer_fps.h): a persistent RGBA8 strip re-uploaded when the
// readout changes, drawn with the frontbuffer blit pipeline into a top-left
// viewport box. Assumes the caller has already bound g_blitPipeline with the
// swapchain as the render target (i.e. runs at the tail of SwapFrontbuffer).
namespace {
void DrawFpsOverlay(std::unique_ptr<RenderCommandList>& commandList) {
  const fps::Overlay overlay = fps::Tick();
  if (!overlay.visible) return;

  bool created = false;
  if (!g_fpsTexture) {
    if (g_nextTextureDescriptor >= kFirstReservedDescriptor) return;
    g_fpsTexture = g_device->createTexture(RenderTextureDesc::Texture2D(
        overlay.width, overlay.height, 1, RenderFormat::R8G8B8A8_UNORM));
    g_fpsView = g_fpsTexture->createTextureView(
        RenderTextureViewDesc::Texture2DArray(RenderFormat::R8G8B8A8_UNORM));
    g_fpsDescriptor = g_nextTextureDescriptor++;
    g_textureDescriptorSet->setTexture(g_fpsDescriptor, g_fpsTexture.get(),
                                       RenderTextureLayout::SHADER_READ,
                                       g_fpsView.get());
    created = true;
  }

  if (overlay.dirty || created) {
    const uint32_t row_bytes = overlay.width * 4;
    const uint32_t padded_row_bytes = (row_bytes + 255u) & ~255u;
    UploadAllocation staging = g_upload[g_frame].Allocate(
        (padded_row_bytes * overlay.height + 511) & ~511u, 512);
    for (uint32_t y = 0; y < overlay.height; y++) {
      std::memcpy(staging.memory + y * padded_row_bytes,
                  overlay.pixels + y * row_bytes, row_bytes);
    }
    commandList->barriers(RenderBarrierStage::COPY,
                          RenderTextureBarrier(g_fpsTexture.get(),
                                               RenderTextureLayout::COPY_DEST));
    commandList->copyTextureRegion(
        RenderTextureCopyLocation::Subresource(g_fpsTexture.get(), 0, 0),
        RenderTextureCopyLocation::PlacedFootprint(
            staging.buffer, RenderFormat::R8G8B8A8_UNORM, overlay.width,
            overlay.height, 1, padded_row_bytes / 4, staging.offset));
    commandList->barriers(
        RenderBarrierStage::GRAPHICS,
        RenderTextureBarrier(g_fpsTexture.get(),
                             RenderTextureLayout::SHADER_READ));
  }

  // Top-left of the game image (with letterboxing the image rect is
  // inset); the rasterizer chooses the scale.
  if (g_fpsPipeline) commandList->setPipeline(g_fpsPipeline.get());
  constexpr float kMargin = 8.0f;
  const float box_w = overlay.width * overlay.scale;
  const float box_h = overlay.height * overlay.scale;
  const float box_x = g_presentRect[0] + kMargin;
  const float box_y = g_presentRect[1] + kMargin;
  // Scissor crops the strip to the text's actual width so short readouts
  // don't trail dead backdrop on the right.
  const float used_w =
      (overlay.used_width ? overlay.used_width : overlay.width) *
      overlay.scale;
  commandList->setViewports(
      RenderViewport(box_x, box_y, box_w, box_h, 0.0f, 1.0f));
  commandList->setScissors(
      RenderRect(int32_t(box_x), int32_t(box_y),
                 int32_t(box_x + used_w), int32_t(box_y + box_h)));
  uint32_t indices[32] = {g_fpsDescriptor};
  UploadAllocation b4 = g_upload[g_frame].Allocate(sizeof(indices), 256);
  std::memcpy(b4.memory, indices, sizeof(indices));
  commandList->setGraphicsRootDescriptor(b4.buffer->at(b4.offset),
                                         kRootB4IndicesPs);
  commandList->drawInstanced(3, 1, 0, 0);
}

}  // namespace

// Fullscreen blit of the frontbuffer texture onto the swapchain backbuffer,
// the only path by which a resolve-based title reaches the screen (the game
// never draws to the real backbuffer; it Swaps its own resolved texture).
void UpdateGammaRamp(const uint16_t* rgb768) {
  if (rq::Active()) { rq::EnqUpdateGammaRamp(rgb768); return; }
  EnsureFrameOpen();
  if (!g_frameOpen || !g_device) return;
  if (!g_gammaRampTexture) {
    g_gammaRampTexture = g_device->createTexture(
        RenderTextureDesc::Texture2D(256, 3, 1, RenderFormat::R16_UNORM));
    if (!g_gammaRampTexture) return;
    g_gammaRampView = g_gammaRampTexture->createTextureView(
        RenderTextureViewDesc::Texture2DArray(RenderFormat::R16_UNORM));
    if (g_nextTextureDescriptor < kFirstReservedDescriptor) {  // top slots are blanks
      g_gammaRampDescriptor = g_nextTextureDescriptor++;
      g_textureDescriptorSet->setTexture(
          g_gammaRampDescriptor, g_gammaRampTexture.get(),
          RenderTextureLayout::SHADER_READ, g_gammaRampView.get());
    }
    REXGPU_INFO("videonative: gamma ramp LUT created -> desc {}",
                g_gammaRampDescriptor);
  }
  auto& commandList = g_commandLists[g_frame];
  commandList->barriers(RenderBarrierStage::COPY,
                        RenderTextureBarrier(g_gammaRampTexture.get(),
                                             RenderTextureLayout::COPY_DEST));
  // 3 rows of 256 R16 texels = 512 bytes/row (256-aligned as placed
  // footprints require).
  UploadAllocation staging = g_upload[g_frame].Allocate(512 * 3, 512);
  if (!staging.memory) return;
  std::memcpy(staging.memory, rgb768, 512 * 3);
  commandList->copyTextureRegion(
      RenderTextureCopyLocation::Subresource(g_gammaRampTexture.get(), 0),
      RenderTextureCopyLocation::PlacedFootprint(staging.buffer,
                                                 RenderFormat::R16_UNORM, 256,
                                                 3, 1, 256, staging.offset));
  commandList->barriers(RenderBarrierStage::GRAPHICS,
                        RenderTextureBarrier(g_gammaRampTexture.get(),
                                             RenderTextureLayout::SHADER_READ));
}

void SwapFrontbuffer(uint32_t frontbuffer_header) {
  if (rq::Active()) { rq::EnqSwapFrontbuffer(frontbuffer_header); return; }
  if (!frontbuffer_header || !g_swapChain) return;
  // Frame boundary: retire the RTT note (a fresh StartTextureRendering
  // re-arms it next frame; a stale note must never outlive its frame).
  g_semRtt.active = false;
  EnsureFrameOpen();
  if (!g_frameOpen) return;


  if (!g_blitTried) {
    g_blitTried = true;
    const auto& vs = detail::BlitVsDxil();
    const auto& ps = detail::BlitPsDxil();
    if (!vs.empty() && !ps.empty()) {
      g_blitVs = g_device->createShader(vs.data(), vs.size(), "main",
                                        RenderShaderFormat::DXIL);
      g_blitPs = g_device->createShader(ps.data(), ps.size(), "main",
                                        RenderShaderFormat::DXIL);
      RenderGraphicsPipelineDesc desc;
      desc.pipelineLayout = g_pipelineLayout.get();
      desc.vertexShader = g_blitVs.get();
      desc.pixelShader = g_blitPs.get();
      desc.renderTargetFormat[0] = kColorFormat;
      desc.renderTargetCount = 1;
      desc.depthTargetFormat = kDepthFormat;
      desc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
      desc.depthEnabled = false;
      desc.depthWriteEnabled = false;
      desc.cullMode = RenderCullMode::NONE;
      g_blitPipeline = g_device->createGraphicsPipeline(desc);
      // FPS overlay: same pair but with a PS that preserves sampled alpha
      // (blit.ps forces a = 1, which would defeat the blend and render the
      // strip's transparent background as an opaque box) + alpha blending.
      g_fpsPs = g_device->createShader(kFpsOverlayPsDxil,
                                       sizeof(kFpsOverlayPsDxil), "main",
                                       RenderShaderFormat::DXIL);
      desc.pixelShader = g_fpsPs.get();
      desc.renderTargetBlend[0] = RenderBlendDesc::AlphaBlend();
      g_fpsPipeline = g_device->createGraphicsPipeline(desc);
    }
  }
  static uint64_t swap_log_budget = 8;
  if (!g_blitPipeline) {
    if (swap_log_budget) {
      swap_log_budget--;
      REXGPU_WARN("videonative: [swaplog] no blit pipeline");
    }
    return;
  }

  CachedTexture* tex = GetOrCreateTexture(frontbuffer_header, true, 0, 0,
                                          RenderFormat::UNKNOWN, nullptr,
                                          /*resolve_scale=*/-1);
  // Present the last resolve into a texture of the frontbuffer's size:
  // within the game's frame the scene resolves precede the composite
  // resolve, so this is by construction the finished image, independent of
  // the game's front/back texture-pointer flip protocol, which does not line
  // up with native present timing.
  bool from_resolve = false;
  if (tex && tex->valid) {
    const uint64_t size_key = SizeKeyOf(tex->width, tex->height);
    auto last = g_lastResolvedBySize.find(size_key);
    if (last != g_lastResolvedBySize.end()) {
      tex = last->second;
      from_resolve = true;
    }
  }
  if (!tex || !tex->valid || !tex->descriptor_index) return;
  auto& commandList = g_commandLists[g_frame];
  if (tex->layout != RenderTextureLayout::SHADER_READ) {
    commandList->barriers(
        RenderBarrierStage::GRAPHICS,
        RenderTextureBarrier(tex->texture.get(),
                             RenderTextureLayout::SHADER_READ));
    tex->layout = RenderTextureLayout::SHADER_READ;
  }

  // Back to the swapchain framebuffer for the composite. Both surfaces must
  // read as unbound, a lingering depth surface would route the blit at the
  // depth-only host RT (see ApplyRenderTargetState).
  uint32_t saved_surfaces[kMrtCount];
  std::memcpy(saved_surfaces, g_state.color_surface, sizeof(saved_surfaces));
  const uint32_t saved_depth = g_state.depth_surface;
  std::memset(g_state.color_surface, 0, sizeof(g_state.color_surface));
  g_state.depth_surface = 0;
  ApplyRenderTargetState();
  std::memcpy(g_state.color_surface, saved_surfaces, sizeof(saved_surfaces));
  g_state.depth_surface = saved_depth;

  // No-resolve titles: the guest never CPU-writes its frontbuffer texture
  // (upload_skipped) and no resolve chain exists, so the no-RT draws have
  // already rendered into the swapchain framebuffer and blitting the zero
  // frontbuffer texture over them would erase the frame. The frontbuffer
  // entry here is the for_resolve flavor: host_rendered with
  // last_resolve_frame == 0 means allocated but never written by anything.
  // Skip the content blit; keep viewport setup + FPS overlay + present.
  const bool blit_content =
      from_resolve || !(tex->upload_skipped ||
                        (tex->host_rendered && tex->last_resolve_frame == 0));
  if (blit_content) commandList->setPipeline(g_blitPipeline.get());
  // Aspect-preserving fit (default): scale the 16:9 game image to the
  // largest box that fits the swapchain, centered, letterbox on 16:10,
  // pillarbox on 4:3, never distorted. The frame-open clear paints the
  // bars. native_video_stretch fills the window instead.
  {
    const float sw = float(g_swapChain->getWidth());
    const float sh = float(g_swapChain->getHeight());
    float vx = 0.0f, vy = 0.0f, vw = sw, vh = sh;
    if (!REXCVAR_GET(native_video_stretch) && tex->height > 0) {
      const float aspect = float(tex->width) / float(tex->height);
      if (sw / sh > aspect) {
        vw = sh * aspect;
        vx = std::floor((sw - vw) * 0.5f);
      } else {
        vh = sw / aspect;
        vy = std::floor((sh - vh) * 0.5f);
      }
    }
    g_presentRect[0] = vx;
    g_presentRect[1] = vy;
    g_presentRect[2] = vw;
    g_presentRect[3] = vh;
    commandList->setViewports(RenderViewport(vx, vy, vw, vh, 0.0f, 1.0f));
    commandList->setScissors(RenderRect(int32_t(vx), int32_t(vy),
                                        int32_t(vx + vw), int32_t(vy + vh)));
  }
  if (blit_content) {
    uint32_t indices[32] = {tex->descriptor_index};
    // b4 [0].y = gamma-ramp LUT descriptor (0 = no ramp): the game's display
    // contrast curve, applied on top of the frontbuffer sample in blit.ps.
    indices[1] =
        REXCVAR_GET(native_video_gamma_ramp) ? g_gammaRampDescriptor : 0;
    UploadAllocation b4 = g_upload[g_frame].Allocate(sizeof(indices), 256);
    std::memcpy(b4.memory, indices, sizeof(indices));
    commandList->setGraphicsRootDescriptor(b4.buffer->at(b4.offset),
                                           kRootB4IndicesPs);
    commandList->drawInstanced(3, 1, 0, 0);
  }

  DrawFpsOverlay(commandList);
}

// Copies the bound render target into the destination guest texture's host
// cache entry (the guest never sees the pixels, draws sampling the header
// hit the same entry).
namespace {
// Shared body for color and depth resolves: copies the (clamped) source
// region of the active RT into the destination texture at dest point.
void ResolveRegion(uint32_t dest_texture_header, bool depth, uint32_t source,
                   bool has_rect, int32_t src_left, int32_t src_top,
                   int32_t src_right, int32_t src_bottom, int32_t dest_x,
                   int32_t dest_y) {
  if (!dest_texture_header) return;
  g_passResolves++;  // per-tile-loop detector (reset on target change)
  EnsureFrameOpen();
  if (!g_frameOpen) return;
  ApplyRenderTargetState();
  auto& commandList = g_commandLists[g_frame];

  RenderTexture* src = nullptr;
  CachedRenderTarget* rt = nullptr;
  const uint32_t rt_index = std::min(source, kMrtCount - 1);
  uint32_t src_w = 0, src_h = 0;
  // Source = the bound render target (hardware resolves read EDRAM at
  // the current binding); fall back to the last-drawn RT only when nothing
  // is bound. Preferring last-drawn breaks the clear-then-resolve idiom,
  // where a pass that draws nothing still resolves the cleared RT.
  const uint64_t src_key = g_activeRtKey ? g_activeRtKey : g_lastDrawnRtKey;
  // A resolve into a base re-establishes host authority: clear any
  // CPU-owned stamp so binds serve the fresh host product again. A
  // compose lock after the resolve re-stamps.
  {
    const uint32_t dest_base =
        (LoadGuestU32(dest_texture_header + 4) & 0xFFFFF000u) & 0x1FFFF000u;
    g_cpuLockedBases.erase(dest_base);
  }
  if (src_key != 0) {
    auto it = g_renderTargets.find(src_key);
    if (it == g_renderTargets.end()) {
      // A silent return here leaves stale content: the guest believes it
      // just wrote this destination, nothing was copied, and the dest
      // texture (plus any registration) keeps serving its previous
      // content.
      g_resolveSrcMissing++;
      static uint64_t srcmiss_logs = 24;
      if (srcmiss_logs) {
        srcmiss_logs--;
        REXGPU_WARN(
            "videonative: [rsvmiss] resolve dropped, source RT not cached "
            "(key={:#x} destHdr={:#x} destBase={:#x} f={})",
            src_key, dest_texture_header,
            LoadGuestU32(dest_texture_header + 4) & 0xFFFFF000u,
            g_frameIndex);
      }
      return;
    }
    rt = &it->second;
    src = depth ? rt->depth.get() : rt->color[rt_index].get();
    src_w = rt->width;
    src_h = rt->height;
  } else {
    // (swapchain fallback below; under a downscale it early-returns, so the
    // 1:1 box math after this block is safe for both source kinds)
    if (depth) return;  // swapchain depth copies not supported (never tiled)
    if (Scaled()) {
      static uint64_t sc_skip_logs = 4;
      if (sc_skip_logs) {
        sc_skip_logs--;
        REXGPU_WARN(
            "videonative: swapchain-source resolve skipped under "
            "resolution_scale (unscaled source, scaled dest)");
      }
      return;
    }
    src = const_cast<RenderTexture*>(g_swapChain->getTexture(g_backBufferIndex));
    src_w = g_swapChain->getWidth();
    src_h = g_swapChain->getHeight();
  }

  // The destination is allocated at guest header dimensions (an
  // oversized depth dest skews every normalized-UV lookup). Depth copies
  // stage to satisfy D3D12's whole-subresource rule; the dest entry is
  // created with the source RT's format family, since CopyTextureRegion
  // cannot convert like a hardware resolve.
  const RenderFormat required =
      depth ? RenderFormat::R32_FLOAT
            : (rt ? rt->formats[rt_index] : kColorFormat);
  CachedTexture* dest = GetOrCreateTexture(dest_texture_header, true, 0, 0,
                                           required, nullptr, 0);
  if (!dest || !dest->valid) return;
  if (dest->host_format != required) {
    static uint64_t mismatch_logs = 8;
    if (mismatch_logs) {
      mismatch_logs--;
      REXGPU_WARN(
          "videonative: resolve format mismatch (depth={} src={} dest={}), "
          "skipped",
          depth, uint32_t(required), uint32_t(dest->host_format));
    }
    return;
  }
  // No rect: resolve the whole source to the destination origin. Every rect
  // below is clamped to both surfaces.
  if (!has_rect) {
    src_left = 0;
    src_top = 0;
    src_right = int32_t(src_w);
    src_bottom = int32_t(src_h);
    dest_x = 0;
    dest_y = 0;
  }
  // EDRAM-aliasing rebase: the guest draws through a small surface and
  // resolves the same EDRAM tiles through the main surface's coordinate
  // space. When a small dest's rect matches its dims but the picked
  // source is larger, and a same-dims RT holds fresher draws, resolve
  // from that RT re-based to (0,0).
  if (REXCVAR_GET(native_video_small_resolve_rebase) && !depth && rt &&
      dest->width <= 256 && dest->height <= 256 &&
      uint32_t(src_right - src_left) == dest->width &&
      uint32_t(src_bottom - src_top) == dest->height &&
      (src_w > dest->width || src_h > dest->height)) {
    CachedRenderTarget* best = nullptr;
    uint64_t best_serial = 0;
    for (auto& [rkey, cand] : g_renderTargets) {
      if (cand.width == dest->width && cand.height == dest->height &&
          cand.color[rt_index] &&
          cand.last_draw_serial > cand.last_sourced_serial &&
          cand.last_draw_serial > best_serial) {
        best = &cand;
        best_serial = cand.last_draw_serial;
      }
    }
    if (best) {
      best->last_sourced_serial = best->last_draw_serial;
      rt = best;
      src = best->color[rt_index].get();
      src_w = best->width;
      src_h = best->height;
      src_left = 0;
      src_top = 0;
      src_right = int32_t(dest->width);
      src_bottom = int32_t(dest->height);
    }
  }
  // Un-bracketed tiled bake resolves: the game re-renders slice N into a
  // tile-height surface and resolves the frame-coordinate rect. Re-base
  // the rect whenever it exceeds the RT (the slice's content occupies RT
  // rows from 0); a fitting rect is left untouched.
  if (has_rect && src_bottom > int32_t(src_h) && dest_y == src_top) {
    src_bottom -= src_top;
    src_top = 0;
  }
  if (has_rect && src_right > int32_t(src_w) && dest_x == src_left) {
    src_right -= src_left;
    src_left = 0;
  }
  src_left = std::clamp(src_left, 0, int32_t(src_w));
  src_top = std::clamp(src_top, 0, int32_t(src_h));
  src_right = std::clamp(src_right, src_left, int32_t(src_w));
  src_bottom = std::clamp(src_bottom, src_top, int32_t(src_h));
  dest_x = std::clamp(dest_x, 0, int32_t(dest->width));
  dest_y = std::clamp(dest_y, 0, int32_t(dest->height));
  const int32_t copy_w =
      std::min(src_right - src_left, int32_t(dest->width) - dest_x);
  const int32_t copy_h =
      std::min(src_bottom - src_top, int32_t(dest->height) - dest_y);
  if (copy_w <= 0 || copy_h <= 0) return;


  // Entry transitions. Lazy mode consults the tracked layouts so a chain
  // of resolves from one source RT pays a single source transition; eager
  // mode transitions unconditionally.
  const bool lazy = REXCVAR_GET(native_video_resolve_lazy);
  const RenderTextureLayout src_now =
      rt ? (depth ? rt->depth_layout : rt->color_layout[rt_index])
         : RenderTextureLayout::UNKNOWN;
  uint32_t entry_count = 0;
  RenderTextureBarrier entry[2];
  if (!lazy || src_now != RenderTextureLayout::COPY_SOURCE) {
    entry[entry_count++] =
        RenderTextureBarrier(src, RenderTextureLayout::COPY_SOURCE);
    g_resolveBarriersEmitted++;
  }
  if (!lazy || dest->layout != RenderTextureLayout::COPY_DEST) {
    entry[entry_count++] = RenderTextureBarrier(
        dest->texture.get(), RenderTextureLayout::COPY_DEST);
    g_resolveBarriersEmitted++;
  }
  if (entry_count) {
    commandList->barriers(RenderBarrierStage::COPY, entry, entry_count);
  }
  if (rt) {
    if (depth) {
      rt->depth_layout = RenderTextureLayout::COPY_SOURCE;
    } else {
      rt->color_layout[rt_index] = RenderTextureLayout::COPY_SOURCE;
    }
  }
  dest->layout = RenderTextureLayout::COPY_DEST;

  if (depth) {
    // Whole depth plane -> staging buffer (whole-subresource, satisfies the
    // D3D12 depth-copy rule), then the requested region -> the guest-sized
    // R32 destination (partial copies from a buffer are unrestricted).
    // All dims/rects here are host units (guest through HostDim/HostCoord)
    // so source and destination match scales.
    const uint32_t host_w = HostDim(src_w);
    const uint32_t host_h = HostDim(src_h);
    // Shader resolve: render the region instead of staging the plane.
    // One read pass + one write pass, versus the copy path's two full-plane
    // trips (the D3D12 depth-copy rule forbids partial copies out of a depth
    // resource, which is what forces the staging buffer).
    if (REXCVAR_GET(native_video_depth_resolve_shader) && g_blitVs &&
        dest->resolve_fb == nullptr && dest->rt_capable) {
      const RenderTexture* color = dest->texture.get();
      RenderFramebufferDesc fbDesc;
      fbDesc.colorAttachments = &color;
      fbDesc.colorAttachmentsCount = 1;
      fbDesc.depthAttachment = nullptr;
      dest->resolve_fb = g_device->createFramebuffer(fbDesc);
    }
    if (REXCVAR_GET(native_video_depth_resolve_shader) && rt && g_blitVs &&
        dest->resolve_fb) {
      // Lazy SRV + bindless slot for the RT's depth plane (kDepthFormat's
      // view specializes to the R32-class depth-plane SRV in plume).
      if (!rt->depth_descriptor &&
          g_nextTextureDescriptor < kFirstReservedDescriptor) {
        rt->depth_srv = rt->depth->createTextureView(
            RenderTextureViewDesc::Texture2DArray(kDepthFormat));
        rt->depth_descriptor = g_nextTextureDescriptor++;
        g_textureDescriptorSet->setTexture(rt->depth_descriptor,
                                           rt->depth.get(),
                                           RenderTextureLayout::SHADER_READ,
                                           rt->depth_srv.get());
      }
      if (!g_depthResolvePs) {
        g_depthResolvePs = g_device->createShader(
            kDepthResolvePsDxil, sizeof(kDepthResolvePsDxil), "main",
            RenderShaderFormat::DXIL);
      }
      if (!g_depthResolvePipeline && g_depthResolvePs) {
        RenderGraphicsPipelineDesc pd;
        pd.pipelineLayout = g_pipelineLayout.get();
        pd.vertexShader = g_blitVs.get();
        pd.pixelShader = g_depthResolvePs.get();
        pd.renderTargetFormat[0] = RenderFormat::R32_FLOAT;
        pd.renderTargetCount = 1;
        pd.depthTargetFormat = RenderFormat::UNKNOWN;
        pd.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
        pd.depthEnabled = false;
        pd.depthWriteEnabled = false;
        pd.cullMode = RenderCullMode::NONE;
        g_depthResolvePipeline = g_device->createGraphicsPipeline(pd);
      }
      if (rt->depth_descriptor && g_depthResolvePipeline) {
        const auto rcs = [&](int32_t v) { return HostCoord(v); };
        // Source must be readable, destination writable.
        if (rt->depth_layout != RenderTextureLayout::SHADER_READ) {
          commandList->barriers(
              RenderBarrierStage::GRAPHICS,
              RenderTextureBarrier(rt->depth.get(),
                                   RenderTextureLayout::SHADER_READ));
          rt->depth_layout = RenderTextureLayout::SHADER_READ;
        }
        commandList->barriers(
            RenderBarrierStage::GRAPHICS,
            RenderTextureBarrier(dest->texture.get(),
                                 RenderTextureLayout::COLOR_WRITE));
        dest->layout = RenderTextureLayout::COLOR_WRITE;
        commandList->setFramebuffer(dest->resolve_fb.get());
        const int32_t dx0 = rcs(dest_x), dy0 = rcs(dest_y);
        const int32_t dw = rcs(dest_x + copy_w) - dx0;
        const int32_t dh = rcs(dest_y + copy_h) - dy0;
        commandList->setViewports(RenderViewport(float(dx0), float(dy0),
                                                 float(dw), float(dh), 0.0f,
                                                 1.0f));
        commandList->setScissors(RenderRect(dx0, dy0, dx0 + dw, dy0 + dh));
        commandList->setPipeline(g_depthResolvePipeline.get());
        commandList->setGraphicsPipelineLayout(g_pipelineLayout.get());
        uint32_t idx[32] = {};
        idx[0] = rt->depth_descriptor;
        const int32_t delta_x = rcs(src_left) - dx0;
        const int32_t delta_y = rcs(src_top) - dy0;
        std::memcpy(&idx[4], &delta_x, 4);   // b4[1].x
        std::memcpy(&idx[5], &delta_y, 4);   // b4[1].y
        const uint32_t src_hw = HostDim(src_w);
        const uint32_t src_hh = HostDim(src_h);
        idx[8] = src_hw;                     // b4[2].x
        idx[9] = src_hh;                     // b4[2].y
        UploadAllocation b4 = g_upload[g_frame].Allocate(sizeof(idx), 256);
        std::memcpy(b4.memory, idx, sizeof(idx));
        commandList->setGraphicsRootDescriptor(b4.buffer->at(b4.offset),
                                               kRootB4IndicesPs);
        commandList->drawInstanced(3, 1, 0, 0);
        g_depthShaderResolves++;
        // Back to sampleable; the RT rebinds on the next draw.
        commandList->barriers(
            RenderBarrierStage::GRAPHICS,
            RenderTextureBarrier(dest->texture.get(),
                                 RenderTextureLayout::SHADER_READ));
        dest->layout = RenderTextureLayout::SHADER_READ;
        dest->host_rendered = true;
        dest->last_resolve_frame = g_frameIndex + 1;
        g_activeRtKey = ~0ull;  // force a framebuffer rebind before drawing
        g_resolveCopies++;
        return;
      }
    }
    const uint32_t row_pitch = (host_w * 4 + 255) & ~255u;
    const uint64_t needed = uint64_t(row_pitch) * host_h;
    // Plane-copy memo: another title's shadow map resolves 4 bands from one
    // unchanged render (tiling flattened), so the whole-plane copy the D3D12
    // depth rule forces need happen only for the first band. Any draw or
    // clear bumps g_rtMutationSeq and forces a fresh copy, so this cannot
    // serve stale depth.
    RenderBuffer* plane = nullptr;
    if (g_depthStageMemo.src == src && g_depthStageMemo.buffer &&
        g_depthStageMemo.frame == g_frameIndex &&
        g_depthStageMemo.mutation == g_rtMutationSeq &&
        g_depthStageMemo.host_w == host_w &&
        g_depthStageMemo.host_h == host_h &&
        g_depthStageMemo.row_pitch == row_pitch) {
      plane = g_depthStageMemo.buffer;  // already staged this frame
      g_depthStageReuses++;
    } else {
      DepthStagingSlot& slot = g_depthStaging[g_depthStagingRot++ % 8];
      if (!slot.buffer || slot.size < needed) {
        slot.buffer =
            g_device->createBuffer(RenderBufferDesc::DefaultBuffer(needed));
        slot.buffer->setName(
            fmt::format("depth staging {}", (g_depthStagingRot - 1) % 8));
        slot.size = needed;
      }
      plane = slot.buffer.get();
      // Explicit write-state barrier: without it the first copy implicitly
      // promotes the buffer to COPY_DEST while plume's tracked state stays
      // stale; the read barrier below then emits a wrong-Before transition
      // that D3D12 ignores, and the write and read copies race the same
      // bytes unsynchronized. AMD's copy engine serializes anyway; Intel's
      // hangs the device.
      commandList->barriers(RenderBarrierStage::COPY,
                            RenderBufferBarrier(plane,
                                                RenderBufferAccess::WRITE));
      commandList->copyTextureRegion(
          RenderTextureCopyLocation::PlacedFootprint(
              plane, RenderFormat::R32_FLOAT, host_w, host_h,
              1, row_pitch / 4, 0),
          RenderTextureCopyLocation::Subresource(src, 0));
      commandList->barriers(RenderBarrierStage::COPY,
                            RenderBufferBarrier(plane,
                                                RenderBufferAccess::READ));
      g_depthStageMemo = {src,    plane,  g_frameIndex, g_rtMutationSeq,
                          host_w, host_h, row_pitch};
    }
    const auto rc = [&](int32_t v) { return HostCoord(v); };
    RenderBox box(rc(src_left), rc(src_top), rc(src_left + copy_w),
                  rc(src_top + copy_h));
    // Sub-pixel after a downscale: nothing to copy (empty boxes are a D3D12
    // error, not a no-op).
    if (box.right > box.left && box.bottom > box.top) {
      commandList->copyTextureRegion(
          RenderTextureCopyLocation::Subresource(dest->texture.get(), 0),
          RenderTextureCopyLocation::PlacedFootprint(
              plane, RenderFormat::R32_FLOAT, host_w, host_h,
              1, row_pitch / 4, 0),
          uint32_t(rc(dest_x)), uint32_t(rc(dest_y)), 0, &box);
    }
  } else {
    const auto rc = [&](int32_t v) { return HostCoord(v); };
    RenderBox box(rc(src_left), rc(src_top), rc(src_left + copy_w),
                  rc(src_top + copy_h));
    if (box.right > box.left && box.bottom > box.top) {
      commandList->copyTextureRegion(
          RenderTextureCopyLocation::Subresource(dest->texture.get(), 0),
          RenderTextureCopyLocation::Subresource(src, 0),
          uint32_t(rc(dest_x)), uint32_t(rc(dest_y)), 0, &box);
    }
  }

  // Exit transitions. Dest -> SHADER_READ stays eager (draw-time sampling
  // has no on-demand transition hook). The source's return to render state
  // is the lazy candidate, except when it is the active render target:
  // ApplyRenderTargetState early-returns on an unchanged RT key, so a
  // mislaid layout there would never heal and the next draws would render
  // into a COPY_SOURCE surface.
  uint32_t exit_count = 0;
  RenderTextureBarrier exits[2];
  if (!lazy) {
    exits[exit_count++] =
        RenderTextureBarrier(src, depth ? RenderTextureLayout::DEPTH_WRITE
                                        : RenderTextureLayout::COLOR_WRITE);
    g_resolveBarriersEmitted++;
    if (rt) {
      if (depth) {
        rt->depth_layout = RenderTextureLayout::DEPTH_WRITE;
      } else {
        rt->color_layout[rt_index] = RenderTextureLayout::COLOR_WRITE;
      }
    }
  } else if (rt && RtKey(rt) == g_activeRtKey) {
    // Mid-pass resolve from the active RT: leave the layout where it is;
    // the next draw / same-key rebind repairs it on demand.
    g_activeRtLayoutsDirty = true;
  }
  exits[exit_count++] = RenderTextureBarrier(dest->texture.get(),
                                             RenderTextureLayout::SHADER_READ);
  g_resolveBarriersEmitted++;
  commandList->barriers(RenderBarrierStage::GRAPHICS, exits, exit_count);
  dest->layout = RenderTextureLayout::SHADER_READ;
  dest->last_resolve_frame = g_frameIndex + 1;
  dest->resolve_serial++;
  // Resolve write-back queue: guest-visible resolve bytes (see the flush
  // above EndFrameAndPresent). 32bpp color only; delivered at the guest's
  // next fence wait.
  const bool wb_color =
      !depth && (required == RenderFormat::R8G8B8A8_UNORM ||
                 required == RenderFormat::R10G10B10A2_UNORM);
  const bool wb_depth = depth && REXCVAR_GET(native_video_depth_writeback);
  const int32_t wb_max_dim = REXCVAR_GET(native_video_writeback_max_dim);
  if ((wb_color || wb_depth) &&
      REXCVAR_GET(native_video_resolve_writeback) && copy_w > 0 &&
      copy_h > 0 && g_resolveWritebacks.size() < 512 && !Scaled() &&
      (wb_max_dim == 0 || (int32_t(dest->width) <= wb_max_dim &&
                           int32_t(dest->height) <= wb_max_dim))) {
    const uint32_t dw0 = LoadGuestU32(dest_texture_header + 0);
    const uint32_t dw1 = LoadGuestU32(dest_texture_header + 4);
    // Snapshot the rect now into the readback arena (ping-pong contract:
    // a deferred flush must deliver this frame's bytes, not whatever the
    // shared host RT holds by then). Grow flushes first so pending
    // records never reference a freed buffer.
    const uint32_t snap_pitch = (uint32_t(copy_w) * 4u + 255u) & ~255u;
    // 512-align the arena slot size: D3D12 placed-footprint offsets require
    // D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT (512). Rounding to 256 is not
    // enough: a slot size that is a 256- but not 512-multiple misaligns
    // every second record's offset and CopyTextureRegion rejects it.
    const uint32_t snap_need =
        (snap_pitch * uint32_t(copy_h) + 511u) & ~511u;
    // Never flush mid-frame: the flush ends the open command list and
    // recording would continue into an ended list. Mid-frame exhaustion
    // drops this record; the resolve itself still happens. Each frame
    // slot owns cap/kNumFrames arena bytes so a deferred bucket's
    // snapshot survives until its slot-reuse delivery.
    const uint32_t wb_seg =
        g_wbSnapBuffer ? (g_wbSnapCap / kNumFrames) & ~511u : 0;
    bool arena_ok = true;
    if (!g_wbSnapBuffer || g_wbSnapUsedSlot[g_frame] + snap_need > wb_seg) {
      if (g_frameOpen && g_wbSnapBuffer) {
        arena_ok = false;
        static std::atomic<uint32_t> wb_dropped{0};
        const uint32_t n = wb_dropped.fetch_add(1, std::memory_order_relaxed);
        if (n == 0 || (n & 0xFF) == 0) {
          REXGPU_WARN(
              "videonative: [writeback] arena segment full mid-frame, "
              "record dropped ({} total; raise the arena or lower "
              "writeback_max_dim)",
              n + 1);
        }
      } else {
        FlushResolveWritebacksInline();
        if (g_wbSnapCap / kNumFrames < snap_need * 2) {
          g_wbSnapCap =
              std::max(8u * 1024u * 1024u, snap_need * 2) * kNumFrames;
          g_wbSnapBuffer.reset();
          g_wbSnapBuffer = g_device->createBuffer(
              RenderBufferDesc::ReadbackBuffer(g_wbSnapCap));
        }
      }
    }
    if (arena_ok) {
    const uint32_t snap_base =
        g_frame * ((g_wbSnapCap / kNumFrames) & ~511u) +
        g_wbSnapUsedSlot[g_frame];
    commandList->barriers(RenderBarrierStage::COPY,
                          RenderTextureBarrier(
                              dest->texture.get(),
                              RenderTextureLayout::COPY_SOURCE));
    RenderBox snap_box(int32_t(dest_x), int32_t(dest_y),
                       int32_t(dest_x + copy_w), int32_t(dest_y + copy_h));
    commandList->copyTextureRegion(
        RenderTextureCopyLocation::PlacedFootprint(
            g_wbSnapBuffer.get(), required, uint32_t(copy_w),
            uint32_t(copy_h), 1, snap_pitch / 4, snap_base),
        RenderTextureCopyLocation::Subresource(dest->texture.get(), 0), 0, 0,
        0, &snap_box);
    commandList->barriers(RenderBarrierStage::GRAPHICS,
                          RenderTextureBarrier(
                              dest->texture.get(),
                              RenderTextureLayout::SHADER_READ));
    g_resolveWritebacks.push_back(ResolveWriteback{
        required, dw1 & 0xFFFFF000u, ((dw0 >> 22) & 0x1FFu) * 32u,
        (dw1 >> 6) & 3u, ((dw0 >> 31) & 1u) != 0, dest->width, dest->height,
        uint32_t(dest_x), uint32_t(dest_y), uint32_t(copy_w),
        uint32_t(copy_h), snap_base, snap_pitch, wb_depth,
        false,
        dest_texture_header});
    {
      ResolveWriteback& rec = g_resolveWritebacks.back();
      if (!QueryDestAllocation(rec.base_address, &rec.alloc_base, &rec.alloc_size,
                               &rec.alloc_state)) {
        rec.alloc_base = rec.alloc_size = rec.alloc_state = 0;
      }
    }
    g_wbSnapUsedSlot[g_frame] += snap_need;
    NoteWritebackBase(dw1 & 0xFFFFF000u);
    UpdateWritebackPending();
    // Early submit: the snapshot copy just recorded is what the guest's
    // imminent fence wait needs executed, hand everything so far to the
    // GPU now so that wait covers an already-running sliver.
    if (REXCVAR_GET(native_video_early_submit)) {
      EarlySubmitSegment();
    }
    // Class-split delivery: hub-plate-class dests (everything except the
    // 192x256 grid-tile class) bake at menu transitions only and their
    // consumers read the pages immediately; deferred delivery races them
    // both ways. Deliver those inline at the resolve; the scroll-churn
    // 192x256 class is GPU-consumed and stays deferred.
    if (!(dest->width == 192 && dest->height == 256)) {
      FlushResolveWritebacksInline();
    }
    }
  }
  if (!depth) {
    g_lastResolvedBySize[SizeKeyOf(dest->width, dest->height)] = dest;
  }
  g_resolveCopies++;
}
}  // namespace

void ResolveToTexture(uint32_t dest_texture_header, uint32_t source,
                      bool has_rect, int32_t src_left, int32_t src_top,
                      int32_t src_right, int32_t src_bottom, int32_t dest_x,
                      int32_t dest_y) {
  if (rq::Active()) { rq::EnqResolveToTexture(dest_texture_header, source, has_rect, src_left, src_top, src_right, src_bottom, dest_x, dest_y); return; }
  ResolveRegion(dest_texture_header, false, source, has_rect, src_left,
                src_top, src_right, src_bottom, dest_x, dest_y);
}

void ResolveDepthToTexture(uint32_t dest_texture_header, bool has_rect,
                           int32_t src_left, int32_t src_top,
                           int32_t src_right, int32_t src_bottom,
                           int32_t dest_x, int32_t dest_y) {
  if (rq::Active()) { rq::EnqResolveDepthToTexture(dest_texture_header, has_rect, src_left, src_top, src_right, src_bottom, dest_x, dest_y); return; }
  ResolveRegion(dest_texture_header, true, 0, has_rect, src_left, src_top,
                src_right, src_bottom, dest_x, dest_y);
}

// ---------------------------------------------------------------------------
// Draws
// ---------------------------------------------------------------------------

namespace {

struct DrawSetup {
  const ResolvedShader* vs;
  const ResolvedShader* ps;
  RenderPrimitiveTopology topology;
  bool rect = false;
  bool point = false;
};

bool TranslateTopology(uint32_t prim_type, RenderPrimitiveTopology* out,
                       bool* rect, bool* point) {
  *rect = false;
  *point = false;
  switch (prim_type) {
    case 4: *out = RenderPrimitiveTopology::TRIANGLE_LIST; return true;
    case 6: *out = RenderPrimitiveTopology::TRIANGLE_STRIP; return true;
    case 2: *out = RenderPrimitiveTopology::LINE_LIST; return true;
    case 3: *out = RenderPrimitiveTopology::LINE_STRIP; return true;
    case 1:  // POINTLIST: sprites expanded by the pack's point GS (sized by
             // the VS oPts export, the deferred sprite-light path).
      *out = RenderPrimitiveTopology::POINT_LIST;
      *point = true;
      return true;
    case 8:  // RECTLIST: triangles expanded by the pack's geometry shader.
      *out = RenderPrimitiveTopology::TRIANGLE_LIST;
      *rect = true;
      return true;
    default: return false;  // quadlist(13): TODO expand
  }
}

// Uploads vertex data for the draw and writes the vertex fetch constant pair
// into the fetch constant array (dwords for vfetch constant 95-stream).
void BindStreamData(uint32_t* fetch_dwords, uint32_t stream,
                    const void* vertex_data, uint32_t size_bytes,
                    bool immediate = false, uint32_t stride = 0,
                    uint32_t guest_key = 0) {
  // Guard against garbage VB sizes (stale/bogus guest objects): anything
  // that cannot fit the ring would run the memcpy off the mapped buffer.
  // Skip the stream; the draw reads the blank fetch.
  if (vertex_data == nullptr || size_bytes == 0 ||
      size_bytes > kUploadBufferSize / 2) {
    static uint64_t bad_stream_logs = 8;
    if (bad_stream_logs) {
      bad_stream_logs--;
      REXGPU_WARN(
          "videonative: stream {} size {} bytes implausible/unreadable, "
          "skipped",
          stream, size_bytes);
    }
    return;
  }
  // Identity component of the cache key: the guest address when known.
  // Under the render queue, vertex_data may be an enqueue-time snapshot in
  // the arena whose host pointer changes every draw, so keying on it would
  // defeat the upload cache and exhaust the per-frame vertex ring.
  const struct {
    uint64_t ptr;
    uint32_t size;
  } ck{guest_key != 0
           ? uint64_t(guest_key)
           : uint64_t(reinterpret_cast<uintptr_t>(vertex_data)),
       size_bytes};
  // UP draws source shared scratch that is rewritten between draws, so a
  // (pointer,size) key collides for same-size batches; uploads are keyed
  // by content instead. Stream VBs too (pooled meshes); large buffers
  // hash head+middle+tail samples to keep the per-draw cost trivial.
  uint64_t cache_key;
  if (immediate) {
    cache_key = XXH3_64bits(vertex_data, size_bytes);
  } else if (size_bytes <= 64 * 1024) {
    cache_key = XXH3_64bits(vertex_data, size_bytes) ^ ck.ptr;
  } else {
    const uint8_t* p = static_cast<const uint8_t*>(vertex_data);
    constexpr uint32_t kWin = 16 * 1024;
    XXH3_state_t* st = XXH3_createState();
    XXH3_64bits_reset(st);
    XXH3_64bits_update(st, &ck, sizeof(ck));
    XXH3_64bits_update(st, p, kWin);
    XXH3_64bits_update(st, p + (size_bytes / 2 & ~15u), kWin);
    XXH3_64bits_update(st, p + size_bytes - kWin, kWin);
    cache_key = XXH3_64bits_digest(st);
    XXH3_freeState(st);
  }
  uint32_t ring_offset = 0xFFFFFFFFu;
  g_lastBindRingOffset = 0xFFFFFFFF;
  // Tier 1: persistent region for stream VBs (identity-keyed content). The
  // cache key embeds the content hash, so a mutated buffer misses here by
  // construction; the demotion map keeps streaming pools from leaking the
  // region one orphan at a time.
  if (!immediate && guest_key && REXCVAR_GET(native_video_vb_persist)) {
    auto git = g_vbPersistByGuest.find(guest_key);
    const bool demoted =
        git != g_vbPersistByGuest.end() && git->second.changes >= 2;
    if (!demoted) {
      auto pit = g_vbPersistCache.find(cache_key);
      if (pit != g_vbPersistCache.end()) {
        ring_offset = pit->second;
        g_vbPersistHits++;
      } else {
        PersistGuestInfo& gi = git != g_vbPersistByGuest.end()
                                   ? git->second
                                   : g_vbPersistByGuest[guest_key];
        if (gi.last_key && gi.last_key != cache_key) {
          gi.changes++;
          g_vbPersistOrphans += gi.size;  // that resident alloc is now dead
          if (gi.changes >= 2) g_vbPersistDemoted++;
        }
        if (gi.changes < 2) {
          const uint32_t poff = PersistAlloc(size_bytes);
          if (poff != 0xFFFFFFFFu) {
            bool stored = false;
            VertexRing& cur = g_vertexRing[g_frame];
            if (cur.gpu_local) {
              // One staging write, copied into every parity's GPU buffer.
              // Same-queue ordering makes the cross-parity copy safe: the
              // other parity's in-flight reads completed before this list
              // executes, and poff is fresh in all buffers.
              cur.staging_offset = (cur.staging_offset + 15) & ~15u;
              if (cur.staging_offset + size_bytes <= kVbStagingSize) {
                const uint32_t soff = cur.staging_offset;
                std::memcpy(cur.staging_memory + soff, vertex_data,
                            size_bytes);
                cur.staging_offset += size_bytes;
                auto* cl = g_commandLists[g_frame].get();
                for (uint32_t i = 0; i < kNumFrames; i++) {
                  g_vertexRing[i].ToCopyState(cl);
                  cl->copyBufferRegion(g_vertexRing[i].buffer->at(poff),
                                       cur.staging->at(soff), size_bytes);
                }
                for (uint32_t i = 0; i < kNumFrames; i++) {
                  if (i != uint32_t(g_frame)) g_vertexRing[i].ToReadState(cl);
                }
                stored = true;
              }
            } else {
              for (uint32_t i = 0; i < kNumFrames; i++) {
                std::memcpy(g_vertexRing[i].memory + poff, vertex_data,
                            size_bytes);
              }
              stored = true;
            }
            if (stored) {
              g_vbPersistCache.emplace(cache_key, poff);
              gi.last_key = cache_key;
              gi.size = size_bytes;
              g_vbPersistUploads++;
              ring_offset = poff;
            }
          }
        }
      }
    }
  }
  // Tier 2: the per-frame ring (UP/immediate data, demoted dynamic VBs,
  // persistent-region overflow).
  if (ring_offset == 0xFFFFFFFFu) {
  auto cached = g_vbUploadCache.find(cache_key);
  if (cached != g_vbUploadCache.end()) {
    ring_offset = cached->second;
  } else {
    UploadAllocation alloc = g_vertexRing[g_frame].Allocate(size_bytes);
    if (!alloc.memory) {
      static uint64_t ring_full_logs = 8;
      if (ring_full_logs) {
        ring_full_logs--;
        REXGPU_WARN(
            "videonative: vertex ring EXHAUSTED (stream {} size {}) - "
            "draw will fetch stale data",
            stream, size_bytes);
      }
      return;
    }
    // Render-queue mode: enqueue snapshots already give hardware-time
    // bytes, and present-time copies would outlive their arena backing,
    // copy at draw time.
    if (immediate || rq::Running() ||
        !REXCVAR_GET(native_video_deferred_vb_copy)) {
      std::memcpy(alloc.memory, vertex_data, size_bytes);
    } else {
      g_deferredCopies.push_back({vertex_data, alloc.memory, size_bytes, 0});
    }
    // gpu-local: the bytes above went to staging, record the copy into
    // the GPU buffer before the draw that fetches them (no-op otherwise).
    g_vertexRing[g_frame].CommitCopy(g_commandLists[g_frame].get(),
                                     uint32_t(alloc.offset),
                                     g_vertexRing[g_frame].staging_offset,
                                     size_bytes);
    ring_offset = uint32_t(alloc.offset);
    g_vbUploadCache.emplace(cache_key, ring_offset);
  }
  }
  g_lastBindRingOffset = ring_offset;
  g_drawVfetchWrittenMask |= 1u << stream;
  const uint32_t vfetch_index = 95 - stream;
  // xe_gpu_vertex_fetch_t: dword0 = byte address (low 2 bits = type, masked
  // off by the shader) | dword1 = endian:2 + size-in-words:24. Endian 2 =
  // 8-in-32 (raw guest big-endian data in the ring).
  fetch_dwords[vfetch_index * 2 + 0] = ring_offset | 3u;
  fetch_dwords[vfetch_index * 2 + 1] = 2u | (((size_bytes + 3) >> 2) << 2);
}

static const void* g_upOverrideVertexData = nullptr;
static const void* g_upOverrideIndexData = nullptr;

bool SetupDraw(uint32_t device, uint32_t prim_type, DrawSetup* setup,
               uint32_t up_vertex_data, uint32_t up_size_bytes,
               uint32_t up_stride_bytes, int32_t vertex_offset) {
  g_currentDevice = device;  // RB shadow fold source (ApplyRenderTargetState)
  EnsureFrameOpen();
  if (!g_frameOpen) {
    g_frameOpenFails++;  // silent whole-draw drops
    return false;
  }
  RepairActiveRtLayouts();  // lazy mid-pass resolves leave write on demand
  g_drawsThisFrame++;
  g_rtMutationSeq++;  // invalidates the depth-staging plane memo

  setup->vs = detail::CurrentResolvedVertexShader();
  setup->ps = detail::CurrentResolvedPixelShader();
  if (!setup->vs || !setup->vs->pack) {
    g_drawsSkipped++;
    g_skipNoVs++;
    return false;
  }
  if (!setup->ps || !setup->ps->pack) {
    if (detail::CurrentDrawPixelShaderBound()) {
      // A PS is bound but did not resolve (pack miss), drawing would be
      // wrong; skip.
      g_drawsSkipped++;
      g_skipNoPs++;
      return false;
    }
    // No PS bound at all: deliberate depth-only draw (PS-less pipeline).
    setup->ps = nullptr;
  }
  if (!TranslateTopology(prim_type, &setup->topology, &setup->rect,
                         &setup->point)) {
    g_drawsSkipped++;
    g_skipTopology++;
    if (prim_type < 32) g_skipPrimTypes[prim_type]++;
    return false;
  }
  // model_shadows off: drop the avatar's depth-twin draws. The pass's own
  // Z clear and resolve still run, so the shadow map reads fully lit. The
  // only other PS-less draw is the fullscreen depth-restore rect, which
  // TranslateTopology marks as rect.
  if (!setup->ps && !setup->rect && !setup->point &&
      !REXCVAR_GET(model_shadows)) {
    g_drawsSkipped++;
    return false;
  }

  // Target the framebuffer of the bound guest surfaces (must precede pipeline
  // creation, the PSO's RT format depends on it).
  ApplyRenderTargetState();
  // Bracket-less tiled pass: the draw's viewport is taller than the
  // bound host RT, so post the viewport as the wanted extent and re-apply
  // (the RT grows to viewport height). Guarded to that exact shape, so
  // screen passes never take this path.
  if (REXCVAR_GET(native_video_vp_grow) && g_activeRtKey &&
      g_state.viewport_h > g_activeRtHeight &&
      g_state.viewport_h <= 4096 && g_state.viewport_w <= 4096 &&
      g_state.viewport_x == 0 && g_state.viewport_y == 0) {
    const SurfaceInfo cinfo = ParseSurface(g_state.color_surface[0]);
    const SurfaceInfo dinfo = ParseSurface(g_state.depth_surface);
    const SurfaceInfo& binfo = cinfo.valid ? cinfo : dinfo;
    if (binfo.valid && binfo.width >= g_state.viewport_w &&
        binfo.width <= g_state.viewport_w + 64 &&
        binfo.height < g_state.viewport_h) {
      g_growWidth = g_state.viewport_w;
      g_growHeight = g_state.viewport_h;
      g_growSurface = cinfo.valid ? g_state.color_surface[0]
                                  : g_state.depth_surface;
      g_activeRtKey = ~0ull;  // force the re-apply to run the expansion
      ApplyRenderTargetState();
      auto& cl = g_commandLists[g_frame];
      cl->clearDepthStencil(true, true, 1.0f, 0, nullptr, 0);
      cl->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f), nullptr, 0);
      g_passResolves = 0;
    }
  }
  (g_activeRtKey ? g_drawsToRt : g_drawsToSwapchain)++;
  if (g_activeRtKey) {
    auto rt_it = g_renderTargets.find(g_activeRtKey);
    if (rt_it != g_renderTargets.end()) {
      rt_it->second.draws++;
      rt_it->second.last_draw_serial = g_drawsToRt;
    }
  }
  g_lastDrawnRtKey = g_activeRtKey;


  // Render state from the shadow registers the recompiled XDK setters keep in
  // the guest device block (register 0x2200-block at dev+10548, 0x2100-block
  // at dev+10444). For command-buffer replays the state captured on the
  // recording device at record time overrides the live shadows, the game
  // sets scene depth/cull/blend during recording, which never touches the
  // main device's block.
  uint32_t depth_control, blend_control[kMrtCount], color_control, cull_bits,
      color_mask, alpha_ref_bits;
  bool z_enable;
  // The game programs blend live between RunCommandBuffer calls and
  // hardware executes those writes before the CB's packets, so
  // RunCommandBuffer consumes the live dirty bits once at Run start and
  // draws consume only on the live branch (mid-replay there is no new
  // live programming).
  if (!g_replayDrawStateValid) {
    ConsumeLiveStateDirty(device);
  }
  if (g_replayDrawStateValid) {
    depth_control = g_replayDrawState[0];
    blend_control[0] = g_replayDrawState[1];
    color_control = g_replayDrawState[2];
    cull_bits = g_replayDrawState[3] & 7;
    color_mask = g_replayDrawState[4];
    alpha_ref_bits = g_replayDrawState[5];
    z_enable = g_replayDrawState[6] != 0;  // composed at record time
    // Recorded blend/mask slots the game programmed on the recording
    // device (CbCall::st_dirty) update the persistent effective register
    // file below and persist after the command buffer ends, exactly like
    // the PM4 register writes a CB leaves behind on hardware. Untouched
    // slots leave the effective state alone (they were never baked into
    // the CB).
    if (g_replayDrawStateDirty & (1u << 1))
      g_fxBlendControl[0] = g_replayDrawState[1];
    if (g_replayDrawStateDirty & (1u << 4))
      g_fxColorMask = g_replayDrawState[4];
    if (g_replayDrawStateDirty & (1u << 7))
      g_fxBlendControl[1] = g_replayDrawState[7];
    if (g_replayDrawStateDirty & (1u << 8))
      g_fxBlendControl[2] = g_replayDrawState[8];
    if (g_replayDrawStateDirty & (1u << 9))
      g_fxBlendControl[3] = g_replayDrawState[9];
  } else {
    depth_control = LoadGuestU32(device + 10548);    // RB_DEPTHCONTROL 0x2200
    color_control = LoadGuestU32(device + 10556);   // RB_COLORCONTROL 0x2202
    // PA_SU_SC_MODE_CNTL, low 3 bits only. Bits 15/16 (msaa, window
    // offset) are the hardware's per-band EDRAM windowing; the native
    // model renders the full frame at true positions, so honoring the
    // window offset would break it.
    cull_bits = LoadGuestU32(device + 10568) & 7;   // PA_SU_SC_MODE_CNTL
    alpha_ref_bits = LoadGuestU32(device + 10500);  // RB_ALPHA_REF 0x210E
    // Z_ENABLE: trust the guest's composed bit; with the guest bodies
    // running, dev+10548 bit1 is exactly what hardware gets, and
    // recomposing it here is wrong for passes that inherit their DS
    // binding. The raw request stays as a floor: if the guest never asked
    // for Z, never force it on.
    z_enable = (depth_control & 2u) != 0 ||
               ((LoadGuestU32(device + 12308) & 1u) != 0 &&
                g_state.depth_surface != 0);
    // XDK live-draw invariant: every state group is re-dirtied after a
    // RunCommandBuffer, so live draws take blend/mask straight from the
    // live shadow and re-sync the persistent register file to it.
    g_fxBlendControl[0] = LoadGuestU32(device + 10552);
    g_fxBlendControl[1] = LoadGuestU32(device + 10584);
    g_fxBlendControl[2] = LoadGuestU32(device + 10588);
    g_fxBlendControl[3] = LoadGuestU32(device + 10592);
    // The composed RB_COLOR_MASK shadow (dev+10460 on this XDK) gates each
    // nibble on the D3D9-style API's current-shader slots, which the XUI
    // path never populates; the composed value then reads 0/4 and masks
    // every native draw. The raw ColorWriteEnable request (dev+12036, the
    // SetRenderState store) carries the app's intent instead; PS-less draws
    // are already skipped. Offset 0 keeps the default slot.
    const uint32_t cm_off = REXCVAR_GET(native_video_colormask_offset);
    g_fxColorMask = LoadGuestU32(device + (cm_off ? cm_off : 10580));
    if (cm_off) g_fxColorMask &= 0xF;  // raw request is RT0-only
  }
  // Both branches: draws consume the effective register file (live draws
  // just re-synced it to the device shadow; replay draws see the recorded
  // overlays + chained-Run state).
  for (uint32_t i = 0; i < kMrtCount; i++) blend_control[i] = g_fxBlendControl[i];
  color_mask = g_fxColorMask;
  depth_control = (depth_control & ~2u) | (z_enable ? 2u : 0u);
  // Bake-source coherence: bake passes sample textures the CPU wrote
  // moments earlier, and the cache can serve a pre-decode upload instead.
  // Bakes are rare, so retire the source's guest-upload entry at every
  // bake-class draw; the binding that follows re-uploads current pages.
  if (g_activeRtWidth && g_activeRtWidth <= 256 && g_activeRtHeight <= 256) {
    const uint32_t src_dw1 = LoadGuestU32(device + 1152 + 4);
    const uint32_t src_base = src_dw1 & 0xFFFFF000u;
    if (src_base && !IsGuestRangeFrozen(src_base, 2048u)) {
      // Content-gated: retire only when the pages differ from the
      // entry's last-known hash (computed with the heal's own span
      // algorithm), so stable assets are never disturbed.
      for (auto itc = g_textureCache.begin(); itc != g_textureCache.end();) {
        CachedTexture& e = itc->second;
        if (!e.host_rendered && e.valid && e.width <= 256 &&
            e.height <= 256 && (e.header_dw[1] & 0xFFFFF000u) == src_base) {
          const uint32_t span = e.approx_bytes ? e.approx_bytes : 2048u;
          const uint8_t* pages = GuestDataPtrFast(src_base, span, false, true);
          if (pages) {
            uint64_t h;
            if (span <= 6144) {
              h = XXH3_64bits(pages, span);
            } else {
              XXH3_state_t* st = XXH3_createState();
              XXH3_64bits_reset(st);
              XXH3_64bits_update(st, pages, 2048);
              XXH3_64bits_update(st, pages + (span / 2 & ~15u), 2048);
              XXH3_64bits_update(st, pages + span - 2048, 2048);
              h = XXH3_64bits_digest(st);
              XXH3_freeState(st);
            }
            if (e.content_hash == 0) {
              e.content_hash = h;  // baseline stamped; retire on next change
            } else if (e.content_hash != h) {
              RetireEntry(e);
              itc = g_textureCache.erase(itc);
              g_texInvalidations++;
              g_bakeSrcRefreshes++;
              continue;
            }
          }
        }
        ++itc;
      }
    }
  }
  // A raw-zero RB_BLENDCONTROL decodes as src*zero + dst*zero, which
  // annihilates the output. The hardware/XDK default is 0x00010001
  // (one/zero = replace); RT1-3 shadows the game never writes read back as
  // 0 here, which would zero every MRT slot the game does not program.
  for (uint32_t i = 1; i < kMrtCount; i++) {
    if (blend_control[i] == 0) blend_control[i] = 0x00010001;
  }

  // Reversed-Z viewport (the avatar passes): the Xenos idiom for reversed
  // depth is an inverted viewport Z range (MinZ=1, MaxZ=0). The viewport
  // code below swaps the range and compensates the stored values via
  // ndc z' = 1-z, but the guest's depth compare still expects the reversed
  // convention, so the PSO's function must invert with it (the
  // less<->greater family) or the far side wins every test. bit4 keys the
  // flipped variant.
  if (g_state.viewport_min_z > g_state.viewport_max_z) cull_bits |= 16;

  // Stencil ref/masks: the XDK stores the D3D9 state as bytes at
  // dev+0x2901..03; read big-endian as one dword at +0x2900, which is
  // exactly RB_STENCILREFMASK's layout (ref 0:7, testmask 8:15,
  // writemask 16:23).
  const uint32_t stencil_ref_mask = LoadGuestU32(device + 0x2900);
  const CachedPipeline* pipeline =
      GetOrCreatePipeline(setup->vs, setup->ps, blend_control, depth_control,
                          cull_bits, setup->topology, setup->rect,
                          setup->point, color_mask, stencil_ref_mask);
  if (!pipeline || !pipeline->pipeline) {
    g_drawsSkipped++;
    g_skipPipeline++;
    return false;
  }

  auto& commandList = g_commandLists[g_frame];
  commandList->setPipeline(pipeline->pipeline.get());

  // Viewport + scissor. A reversed guest depth range (MinZ > MaxZ) is
  // illegal as a D3D12 viewport, so pass the sorted range and fold the
  // reversal into the shader-side z transform. Host depth then equals the
  // guest depth-buffer convention.
  float vp_min_z = std::clamp(g_state.viewport_min_z, 0.0f, 1.0f);
  float vp_max_z = std::clamp(g_state.viewport_max_z, 0.0f, 1.0f);
  float ndc_scale_z = 1.0f;
  float ndc_offset_z = 0.0f;
  if (vp_min_z > vp_max_z) {
    std::swap(vp_min_z, vp_max_z);
    ndc_scale_z = -1.0f;
    ndc_offset_z = 1.0f;
  }
  // Guest-unit viewport/scissor scaled to the host allocation. Swapchain-
  // targeted draws (no active RT) stay 1:1.
  const bool rt_scaled = g_activeRtKey != 0;
  const float vsc = rt_scaled ? HostScaleF() : 1.0f;
  const auto sc = [&](int32_t v) { return rt_scaled ? HostCoord(v) : v; };
  // Viewport-disable sentinel (XUI: SetRenderState_ViewportEnable(0) leaves
  // the viewport state at 65535x65535 and the hardware windows NDC to the
  // full target, while a literal 65535-wide host viewport squeezes the
  // frame into a sliver). Clamp absurd viewports to the bound target's
  // dimensions.
  uint32_t vp_x = g_state.viewport_x, vp_y = g_state.viewport_y;
  uint32_t vp_w = g_state.viewport_w, vp_h = g_state.viewport_h;
  {
    const uint32_t tgt_w = g_activeRtKey && g_activeRtWidth
                               ? g_activeRtWidth
                               : (g_swapChain ? g_swapChain->getWidth() : 1280);
    const uint32_t tgt_h = g_activeRtKey && g_activeRtHeight
                               ? g_activeRtHeight
                               : (g_swapChain ? g_swapChain->getHeight() : 720);
    if (vp_w > 8192 || vp_h > 8192) {
      // tgt_* are host dims; the shared path below multiplies by vsc, so
      // store guest-unit dims here (vsc rescales them back to host).
      // Band-widened RTs are tile-height (736), but the logical frame is
      // the swapchain height (720): the XUI ortho constants assume it, and
      // the band resolves only read rows 0..720.
      uint32_t eff_h = tgt_h;
      if (g_activeRtBandWidened && g_swapChain &&
          g_swapChain->getHeight() < eff_h) {
        eff_h = g_swapChain->getHeight();
      }
      vp_x = 0;
      vp_y = 0;
      vp_w = uint32_t(float(tgt_w) / vsc);
      vp_h = uint32_t(float(eff_h) / vsc);
      // Sentinel rule (see native_video_sentinel_vp_guest): guest units
      // already, no /vsc (the shared path below re-applies it).
      const SurfaceInfo si = ParseSurface(g_state.color_surface[0]);
      const bool bracket = g_tilingWidth != 0 && g_tilingHeight != 0;
      if (REXCVAR_GET(native_video_sentinel_vp_guest)) {
        if (bracket) {
          vp_w = g_tilingWidth;
          vp_h = g_tilingHeight;
        } else if (si.valid && si.width && si.height) {
          vp_w = si.width;
          vp_h = si.height;
        }
      }
    }
  }
  commandList->setViewports(RenderViewport(
      float(vp_x) * vsc, float(vp_y) * vsc, float(vp_w) * vsc,
      float(vp_h) * vsc, vp_min_z, vp_max_z));
  commandList->setScissors(
      RenderRect(sc(g_state.scissor[0]), sc(g_state.scissor[1]),
                 sc(g_state.scissor[2]), sc(g_state.scissor[3])));

  // b0: system constants.
  SystemConstants sys{};
  // W_NOT_RECIPROCAL (1<<3): the XDK standard path exports true clip W
  // (VTX_W0_FMT=1); without this every perspective VS reciprocates W and 3D
  // geometry turns inside out.
  sys.flags = kSysFlagWNotReciprocal;
  // Alpha test from RB_COLORCONTROL: bit3 = enable, bits 0-2 = compare func
  // (0 Never .. 7 always), mapped onto the pass-if-less/equal/greater flag
  // bits xe_alpha_test consumes (no pass bit set = discard everything, so
  // disabled alpha test must set all three). Foliage/leaf cards need this,
  // without it they render as solid quads.
  float alpha_ref = 0.0f;
  if (color_control & 0x8) {
    const uint32_t func = color_control & 0x7;
    if (func == 1 || func == 3 || func == 5) sys.flags |= kSysFlagAlphaPassIfLess;
    if (func == 2 || func == 3 || func == 6) sys.flags |= kSysFlagAlphaPassIfEqual;
    if (func == 4 || func == 5 || func == 6) sys.flags |= kSysFlagAlphaPassIfGreater;
    if (func == 7) {
      sys.flags |= kSysFlagAlphaPassIfLess | kSysFlagAlphaPassIfEqual |
                   kSysFlagAlphaPassIfGreater;
    }
    std::memcpy(&alpha_ref, &alpha_ref_bits, 4);
  } else {
    sys.flags |= kSysFlagAlphaPassIfLess | kSysFlagAlphaPassIfEqual |
                 kSysFlagAlphaPassIfGreater;
  }
  sys.line_loop_index = 0xFFFFFFFFu;
  sys.vertex_index_endian = 0;  // host-generated / pre-swapped indices
  sys.vertex_index_offset = vertex_offset;
  sys.vertex_index_min = 0;
  sys.vertex_index_max = 0xFFFFFFu;
  // Y not negated: unlike the ring path, the native present blit does not
  // flip V, so the guest's +Y-up NDC maps straight through.
  sys.ndc_scale[0] = 1.0f;
  sys.ndc_scale[1] = 1.0f;
  sys.ndc_scale[2] = ndc_scale_z;
  sys.ndc_offset[0] = 1.0f / float(g_state.viewport_w);
  // Half-pixel Y: the ring contract is -1/h with ndc_scale.y = +1 (c9.y =
  // -1/1024 for the shadow pass, -1/720 for the scene). A +1/h offset here
  // misregisters every pass by one pixel.
  sys.ndc_offset[1] = -1.0f / float(g_state.viewport_h);
  sys.ndc_offset[2] = ndc_offset_z;
  // Screen-space position mode (PA_CL_VTE_CNTL enables clear): the guest
  // VS outputs framebuffer-pixel positions, so map pixels to NDC through
  // the same guest-unit viewport the host draw sets and the transform
  // round-trips exactly.
  {
    const uint32_t vte_cntl = LoadGuestU32(device + 10572);
    if ((vte_cntl & 0x3Fu) == 0 && vp_w && vp_h) {
      sys.ndc_scale[0] = 2.0f / float(vp_w);
      sys.ndc_scale[1] = -2.0f / float(vp_h);
      sys.ndc_offset[0] = 1.0f / float(vp_w) - 1.0f;
      sys.ndc_offset[1] = 1.0f - 1.0f / float(vp_h);
    }
  }
  sys.alpha_test_reference = alpha_ref;
  if (setup->point) {
    // PA_SU_POINT_SIZE / PA_SU_POINT_MINMAX shadows (0x2200-block at
    // dev+10548; regs 0x2280/0x2281 -> +11060/+11064). Register fields are
    // half-size 12.4 fixed -> diameter px = field * 2 / 16.
    const uint32_t point_size = LoadGuestU32(device + 11060);
    const uint32_t point_minmax = LoadGuestU32(device + 11064);
    sys.point_constant_diameter[0] = float(point_size >> 16) * 0.125f;
    sys.point_constant_diameter[1] = float(point_size & 0xFFFF) * 0.125f;
    sys.point_vertex_diameter_min = float(point_minmax & 0xFFFF) * 0.125f;
    sys.point_vertex_diameter_max = float(point_minmax >> 16) * 0.125f;
  }
  sys.color_exp_bias[0] = sys.color_exp_bias[1] = sys.color_exp_bias[2] =
      sys.color_exp_bias[3] = 1.0f;
  if (REXCVAR_GET(native_video_write_exp_bias)) {
    for (uint32_t i = 0; i < kMrtCount; i++) {
      const uint32_t surf = g_state.color_surface[i];
      if (!surf) continue;
      const SurfaceInfo si = ParseSurface(surf);
      if (si.valid && si.exp_bias != 0) {
        sys.color_exp_bias[i] = std::exp2f(float(si.exp_bias));
      }
    }
  }
  UploadAllocation b0 = g_upload[g_frame].Allocate(sizeof(sys), 256);
  std::memcpy(b0.memory, &sys, sizeof(sys));

  // b1: float constants, one buffer per stage. Live draws read the
  // device float shadow (VS regs at dev+1920, PS at dev+6016); replayed
  // CB draws keep the record-time capture in g_state. Dedup: while the
  // source bytes are unchanged within the frame, rebind the previous ring
  // allocation instead of re-uploading; never across frames.
  struct B1Cache {
    alignas(16) uint8_t src[4096];
    UploadAllocation alloc{};
    uint64_t frame = ~0ull;
    uint32_t device = 0;
    bool replay_src = false;
    bool valid = false;
  };
  static B1Cache b1_cache[2];  // [0] = VS, [1] = PS (render thread only)
  const bool b1_replay_src = g_replayDrawStateValid;
  const bool b1_dedup = REXCVAR_GET(native_video_b1_dedup);
  UploadAllocation b1_allocs[2];
  for (int st = 0; st < 2; st++) {
    const uint8_t* src =
        b1_replay_src
            ? reinterpret_cast<const uint8_t*>(st ? g_state.ps_floats
                                                  : g_state.vs_floats)
            : GuestPtr(device + (st ? 6016u : 1920u));
    B1Cache& c = b1_cache[st];
    if (b1_dedup && c.valid && c.frame == g_frameIndex &&
        c.replay_src == b1_replay_src &&
        (b1_replay_src || c.device == device) &&
        std::memcmp(c.src, src, 4096) == 0) {
      b1_allocs[st] = c.alloc;
      continue;
    }
    UploadAllocation a = g_upload[g_frame].Allocate(4096, 256);
    if (b1_replay_src) {
      std::memcpy(a.memory, src, 4096);
    } else {
      const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src);
      uint32_t* d32 = reinterpret_cast<uint32_t*>(a.memory);
      for (uint32_t i = 0; i < 1024; i++) d32[i] = __builtin_bswap32(s32[i]);
    }
    if (b1_dedup) {
      std::memcpy(c.src, src, 4096);
      c.alloc = a;
      c.frame = g_frameIndex;
      c.device = device;
      c.replay_src = b1_replay_src;
      c.valid = true;
    }
    b1_allocs[st] = a;
  }
  UploadAllocation b1vs = b1_allocs[0];
  UploadAllocation b1ps = b1_allocs[1];

  // b2: bool/loop constants from the device shadow block (the XDK draw flush
  // emits dev+10112 as PM4 block 0x4900 = SHADER_CONSTANT_BOOL_000_031 x8 +
  // LOOP_00..31 x32, exactly the 40-dword b2 layout, big-endian). Constants
  // are inherited live at CB replay per the XDK contract, so the draw
  // device's block is correct for replays too.
  UploadAllocation b2 = g_upload[g_frame].Allocate(160, 256);
  if (g_replayBoolLoopValid) {
    // Replayed CB draw: the bools the game set on the recording device at
    // record time (host order already).
    std::memcpy(b2.memory, g_replayBoolLoop, sizeof(g_replayBoolLoop));
  } else {
    const uint32_t* src =
        reinterpret_cast<const uint32_t*>(GuestPtr(device + 10112));
    uint32_t* dst = reinterpret_cast<uint32_t*>(b2.memory);
    for (uint32_t i = 0; i < 40; i++) dst[i] = __builtin_bswap32(src[i]);
  }

  // b3: fetch constants come from the guest device fetch shadow
  // (dev+1152+24*slot), the same ground truth the SRV selection reads;
  // live headers diverge (inline shadow writes, dead headers). Vertex
  // fetch constants overwrite their slots below.
  uint32_t fetch_dwords[192] = {};
  g_drawVfetchWrittenMask = 0;
  for (uint32_t slot = 0; slot < 32; slot++) {
    // Replayed draws: the recording device's shadow at record time is the
    // truth; the run device's live shadow belongs to other passes by
    // replay time.
    if (g_replayFetchValid & (1u << slot)) {
      std::memcpy(&fetch_dwords[slot * 6], g_replayFetch[slot], 24);
      continue;
    }
    const uint32_t slot_addr = device + 1152 + 24 * slot;
    const uint32_t dw0 = LoadGuestU32(slot_addr);
    const uint32_t dw1 = LoadGuestU32(slot_addr + 4);
    if (!(dw0 | dw1)) continue;  // empty slot, keep zeros (matches unbound)
    fetch_dwords[slot * 6 + 0] = dw0;
    fetch_dwords[slot * 6 + 1] = dw1;
    for (uint32_t i = 2; i < 6; i++) {
      fetch_dwords[slot * 6 + i] = LoadGuestU32(slot_addr + i * 4);
    }
  }
  // The vfetch dword range (160-191 = vertex streams 15..0) must never
  // carry shadow/replay values: guest-space vertex pairs are untranslatable
  // in ring space. BindStreamData rewrites bound streams below; the
  // blank-fill after it covers the rest.
  std::memset(&fetch_dwords[160], 0, 32 * sizeof(uint32_t));
  if (up_vertex_data) {
    const void* up_src = g_upOverrideVertexData
                             ? g_upOverrideVertexData
                             : GuestDataPtrFast(up_vertex_data, up_size_bytes);
    BindStreamData(fetch_dwords, 0, up_src, up_size_bytes,
                   /*immediate=*/true,  // XDK copies UP data at call time
                   up_stride_bytes);
  } else {
    // Bind the streams the game set (VB object dword 16 = data pointer per
    // RenderVertexBufferDx9; offset/stride from SetStreamSource).
    for (uint32_t stream = 0; stream < 16; stream++) {
      const StreamState& ss = g_state.streams[stream];
      // Fetch shadow first: the guest device shadow pair is the
      // draw-time ground truth; the VB-object parse below is only a
      // fallback for streams with no shadow pair (game-built VB objects
      // do not all follow the XDK field layout).
      if (REXCVAR_GET(native_video_fetch_shadow_streams) && device) {
        const uint32_t dw_index = 190 - 2 * stream;
        const uint32_t g0 = LoadGuestU32(device + 1152 + dw_index * 4);
        const uint32_t g1 = LoadGuestU32(device + 1152 + dw_index * 4 + 4);
        if ((g0 & 3u) == 3u) {
          const uint32_t base = g0 & ~3u;
          const uint32_t size = g1 & 0x3FFFFFCu;
          if (base && size && size <= kUploadBufferSize / 2) {
            // E0-window VB rebase: the shadow pair carries the
            // GPU-translated physical, whose virtual page is unrelated
            // memory. When the bound VB object's raw pointer translates
            // to this shadow base via the console formula, read the bytes
            // through the raw virtual instead.
            uint32_t read_base = base;
            if (ss.vb_object) {
              const uint32_t raw = ss.snap_valid
                                       ? ss.snap_data
                                       : LoadGuestU32(ss.vb_object + 24);
              if ((raw & 3u) == 3u) {
                const uint32_t raw_base = raw & ~3u;
                if (raw_base != base &&
                    GuestPhysFromHeaderBase(raw_base) == base) {
                  read_base = raw_base;
                }
              }
            }
            // Not physical_first: stream data lives in the virtual pages,
            // the opposite of textures.
            const uint8_t* src = GuestDataPtrFast(read_base, size);
            BindStreamData(fetch_dwords, stream, src, size,
                           /*immediate=*/false,
                           /*stride=*/ss.vb_object ? ss.stride_bytes : 0,
                           /*guest_key=*/base);
            continue;
          }
        }
      }
      // stride 0 is legal (Avatar_Draw binds the bone-matrix VB and the mesh
      // VBs with stride 0, the fetch stride lives in the shader/decl); only
      // an unbound stream is skipped.
      if (!ss.vb_object) {
        continue;
      }
      // Two VB object flavors reach SetStreamSource: XDK D3DVertexBuffer
      // (+24 = data | vfetch type bits, +28 = size | endian flags) and
      // plain objects (data at +24, size at +28); the pointer's low bits
      // discriminate. Snapshot at bind time, never a live object read
      // (stack-built headers are dead by deferred-flush time).
      uint32_t data_base = ss.snap_valid ? ss.snap_data
                                         : LoadGuestU32(ss.vb_object + 24);
      uint32_t total = ss.snap_valid ? ss.snap_size
                                     : LoadGuestU32(ss.vb_object + 28);
      if ((data_base & 3u) == 3u) {
        data_base &= ~3u;
        total &= 0x3FFFFFCu;
      } else if (total > kUploadBufferSize / 2) {
        // Unrecognized third flavor: dump the object once.
        static uint64_t vb_dump_budget = 4;
        if (vb_dump_budget) {
          vb_dump_budget--;
          REXGPU_WARN(
              "videonative: [vblog] unknown VB layout obj={:#x} dw={:#x} "
              "{:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
              ss.vb_object, LoadGuestU32(ss.vb_object),
              LoadGuestU32(ss.vb_object + 4), LoadGuestU32(ss.vb_object + 8),
              LoadGuestU32(ss.vb_object + 12), LoadGuestU32(ss.vb_object + 16),
              LoadGuestU32(ss.vb_object + 20), LoadGuestU32(ss.vb_object + 24),
              LoadGuestU32(ss.vb_object + 28));
        }
      }
      const uint32_t data_ptr = data_base + ss.offset_bytes;
      if (!data_base || total <= ss.offset_bytes) continue;
      // guest_key is the last parameter; a bare address in the 5th slot
      // converts silently to `bool immediate`, keep the argument comments.
      BindStreamData(fetch_dwords, stream,
                     GuestDataPtrFast(data_ptr, total - ss.offset_bytes),
                     total - ss.offset_bytes, /*immediate=*/false,
                     /*stride=*/0, /*guest_key=*/data_ptr);
    }
  }
  // Untranslated vertex-fetch pairs would be read as offsets into the
  // upload ring and fetch some other upload's bytes; blank them instead.
  // A blank pair points past the ring (raw-buffer OOB reads return 0),
  // since base 0 would alias the frame's first real upload.
  for (uint32_t stream = 0; stream < 16; stream++) {
    if (g_drawVfetchWrittenMask & (1u << stream)) continue;
    const uint32_t vfetch_index = 95 - stream;
    fetch_dwords[vfetch_index * 2 + 0] = kUploadBufferSize | 3u;
    fetch_dwords[vfetch_index * 2 + 1] = 2u;
  }
  UploadAllocation b3 = g_upload[g_frame].Allocate(sizeof(fetch_dwords), 256);
  std::memcpy(b3.memory, fetch_dwords, sizeof(fetch_dwords));

  // b4: descriptor indices, one buffer per stage, each shader's slots are
  // 0-based within its own cbuffer (translator contract), so VS and PS get
  // separate uploads bound via the stage-visible b4 root params.
  auto upload_bindings = [&](const ResolvedShader* shader) {
    uint32_t descriptor_indices[32] = {};
    if (!shader) {  // PS-less depth-only draw: zeros
      UploadAllocation b4z =
          g_upload[g_frame].Allocate(sizeof(descriptor_indices), 256);
      std::memcpy(b4z.memory, descriptor_indices, sizeof(descriptor_indices));
      return b4z;
    }
    uint32_t slot = 0;
    for (const auto& binding : shader->pack->bindings) {
      if (slot >= 32) break;
      if (binding.is_sampler) {
        // Sampler slot selection. Legacy: clamp bits only (pick_sampler
        // below). Semantics path: the full ring contract, filters, aniso,
        // per-axis clamp incl. mirror-once/border, cube/1D dimension
        // rules, static slots 0-12 + write-once dynamic 13-15.
        uint32_t sampler_index = 0;
        // Xenos clamp mode -> axis-mode index {0 wrap, 1 mirror, 2 clamp}.
        auto axis_mode = [](uint32_t m) -> uint32_t {
          return m == 0 ? 0u : (m == 1 ? 1u : 2u);
        };
        auto pick_sampler = [&](uint32_t dw0) -> uint32_t {
          if (!REXCVAR_GET(native_video_per_axis_clamp)) {
            return ((dw0 >> 10) & 7) >= 2 ? 1u : 0u;  // legacy: clamp_x only
          }
          const uint32_t u = axis_mode((dw0 >> 10) & 7);
          const uint32_t v = axis_mode((dw0 >> 13) & 7);
          return 4 + u * 3 + v;
        };
        if (binding.fetch_constant < 32) {
          // The guest device fetch shadow (dev+1152+24*fc) is the bind
          // ground truth: written by the XDK bind bodies (mirrored by the
          // hooks here) and by inline engine writers no hook can see (the
          // avatar sun-shadow bind). The shadow lives in the device block,
          // never freed, so a draw-time read stays immune to dead headers.
          uint32_t sdw[6];
          if (g_replayFetchValid & (1u << binding.fetch_constant)) {
            std::memcpy(sdw, g_replayFetch[binding.fetch_constant], 24);
          } else {
            const uint32_t slot_addr =
                device + 1152 + 24 * binding.fetch_constant;
            for (int i = 0; i < 6; i++) {
              sdw[i] = LoadGuestU32(slot_addr + 4 * i);
            }
          }
          if (sdw[0] | sdw[1]) {  // unbound slot keeps sampler 0
            const uint32_t legacy = pick_sampler(sdw[0]);
            if (REXCVAR_GET(native_video_sem_sampler)) {
              semantics::SamplerSpec spec;
              semantics::DecodeSamplerSpec(sdw, &spec);
              sampler_index = semantics::MapToSlot(spec);
            } else {
              sampler_index = legacy;
            }
          }
        }
        descriptor_indices[slot++] = sampler_index;
      } else {
        uint32_t index = 0;
        CachedTexture* tex = nullptr;
        if (binding.fetch_constant < 32) {
          // Source the fetch constant from the guest device fetch shadow,
          // see the sampler block above. Empty slot (all-zero dwords) =
          // nothing bound.
          const uint32_t slot_addr =
              device + 1152 + 24 * binding.fetch_constant;
          uint32_t sdw[6];
          if (g_replayFetchValid & (1u << binding.fetch_constant)) {
            std::memcpy(sdw, g_replayFetch[binding.fetch_constant], 24);
          } else {
            for (int i = 0; i < 6; i++) {
              sdw[i] = LoadGuestU32(slot_addr + 4 * i);
            }
          }
          // Slot validity = dword0's TYPE bits (the guest's own unbind
          // convention: 0x9211BA20's NULL path clears exactly those 2 bits,
          // leaving the rest of the dwords populated).
          if ((sdw[0] & 3u) != 0 && (sdw[0] | sdw[1])) {
            tex = GetOrCreateTexture(slot_addr, false, 0, 0,
                                     RenderFormat::UNKNOWN, sdw);
            if (tex && tex->valid) index = tex->descriptor_index;
          }
        }
        // Dimension-matched descriptors only: a descriptor whose
        // ViewDimension mismatches the shader's declared type is D3D12 UB
        // (Intel pagefaults on it). Serve the blank of the declared
        // dimension on any mismatch or empty slot. binding.dimension:
        // 0/1 = 1D/2D, 2 = 3D, 3 = cube.
        if (binding.dimension == 3) {
          if (!tex || !tex->valid || tex->view_dim != 3) {
            index = kBlankCubeDescriptor;
          }
        } else if (binding.dimension == 2) {
          // No real 3D textures exist (guest 3D is treated as 2D at
          // creation), a Texture3D declaration always gets the 3D blank.
          index = kBlank3dDescriptor;
        } else if (tex && tex->valid && tex->view_dim == 3) {
          index = 0;  // cube texture through the 2D declaration: 2D blank
        }
        descriptor_indices[slot++] = index;
      }
    }
    UploadAllocation b4 =
        g_upload[g_frame].Allocate(sizeof(descriptor_indices), 256);
    std::memcpy(b4.memory, descriptor_indices, sizeof(descriptor_indices));
    return b4;
  };
  UploadAllocation b4vs = upload_bindings(setup->vs);
  UploadAllocation b4ps = upload_bindings(setup->ps);

  commandList->setGraphicsRootDescriptor(b0.buffer->at(b0.offset),
                                         kRootB0System);
  commandList->setGraphicsRootDescriptor(b1vs.buffer->at(b1vs.offset),
                                         kRootB1FloatsVs);
  commandList->setGraphicsRootDescriptor(b1ps.buffer->at(b1ps.offset),
                                         kRootB1FloatsPs);
  commandList->setGraphicsRootDescriptor(b2.buffer->at(b2.offset),
                                         kRootB2BoolLoop);
  commandList->setGraphicsRootDescriptor(b3.buffer->at(b3.offset),
                                         kRootB3Fetch);
  commandList->setGraphicsRootDescriptor(b4vs.buffer->at(b4vs.offset),
                                         kRootB4IndicesVs);
  commandList->setGraphicsRootDescriptor(b4ps.buffer->at(b4ps.offset),
                                         kRootB4IndicesPs);
  auto& ring = g_vertexRing[g_frame];
  if (ring.buffer) {
    ring.ToReadState(commandList.get());  // copies recorded above must land
    commandList->setGraphicsRootDescriptor(ring.buffer->at(0),
                                           kRootSharedMemSrv);
  }
  commandList->setGraphicsRootDescriptor(g_dummyUav->at(0), kRootDummyUav);
  return true;
}

}  // namespace

void DrawVertices(uint32_t device, uint32_t prim_type, uint32_t start_vertex,
                  uint32_t vertex_count) {
  if (rq::Active()) { rq::EnqDrawVertices(device, prim_type, start_vertex, vertex_count); return; }
  // Xenos QUADLIST (13) / TRIANGLE FAN (5): no D3D12 topology, draw as an
  // indexed tri-list over the auto-index vertex stream. Only the index
  // stream is synthesized (host-side, already little-endian); the vertex
  // fetch still reads the bound/fetch-shadow streams. AE's XUI glyph and
  // RTT quads arrive here via the 0x92121380 auto-index commit.
  if ((prim_type == 13 || prim_type == 5) && vertex_count >= 3 &&
      vertex_count <= 0xFFFF) {
    DrawSetup setup;
    if (!SetupDraw(device, 4, &setup, 0, 0, 0, int32_t(start_vertex))) {
      return;
    }
    static thread_local std::vector<uint16_t> tri;
    tri.clear();
    if (prim_type == 13) {
      for (uint32_t q = 0; q + 4 <= vertex_count; q += 4) {
        tri.push_back(uint16_t(q + 0));
        tri.push_back(uint16_t(q + 1));
        tri.push_back(uint16_t(q + 2));
        tri.push_back(uint16_t(q + 0));
        tri.push_back(uint16_t(q + 2));
        tri.push_back(uint16_t(q + 3));
      }
    } else {
      for (uint32_t i = 2; i < vertex_count; i++) {
        tri.push_back(0);
        tri.push_back(uint16_t(i - 1));
        tri.push_back(uint16_t(i));
      }
    }
    if (tri.empty()) return;
    const uint32_t size_bytes = uint32_t(tri.size()) * 2;
    UploadAllocation alloc = g_upload[g_frame].Allocate(size_bytes, 4);
    if (!alloc.memory) return;
    std::memcpy(alloc.memory, tri.data(), size_bytes);
    RenderIndexBufferView view(alloc.buffer->at(alloc.offset), size_bytes,
                               RenderFormat::R16_UINT);
    g_commandLists[g_frame]->setIndexBuffer(&view);
    g_commandLists[g_frame]->drawIndexedInstanced(uint32_t(tri.size()), 1, 0,
                                                  0, 0);
    return;
  }
  DrawSetup setup;
  if (!SetupDraw(device, prim_type, &setup, 0, 0, 0, int32_t(start_vertex))) {
    return;
  }
  g_commandLists[g_frame]->drawInstanced(vertex_count, 1, 0, 0);
}

void DrawIndexedVertices(uint32_t device, uint32_t prim_type,
                         uint32_t base_vertex, uint32_t start_index,
                         uint32_t index_count) {
  if (rq::Active()) { rq::EnqDrawIndexedVertices(device, prim_type, base_vertex, start_index, index_count); return; }
  DrawSetup setup;
  if (!SetupDraw(device, prim_type, &setup, 0, 0, 0, int32_t(base_vertex))) {
    return;
  }
  // Guest IB object (D3D_CreateIndexBuffer 0x82528B58, 32 bytes): +24 = raw
  // data pointer, +28 = byte size, dword0 bit 31 = 32-bit index format (the
  // XDK draw body 0x82534DF8 tests the sign bit and switches to 4-byte index
  // addressing); big-endian either way. Fields come from the bind-time
  // snapshot: the avatar section renderers pass stack-built IB headers
  // (XGSetIndexBufferHeader onto sp) that are dead by deferred-flush time,
  // so a live object read here would fetch stack garbage as the data
  // pointer.
  const uint32_t ib = g_state.index_buffer_object;
  if (!ib) return;
  uint32_t data_ptr;
  bool idx32;
  if (g_state.ib_snap_valid) {
    data_ptr = g_state.ib_snap_data;
    idx32 = g_state.ib_snap_idx32;
  } else {
    data_ptr = LoadGuestU32(ib + 24);
    idx32 = (LoadGuestU32(ib) & 0x80000000u) != 0;
  }
  if (!data_ptr) return;
  if (idx32) {
    const uint32_t size_bytes = index_count * 4;
    UploadAllocation alloc = g_upload[g_frame].Allocate(size_bytes, 4);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(
        GuestDataPtrFast(data_ptr + start_index * 4, size_bytes));
    if (!src) return;
    if (!rq::Running() && REXCVAR_GET(native_video_deferred_vb_copy)) {
      g_deferredCopies.push_back({src, alloc.memory, size_bytes, 4});
    } else {
      uint32_t* dst = reinterpret_cast<uint32_t*>(alloc.memory);
      for (uint32_t i = 0; i < index_count; i++) {
        dst[i] = __builtin_bswap32(src[i]);
      }
    }
    RenderIndexBufferView view(alloc.buffer->at(alloc.offset), size_bytes,
                               RenderFormat::R32_UINT);
    g_commandLists[g_frame]->setIndexBuffer(&view);
  } else {
    const uint32_t size_bytes = index_count * 2;
    UploadAllocation alloc = g_upload[g_frame].Allocate(size_bytes, 4);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(
        GuestDataPtrFast(data_ptr + start_index * 2, size_bytes));
    if (!src) return;
    if (!rq::Running() && REXCVAR_GET(native_video_deferred_vb_copy)) {
      g_deferredCopies.push_back({src, alloc.memory, size_bytes, 2});
    } else {
      uint16_t* dst = reinterpret_cast<uint16_t*>(alloc.memory);
      for (uint32_t i = 0; i < index_count; i++) {
        dst[i] = __builtin_bswap16(src[i]);
      }
    }
    RenderIndexBufferView view(alloc.buffer->at(alloc.offset), size_bytes,
                               RenderFormat::R16_UINT);
    g_commandLists[g_frame]->setIndexBuffer(&view);
  }
  g_commandLists[g_frame]->drawIndexedInstanced(index_count, 1, 0, 0, 0);
}

void SetUPDataOverride(const void* vertex_data, const void* index_data) {
  if (rq::Active()) { rq::SetPendingUPOverride(vertex_data, index_data); return; }
  g_upOverrideVertexData = vertex_data;
  g_upOverrideIndexData = index_data;
}


void DrawVerticesUP(uint32_t device, uint32_t prim_type, uint32_t vertex_count,
                    uint32_t vertex_data, uint32_t stride_bytes) {
  if (rq::Active()) { rq::EnqDrawVerticesUP(device, prim_type, vertex_count, vertex_data, stride_bytes); return; }
  // Prim 5 = triangle fan, prim 13 = quadlist (XUI quads; BeginVertices
  // glyph runs arrive as quadlists, perimeter order TL TR BR BL). D3D12 has
  // neither topology, so triangulate to an indexed tri-list
  // (fan: 0,i-1,i; quads: 0,1,2 0,2,3 per quad). Indices are fed
  // pre-byteswapped through the host override (the UP-index path swaps
  // guest BE -> host LE on copy).
  if ((prim_type == 5 || prim_type == 13) && vertex_count >= 3 &&
      vertex_count <= 0xFFFF) {
    static thread_local std::vector<uint16_t> fan;
    fan.clear();
    if (prim_type == 13) {
      for (uint32_t q = 0; q + 4 <= vertex_count; q += 4) {
        fan.push_back(__builtin_bswap16(uint16_t(q + 0)));
        fan.push_back(__builtin_bswap16(uint16_t(q + 1)));
        fan.push_back(__builtin_bswap16(uint16_t(q + 2)));
        fan.push_back(__builtin_bswap16(uint16_t(q + 0)));
        fan.push_back(__builtin_bswap16(uint16_t(q + 2)));
        fan.push_back(__builtin_bswap16(uint16_t(q + 3)));
      }
    } else {
      for (uint32_t i = 2; i < vertex_count; i++) {
        fan.push_back(0);
        fan.push_back(__builtin_bswap16(uint16_t(i - 1)));
        fan.push_back(__builtin_bswap16(uint16_t(i)));
      }
    }
    if (fan.empty()) return;
    g_upOverrideIndexData = fan.data();
    DrawIndexedVerticesUP(device, 4, 0, vertex_count, uint32_t(fan.size()), 0,
                          1, vertex_data, stride_bytes);
    g_upOverrideIndexData = nullptr;
    return;
  }
  DrawSetup setup;
  if (!SetupDraw(device, prim_type, &setup, vertex_data,
                 vertex_count * stride_bytes, stride_bytes, 0)) {
    return;
  }
  g_commandLists[g_frame]->drawInstanced(vertex_count, 1, 0, 0);
}

void DrawIndexedVerticesUP(uint32_t device, uint32_t prim_type,
                           uint32_t min_vertex, uint32_t num_vertices,
                           uint32_t index_count, uint32_t index_data,
                           uint32_t index_format, uint32_t vertex_data,
                           uint32_t stride_bytes) {
  if (rq::Active()) { rq::EnqDrawIndexedVerticesUP(device, prim_type, min_vertex, num_vertices, index_count, index_data, index_format, vertex_data, stride_bytes); return; }
  (void)min_vertex;
  (void)index_format;  // 1 = 16-bit, the only format another title uses
  DrawSetup setup;
  if (!SetupDraw(device, prim_type, &setup, vertex_data,
                 num_vertices * stride_bytes, stride_bytes, 0)) {
    return;
  }
  const uint32_t size_bytes = index_count * 2;
  UploadAllocation alloc = g_upload[g_frame].Allocate(size_bytes, 4);
  const uint16_t* src = reinterpret_cast<const uint16_t*>(
      g_upOverrideIndexData ? g_upOverrideIndexData
                            : GuestDataPtrFast(index_data, size_bytes));
  if (!src) return;
  uint16_t* dst = reinterpret_cast<uint16_t*>(alloc.memory);
  for (uint32_t i = 0; i < index_count; i++) dst[i] = __builtin_bswap16(src[i]);
  RenderIndexBufferView view(alloc.buffer->at(alloc.offset), size_bytes,
                             RenderFormat::R16_UINT);
  g_commandLists[g_frame]->setIndexBuffer(&view);
  g_commandLists[g_frame]->drawIndexedInstanced(index_count, 1, 0, 0, 0);
}

}  // namespace rex::videonative::renderer
