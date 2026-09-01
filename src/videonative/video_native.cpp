// rexvideonative, see video_native.h and LICENSE (GPL-3.0).
//
// Guest API surface + shader resolution live here; the plume backend (frame
// flow, pipelines, texture cache, draw submission) lives in renderer.cpp.

#include "video_native.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>  // SetEnvironmentVariableA (DRED arming)
#endif

#include <fmt/format.h>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>

#include <plume_render_interface.h>

#include "renderer.h"
#include "shader_cache.h"
#include "video_native_internal.h"
#include "render_queue.h"

namespace plume {
// Defined in plume_d3d12.cpp; not declared in the public headers.
extern std::unique_ptr<RenderInterface> CreateD3D12Interface();
}  // namespace plume

REXCVAR_DEFINE_BOOL(native_video, false, "GPU",
                    "Enable the native video layer (UnleashedRecomp-style "
                    "guest D3D implementation; experimental)");
REXCVAR_DEFINE_BOOL(native_video_swap_bookkeeping, true, "GPU",
                    "Emulate the skipped XDK Swap body's frame bookkeeping "
                    "per present: write "
                    "the two per-frame fence-ring slots (pool[16+(f&7)], "
                    "pool[16+((f+1)&7)]) and the VdSwap writeback (pool+8) "
                    "with an advancing counter, then frame counter "
                    "dev+22036 += 2 and completed dev+22032 kept current. "
                    "Off leaves them at zero.");
REXCVAR_DEFINE_BOOL(native_video_device_ring_scratch, true, "GPU",
                    "Give the replacement guest device a REAL command-buffer "
                    "scratch segment (dev+48/+56) and fence pool (dev+11024). "
                    "With them zeroed, any recompiled XDK PM4 "
                    "emitter still running natively appends its packet dwords "
                    "through *(dev+48)=0 into guest page zero, corrupting the "
                    "null-object reads some titles legitimately perform "
                    "there.");
REXCVAR_DEFINE_INT32(native_video_heap_pad, 0, "GPU",
                     "Guest-heap padding (bytes) allocated at native-video "
                     "init. Shifts the game's subsequent allocation layout, "
                     "which changes which addresses any address-sensitive "
                     "game bug lands on. 0 = off.");
REXCVAR_DEFINE_INT32(native_video_gamma_ramp_offset, 0, "GPU",
                     "Device-block offset of the D3DGAMMARAMP. 0 = auto "
                     "(score +15408 vs +15216 as a real ramp and pick "
                     "the valid one). Set "
                     "explicitly to override.");
REXCVAR_DEFINE_BOOL(native_video_fence_pool_max, false, "GPU",
                    "Pin the device fence pool's completed counter at max "
                    "(0xFFFFFF00): every guest fence wait is instantly "
                    "satisfied. Needed by titles whose loader blocks on "
                    "resource fences; the default keeps the per-present "
                    "idiom.");
REXCVAR_DEFINE_BOOL(native_video_fence_emulation, false, "GPU",
                    "Emulate GPU resource-fence progression: "
                    "binds stamp the resource Fence field (obj+8) with the "
                    "device submit counter (dev+11036); Swap advances the "
                    "counter and writes completed=cur-1 to *(dev+11024). "
                    "D3DResource_Release then defers frees of in-flight "
                    "textures exactly as on the ring path, closing the "
                    "instant-recycle channel that bakes stale placeholder "
                    "art into texture-cache entries.");
REXCVAR_DEFINE_STRING(native_video_adapter, "", "GPU",
                      "Pick the native-video GPU: a case-insensitive name "
                      "fragment ('intel', 'vega', 'radeon') or a hex DXGI "
                      "vendor id (8086=Intel, 1002=AMD, 10de=NVIDIA). Empty "
                      "= first hardware adapter. No match = warn + default. "
                      "Hybrid-GPU machines pick per toml.");
REXCVAR_DEFINE_STRING(native_video_shader_pack, "", "GPU",
                      "Directory of the XenosRecomp native shader pack "
                      "(manifest.csv + DXIL) used by the native video layer");
REXCVAR_DEFINE_BOOL(native_video_cb_const_diffs, true, "GPU",
                    "Replay command-buffer draws with the float constants the "
                    "recording device held at record time (registers dirtied "
                    "since Begin, the XDK bakes exactly those into the CB as "
                    "PM4). false = inherit-only replay.");

namespace rex::videonative {

namespace {

// Size of the guest-visible device block returned by CreateDevice. Unhooked
// recompiled XDK state setters write shadow registers into it; the renderer
// reads render state back out of those shadows at draw time.
constexpr uint32_t kGuestDeviceSize = 0x8000;

std::mutex g_initMutex;
std::atomic<bool> g_initialized{false};
bool g_initFailed = false;

std::unique_ptr<plume::RenderInterface> g_interface;
std::unique_ptr<plume::RenderDevice> g_device;
std::unique_ptr<plume::RenderCommandQueue> g_queue;

std::atomic<uint64_t> g_frameCount{0};

void StoreGuestU32(GuestAddr addr, uint32_t value) {
  auto* p = rex::system::kernel_memory()->TranslateVirtual<uint32_t*>(addr);
  *p = __builtin_bswap32(value);
}

uint32_t LoadGuestU32(GuestAddr addr) {
  return __builtin_bswap32(
      *rex::system::kernel_memory()->TranslateVirtual<const uint32_t*>(addr));
}

// --- shader resolution -------------------------------------------------------

ShaderCache g_shaderCache;
std::once_flag g_shaderPackLoadOnce;
uint64_t g_shaderPackHits = 0;
uint64_t g_shaderPackMisses = 0;

// XDK shader object layout (D3DDevice_CreateVertexShader/CreatePixelShader,
// base 0x82525320/0x82525138): dword0 low nibble = resource type.
constexpr uint32_t kShaderTypeVertex = 6;
constexpr uint32_t kShaderTypePixel = 7;

VfetchPatchFn g_vfetchPatcher = nullptr;

// Current bind state relevant to VS resolution (render thread only).
GuestAddr g_currentVertexShader = 0;
GuestAddr g_currentPixelShader = 0;
GuestAddr g_currentVertexDecl = 0;
uint8_t g_streamStrideDwords[16] = {};
// The device of the draw currently being resolved, and whether that draw is
// a UP draw (inline vertex data bound as a temporary stream 0). Both feed
// the vfetch-patch stride table, see native_video_stride_table_offset.
GuestAddr g_currentDrawDevice = 0;
bool g_currentDrawIsUp = false;

const ResolvedShader* g_currentResolvedVs = nullptr;
const ResolvedShader* g_currentResolvedPs = nullptr;

// VS/PS resolution caches. Both maps are insert-only and nodes are never
// rewritten in place: the render-queue worker dereferences ResolvedShader
// pointers from in-flight draw records, so a re-resolve (the game
// link-patches microcode after load) must land in a new node;
// its content hash is part of the key.
struct VsKey {
  uint64_t k0, k1;
  bool operator==(const VsKey& o) const { return k0 == o.k0 && k1 == o.k1; }
};
struct VsKeyHash {
  size_t operator()(const VsKey& k) const {
    return size_t(k.k0 ^ (k.k1 * 0x9E3779B97F4A7C15ull));
  }
};
std::unordered_map<VsKey, ResolvedShader, VsKeyHash> g_resolvedVertexShaders;
// PS keyed by (object, code ptr) + content hash.
std::unordered_map<VsKey, ResolvedShader, VsKeyHash> g_resolvedPixelShaders;

// Guest scratch used for the patched microcode copy + the stride table.
uint32_t g_patchScratch = 0;
uint32_t g_patchScratchSize = 0;
uint32_t g_strideTableGuest = 0;

void EnsureShaderPackLoaded() {
  std::call_once(g_shaderPackLoadOnce, [] {
    const std::string& pack = REXCVAR_GET(native_video_shader_pack);
    if (!pack.empty()) {
      g_shaderCache.Load(pack);
    } else {
      REXGPU_WARN(
          "videonative: native_video_shader_pack not set, all shaders will "
          "miss");
    }
  });
}

void NoteResolveOutcome(const ResolvedShader& resolved,
                        const void* ucode_guest) {
  if (resolved.pack) {
    g_shaderPackHits++;
  } else {
    g_shaderPackMisses++;
    REXGPU_WARN("videonative: pack MISS {} ucode {:016X} ({} bytes)",
                resolved.is_pixel ? "ps" : "vs", resolved.ucode_hash,
                resolved.code_size);
  }
  if ((g_shaderPackHits + g_shaderPackMisses) % 64 == 0) {
    REXGPU_INFO("videonative: shader resolves {} hit / {} miss",
                g_shaderPackHits, g_shaderPackMisses);
  }
}

// Pixel shaders: hash the object's microcode directly. Object layout
// (CreatePixelShader 0x82525138): 40-byte header (physical-blob pointer at
// +0x18, stored by 0x825235E0) followed by a copy of the container's virtual
// image. The physical blob may carry default-constant data before the ucode
// (a 64-byte float4 default block on def-carrying shaders), so the ucode
// offset/size must come from the container Shader struct; the blob-wide
// fields would hash the defaults too.
const ResolvedShader* ResolvePixelShader(GuestAddr shader_obj) {
  EnsureShaderPackLoaded();
  const uint32_t phys_base = LoadGuestU32(shader_obj + 0x18);
  const uint32_t container = shader_obj + 40;
  const uint32_t shader_off = LoadGuestU32(container + 24);
  if (!phys_base || !shader_off || shader_off > 0x8000) {
    return nullptr;
  }
  const uint32_t code_off = LoadGuestU32(container + shader_off);
  const uint32_t code_size = LoadGuestU32(container + shader_off + 4);
  const uint32_t code_ptr = phys_base + code_off;
  if (!code_size || code_size > 0x100000) return nullptr;
  // The game link-patches shader microcode in place after load (physical
  // constant-register allocation: the FX shaders' c8-11 window becomes
  // c48-51 once the material links), so the cache keys on content rather
  // than on the code pointer alone.
  const void* code_host =
      rex::system::kernel_memory()->TranslateVirtual<const void*>(code_ptr);
  const uint64_t raw_hash = XXH3_64bits(code_host, code_size);
  VsKey key;
  key.k0 = (uint64_t(shader_obj) << 32) | code_ptr;
  key.k1 = raw_hash;
  auto it = g_resolvedPixelShaders.find(key);
  if (it != g_resolvedPixelShaders.end()) {
    return &it->second;
  }

  ResolvedShader resolved;
  resolved.guest_object = shader_obj;
  resolved.code_ptr = code_ptr;
  resolved.code_size = code_size;
  resolved.is_pixel = true;
  resolved.ucode_hash = raw_hash;
  resolved.pack = g_shaderCache.Find(resolved.ucode_hash, true);
  NoteResolveOutcome(resolved, code_host);
  return &(g_resolvedPixelShaders[key] = resolved);
}

// Mirror-mode alternative to the scratch re-patch: hash the live
// physical-blob bytes, valid whenever the guest executes its own
// bind-time vfetch patch (the scratch re-patch drifts on titles whose
// stream tracking is incomplete).
REXCVAR_DEFINE_BOOL(native_video_vs_hash_live, false, "GPU",
                    "Hash live (guest-patched) VS microcode instead of "
                    "re-patching a scratch copy. Mirror-mode titles only.");

// Where the scratch re-patch gets its strides: the XDK's draw flush uses the
// device's per-stream stride table (dev+12704 here), while the tracked
// reconstruction drifts on engines that write stream state inline. Set the
// device offset of the real table; 0 keeps the reconstruction.
REXCVAR_DEFINE_INT32(native_video_stride_table_offset, 12704, "GPU",
                     "Device-block offset of the per-stream stride-in-dwords "
                     "byte table handed to the vfetch patcher (0 = use vn's "
                     "tracked strides).");

// Vertex shaders: reproduce the XDK's per-binding vfetch patch (via the
// recompiled patcher) on a guest scratch copy, then hash the patched bytes,
// which is what the CP sees on hardware and what the pack is keyed on.
const ResolvedShader* ResolveVertexShaderForDraw() {
  const GuestAddr shader_obj = g_currentVertexShader;
  if (REXCVAR_GET(native_video_vs_hash_live)) {
    if (!shader_obj) return nullptr;
    EnsureShaderPackLoaded();
    auto* mem = rex::system::kernel_memory();
    // Variant selection: the XDK draw flush picks between two code bodies
    // per VS object, descriptor obj[224] (variant 0) vs obj[226] (variant 1),
    // switched by obj_dword[218] & 0x20. Skinned avatar draws select
    // variant 1.
    const uint32_t d218 = LoadGuestU32(shader_obj + 218 * 4);
    const bool variant1 = (d218 & 0x20u) != 0;
    uint32_t desc = LoadGuestU32(shader_obj + (variant1 ? 226 : 224) * 4);
    if (variant1 && (!desc || desc >= 0x10000)) {
      desc = LoadGuestU32(shader_obj + 224 * 4);  // degenerate obj: fall back
    }
    const uint32_t code_off = LoadGuestU32(shader_obj + desc + 872);
    const uint32_t code_size = LoadGuestU32(shader_obj + desc + 876);
    const uint32_t code_ptr = LoadGuestU32(shader_obj + 8 * 4) + code_off;
    if (!code_ptr || !code_size || code_size > 0x100000) return nullptr;
    const void* code_host = mem->TranslateVirtual<const void*>(code_ptr);
    const uint64_t live_hash = XXH3_64bits(code_host, code_size);
    VsKey key;
    key.k0 = (uint64_t(shader_obj) << 32) | code_ptr;
    key.k1 = live_hash;
    auto it = g_resolvedVertexShaders.find(key);
    if (it != g_resolvedVertexShaders.end()) return &it->second;
    ResolvedShader resolved;
    resolved.guest_object = shader_obj;
    resolved.code_ptr = code_ptr;
    resolved.code_size = code_size;
    resolved.is_pixel = false;
    resolved.ucode_hash = live_hash;
    resolved.pack = g_shaderCache.Find(resolved.ucode_hash, false);
    NoteResolveOutcome(resolved, code_host);
    return &(g_resolvedVertexShaders[key] = resolved);
  }
  if (!shader_obj || !g_vfetchPatcher) return nullptr;
  EnsureShaderPackLoaded();

  auto* mem = rex::system::kernel_memory();
  // Code pointer/size as the XDK flush reads them: descriptor offset at
  // obj[224], code offset/size at obj+desc+872/876, base at obj dword8.
  // The flush picks between two code bodies per VS object (obj[224] vs
  // obj[226], switched by obj_dword[218] & 0x20); mirror that.
  const uint32_t d218_sel = LoadGuestU32(shader_obj + 218 * 4);
  uint32_t variant = (d218_sel & 0x20u) ? 1u : 0u;
  uint32_t desc = LoadGuestU32(shader_obj + (variant ? 226 : 224) * 4);
  if (variant && (!desc || desc >= 0x10000)) {
    // Degenerate object: fall back to variant 0, and patch as variant 0;
    // the patcher indexes its own per-variant tables off this same number.
    variant = 0;
    desc = LoadGuestU32(shader_obj + 224 * 4);
  }
  const uint32_t code_off = LoadGuestU32(shader_obj + desc + 872);
  const uint32_t code_size = LoadGuestU32(shader_obj + desc + 876);
  const uint32_t code_ptr = LoadGuestU32(shader_obj + 8 * 4) + code_off;
  if (!code_ptr || !code_size || code_size > 0x100000) return nullptr;

  // Mix the raw (pre-vfetch-patch) code content into the cache key: the game
  // link-patches constant registers in the microcode after load (physical
  // register allocation; the FX UV-transform window c8-11 becomes c48-51),
  // so a pointer-keyed cache would serve the pre-link translation forever.
  const uint64_t raw_hash = XXH3_64bits(
      mem->TranslateVirtual<const void*>(code_ptr), code_size);

  // The stride table the patch will use. Prefer the guest's own (the very
  // bytes the XDK flush hands the patcher); the tracked reconstruction is
  // the fallback. Either way the effective bytes, not their source, are
  // what the cache keys on, since they change the patched microcode.
  uint8_t strides[sizeof(g_streamStrideDwords)];
  const int32_t stride_off = REXCVAR_GET(native_video_stride_table_offset);
  const bool from_device = stride_off > 0 && g_currentDrawDevice != 0;
  if (from_device) {
    const auto* dev_table = mem->TranslateVirtual<const uint8_t*>(
        g_currentDrawDevice + uint32_t(stride_off));
    std::memcpy(strides, dev_table, sizeof(strides));
    // UP draws bind their inline data as a temporary stream 0, and the XDK
    // writes that stride into the device table from inside the draw body,
    // which replace mode skips. g_streamStrideDwords[0] was set from the draw
    // argument on the way in, so it holds what the guest would have written;
    // everything else stays the device's.
    if (g_currentDrawIsUp) strides[0] = g_streamStrideDwords[0];
  } else {
    std::memcpy(strides, g_streamStrideDwords, sizeof(strides));
  }

  VsKey key;
  key.k0 = (uint64_t(shader_obj) << 32) | g_currentVertexDecl;
  key.k1 = XXH3_64bits(strides, sizeof(strides)) ^ raw_hash ^ variant;
  auto it = g_resolvedVertexShaders.find(key);
  if (it != g_resolvedVertexShaders.end()) return &it->second;

  if (g_patchScratchSize < code_size) {
    // Grow-only scratch; leaked on shrink which is fine for a scratch.
    g_patchScratch = mem->SystemHeapAlloc((code_size + 0xFFF) & ~0xFFFu, 0x100);
    g_patchScratchSize = (code_size + 0xFFF) & ~0xFFFu;
  }
  if (!g_strideTableGuest) {
    g_strideTableGuest = mem->SystemHeapAlloc(16, 0x10);
  }
  if (!g_patchScratch || !g_strideTableGuest) return nullptr;

  std::memcpy(mem->TranslateVirtual<void*>(g_patchScratch),
              mem->TranslateVirtual<const void*>(code_ptr), code_size);
  std::memcpy(mem->TranslateVirtual<void*>(g_strideTableGuest), strides,
              sizeof(strides));

  if (g_currentVertexDecl) {
    g_vfetchPatcher(shader_obj, g_patchScratch, g_currentVertexDecl,
                    g_strideTableGuest, variant);
  }

  ResolvedShader resolved;
  resolved.guest_object = shader_obj;
  resolved.code_ptr = code_ptr;
  resolved.code_size = code_size;
  resolved.is_pixel = false;
  resolved.ucode_hash = XXH3_64bits(
      mem->TranslateVirtual<const void*>(g_patchScratch), code_size);
  resolved.pack = g_shaderCache.Find(resolved.ucode_hash, false);
  NoteResolveOutcome(resolved,
                     mem->TranslateVirtual<const void*>(g_patchScratch));
  return &(g_resolvedVertexShaders[key] = resolved);
}

// `device` and `is_up` are the vfetch-patch inputs the shader resolver needs
// but cannot see: the guest stride table lives in the device block, and UP
// draws carry stream 0's stride in the draw argument rather than in it.
void OnDraw(GuestAddr device, bool is_up) {
  g_currentDrawDevice = device;
  g_currentDrawIsUp = is_up;
  g_currentResolvedVs = ResolveVertexShaderForDraw();
  if (!g_currentPixelShader) {
    static int diag = 0;
    if (diag < 4) {
      diag++;
      REXGPU_WARN("videonative: draw with NO pixel shader bound (vs obj {:#x})",
                  g_currentVertexShader);
    }
  }
  g_currentResolvedPs =
      g_currentPixelShader ? ResolvePixelShader(g_currentPixelShader) : nullptr;
}

// Per-device float-constant dirty bitmaps (bit = ALU register, [0]=VS,
// [1]=PS), mirroring the XDK's tracking: set by the public constant
// entries, cleared by every draw. Load-time shader defaults must not
// override dirty registers; on hardware the flush re-emits dirty ranges
// after loading defaults, so game-supplied values win.
struct FloatDirtyMap {
  uint32_t bits[2][8];
};
std::unordered_map<uint32_t, FloatDirtyMap> g_floatDirty;

void MarkFloatsDirty(GuestAddr device, bool pixel, uint32_t start,
                     uint32_t count) {
  if (!device) return;
  auto& m = g_floatDirty[device];
  const uint32_t end = std::min(start + count, 256u);
  for (uint32_t r = start; r < end; r++) {
    m.bits[pixel ? 1 : 0][r >> 5] |= 1u << (r & 31);
  }
}
bool FloatRegDirty(GuestAddr device, bool pixel, uint32_t reg) {
  auto it = g_floatDirty.find(device);
  if (it == g_floatDirty.end()) return false;
  return (it->second.bits[pixel ? 1 : 0][reg >> 5] >> (reg & 31)) & 1u;
}
void ClearFloatDirty(GuestAddr device) {
  auto it = g_floatDirty.find(device);
  if (it != g_floatDirty.end()) {
    std::memset(it->second.bits, 0, sizeof(it->second.bits));
  }
}

// Shader default constants (three definition-table groups). Group 1
// (register/count/physicalOffset records) is applied by the CP at shader
// load on hardware, so apply it here at bind time. Groups 2/3 are applied
// by the recompiled Set*Shader body. Live binds write the device float
// shadow; replayed CB binds write g_state.
void ApplyShaderFloatDefaults(GuestAddr device, bool pixel,
                              GuestAddr container, uint32_t phys_base,
                              bool replaying) {
  const uint32_t def_off = LoadGuestU32(container + 20);
  if (!def_off || def_off > 0x8000) return;
  GuestAddr def = container + def_off + 20;
  // Table extent: dword at (container+def_off)+16 = byte size of the record
  // area starting at (container+def_off)+20 (XDK walk 0x82523618:
  // end = table + size + 20).
  const GuestAddr def_end =
      container + def_off + 20 + LoadGuestU32(container + def_off + 16);
  for (uint32_t guard = 0; guard < 256; guard++, def += 8) {
    const uint32_t head = LoadGuestU32(def);
    const uint32_t count_floats = head & 0xFFFF;
    if (!count_floats) break;  // group-1 terminator (count == 0)
    if (!phys_base) continue;  // group-1 data lives in the physical blob
    const uint32_t reg = head >> 16;  // ALU-space index (PS regs are 256+)
    const uint32_t phys_off = LoadGuestU32(def + 4);
    const uint32_t reg_base = pixel ? reg - 256 : reg;
    if (reg_base >= 256) continue;
    const uint32_t vec4s = std::min((count_floats + 3) / 4, 256 - reg_base);
    if (replaying) {
      renderer::SetShaderConstantsF(pixel, reg_base, phys_base + phys_off,
                                    vec4s);
    } else if (device) {
      // m_Constants.Alu at dev+1920 spans VS 0-255 and PS 256-511; both
      // sides big-endian, raw copy. Registers the game has set since the
      // last draw stay untouched (XDK contract: the draw flush loads the
      // shader defaults first, then re-emits the dirty ranges, see
      // g_floatDirty above).
      auto* mem = rex::system::kernel_memory();
      for (uint32_t v = 0; v < vec4s; v++) {
        if (FloatRegDirty(device, pixel, reg_base + v)) {
          continue;
        }
        std::memcpy(
            mem->TranslateVirtual<void*>(device + 1920 + (reg + v) * 16),
            mem->TranslateVirtual<const void*>(phys_base + phys_off + v * 16),
            16);
      }
    }
  }
  if (!replaying) return;
  // Groups 2/3 never run for a replayed bind (the recompiled body only
  // runs live), so reproduce them here. After the group-1 terminator:
  //   group 2: u16 byteOffset, u16 dwordCount, inline BE dwords memcpy'd.
  //   group 3: u16 byteOffset, u16 n, then n/2 (mask, or) pairs merged.
  // Only ALU-float destinations are mirrored.
  GuestAddr p = def + 4;  // past the group-1 terminator
  for (uint32_t guard = 0; guard < 1024 && p + 4 <= def_end; guard++) {
    const uint32_t head = LoadGuestU32(p);
    const uint32_t byte_off = head >> 16;
    const uint32_t dwords = head & 0xFFFF;
    p += 4;
    if (!dwords) break;
    for (uint32_t i = 0; i < dwords; i++) {
      const uint32_t dst = byte_off + i * 4;
      const uint32_t value = LoadGuestU32(p + i * 4);
      if (dst < 4096) {
        renderer::PatchShaderFloatDwordsHost(false, dst / 4, &value, 1);
      } else if (dst < 8192) {
        renderer::PatchShaderFloatDwordsHost(true, (dst - 4096) / 4, &value,
                                             1);
      }
    }
    p += dwords * 4;
  }
  for (uint32_t guard = 0; guard < 1024 && p + 4 <= def_end; guard++) {
    const uint32_t head = LoadGuestU32(p);
    const uint32_t byte_off = head >> 16;
    const uint32_t n = head & 0xFFFF;
    p += 4;
    if (!n) break;
    for (uint32_t pair = 0; pair < n / 2 && p + 8 <= def_end; pair++) {
      const uint32_t mask = LoadGuestU32(p);
      const uint32_t orv = LoadGuestU32(p + 4);
      p += 8;
      const uint32_t dst = byte_off + pair * 4;
      if (dst < 4096) {
        renderer::MergeShaderFloatDwordHost(false, dst / 4, mask, orv);
      } else if (dst < 8192) {
        renderer::MergeShaderFloatDwordHost(true, (dst - 4096) / 4, mask, orv);
      }
    }
  }
}

// --- command buffers --------------------------------------------------------
// The game records public D3DDevice_* calls on the secondary device between
// Begin/End and replays them with Run. The hooked call stream is captured
// per CB object; replay re-invokes the same functions, so run-varying state
// is inherited live per the XDK contract.

enum class CbOp : uint8_t {
  kSetRenderTarget,
  kSetDepthStencilSurface,
  kSetSurfaces,
  kClear,
  kSetViewport,
  kSetScissorRect,
  kSetTexture,
  kSetStreamSource,
  kSetIndices,
  kSetVertexDeclaration,
  kSetVertexShader,
  kSetPixelShader,
  kSetVertexShaderConstantsF,
  kSetPixelShaderConstantsF,
  kResolve,
  kDrawVertices,
  kDrawIndexedVertices,
  kDrawVerticesUP,
  kDrawIndexedVerticesUP,
  kPredicationMark,
  kRunNested,
  kSetBlendControlDirect,
};

struct CbCall {
  CbOp op;
  bool is_draw;     // render-state snapshot in st[]/bool_loop[] is valid
  uint32_t a[9];
  // Render-state shadows of the recording device at record time (the game
  // sets scene depth/cull/blend while recording; those writes land in the
  // recording device's block, never the main device's): RB_DEPTHCONTROL,
  // RB_BLENDCONTROL0, RB_COLORCONTROL, PA_SU_SC_MODE_CNTL, RB_COLOR_MASK,
  // RB_ALPHA_REF, raw ZEnable request (dev+12308, the shadow's bit1 is
  // gated on a DS surface being bound through the replaced XDK setters, so
  // it is never valid in native mode), RB_BLENDCONTROL1-3.
  uint32_t st[10];
  // Bool/loop constant block (dev+10112, 40 dwords, host order) of the
  // recording device at record time; the recompiled SetShaderConstantB/I
  // setters are unhooked and write the recording device's shadows (the
  // avatar/bake passes set their branch bools while recording).
  uint32_t bool_loop[40];
  // Bool/loop dwords that changed on the recording device since Begin
  // (what the XDK's recorded flush would bake into the CB). Replay
  // applies only these; the rest inherit the run device's live shadow.
  uint64_t bool_dirty;
  // UP-draw payload captured at record time: raw guest vertex bytes, then
  // (indexed variant) raw index bytes at up_index_offset. The XDK copies UP
  // data into the CB; replay must not re-read the guest scratch, which the
  // game rewrites between the CB's predicated Runs.
  std::vector<uint8_t> up_data;
  uint32_t up_index_offset = 0;
  // Recording-device fetch-constant shadow slots that changed since Begin /
  // the last recorded draw: 7-dword records {slot, dw0..dw5}. The game's
  // FX-material and precache code edits the fetch shadow directly
  // (sub_82A49948 clamp-bit masks on dev+1152/+1176 with dirty marks;
  // streamer pokes) and the XDK recorded flush bakes those pokes into the
  // CB; a kSetTexture record alone carries only the header pointer.
  std::vector<uint32_t> fetch_data;
  // Which st[] slots the game programmed on the recording
  // device; untouched slots must inherit the run device's live state at
  // replay (the XDK bakes only dirty registers into a CB). Tracked for
  // the blend/mask slots; the rest are always authoritative.
  uint16_t st_dirty;
  // Payload (host-order dwords copied at record time; the guest source
  // pointers are stack temporaries). Constant-set ops carry the raw vec4
  // data; draws carry the record-time dirty float constants as 5-dword
  // records, since most are written through inlined paths the public
  // entries never see.
  std::vector<uint32_t> data;
  double f;
};

std::unordered_map<uint32_t, std::vector<CbCall>> g_commandBuffers;
uint32_t g_recordingCb = 0;
GuestAddr g_recordingDevice = 0;
// Last device ever used for recording (dev2). Run-time seeding consults its
// float shadow for registers the run device never set, see
// renderer::SeedReplayFloatConstants.
GuestAddr g_lastRecordingDevice = 0;
// Last device passed to SetPixelShader.
GuestAddr g_lastSetDevice = 0;
// Main (type-1) guest device block, owner of the D3DGAMMARAMP the game
// maintains at dev+15408 through the recompiled D3DDevice_SetGammaRamp.
std::unordered_map<uint32_t, uint32_t> g_deviceScratch;
GuestAddr g_mainGuestDevice = 0;
// Record-time float-constant baseline (host order; [0] = VS regs at
// dev+1920, [1] = PS regs at dev+6016): snapshotted at BeginCommandBuffer,
// advanced as recorded draws capture their dirty registers. Mirrors the
// XDK's dirty-range tracking, registers untouched during recording stay
// out of the CB and inherit the live device state at Run.
uint32_t g_recordFloatBaseline[2][256 * 4];
// Bool/loop baseline for the recorded draws' dirty capture (see
// CbCall::bool_dirty), recording device's dev+10112 block at Begin,
// advanced at each recorded draw.
uint32_t g_recordBoolBaseline[40];
// Fetch-constant shadow baseline (dev+1152, 32 slots x 6 dwords) for the
// recorded draws' dirty capture (see CbCall::fetch_data).
uint32_t g_recordFetchBaseline[32 * 6];
// Replay-time bool/loop base: seeded from the run device's live shadow at
// outermost RunCommandBuffer entry, recorded dirty dwords overlay per draw.
uint32_t g_replayBoolBase[40];
// Per-device bool/loop flush shadow: the block as of the last flush
// point. At Begin, dwords whose live value differs are pending and the
// XDK's first recorded-draw flush bakes exactly those into the CB; the
// setters are inlined (unhookable), so pending is recovered by diffing.
std::unordered_map<uint32_t, std::array<uint32_t, 40>> g_boolFlushShadow;
uint64_t g_recordBoolPending = 0;
// Persistent effective bool/loop file: on hardware a CB's bool writes
// persist in ring state after the Run and the next CB inherits them,
// while the CPU shadow never sees them. Live flush points fold in only
// the shadow's pending dwords; recorded dirty overlays write through.
uint32_t g_boolPersist[40];
bool g_boolPersistInit = false;

// Fold the device shadow's pending (changed-since-last-flush) dwords into
// the persist file and advance the flush shadow, the bool analog of the
// blend pre-flush (ConsumeLiveStateDirty).
void FoldBoolPendingIntoPersist(GuestAddr device) {
  if (!device) return;
  auto it = g_boolFlushShadow.find(device);
  const bool have_shadow = it != g_boolFlushShadow.end();
  auto& s = g_boolFlushShadow[device];
  for (uint32_t d = 0; d < 40; d++) {
    const uint32_t live = LoadGuestU32(device + 10112 + d * 4);
    if (!g_boolPersistInit || !have_shadow || it->second[d] != live) {
      g_boolPersist[d] = live;
    }
    s[d] = live;
  }
  g_boolPersistInit = true;
}

// Registers dirty on the recording device at Begin: the XDK's recorded
// flush also bakes registers set before Begin that no draw has flushed
// yet, which a diff-vs-baseline capture alone misses. Cleared once the
// first recorded draw captures them.
uint32_t g_recordPendingDirty[2][8];
// CbCall::st_dirty reads the recording device's live dev+16 dirty bits at
// capture time, the XDK's own flush source (see Record()).
// CreateDevice's seeded defaults for the tracked slots (st index -> value).
constexpr uint32_t kStSeedBlend = 0x00010001u;  // BLENDCONTROL0-3: replace
constexpr uint32_t kStSeedMask = 0xFFFFu;       // RB_COLOR_MASK: all RTs RGBA

void SnapshotRecordFloatBaseline(GuestAddr device) {
  auto* mem = rex::system::kernel_memory();
  for (int stage = 0; stage < 2; stage++) {
    const uint32_t* src = mem->TranslateVirtual<const uint32_t*>(
        device + (stage ? 6016 : 1920));
    uint32_t* dst = g_recordFloatBaseline[stage];
    for (uint32_t i = 0; i < 256 * 4; i++) dst[i] = __builtin_bswap32(src[i]);
  }
  auto it = g_floatDirty.find(device);
  if (it != g_floatDirty.end()) {
    std::memcpy(g_recordPendingDirty, it->second.bits,
                sizeof(g_recordPendingDirty));
  } else {
    std::memset(g_recordPendingDirty, 0, sizeof(g_recordPendingDirty));
  }
  for (uint32_t i = 0; i < 32 * 6; i++) {
    g_recordFetchBaseline[i] = LoadGuestU32(device + 1152 + i * 4);
  }
  g_recordBoolPending = 0;
  auto shadow_it = g_boolFlushShadow.find(device);
  for (uint32_t d = 0; d < 40; d++) {
    g_recordBoolBaseline[d] = LoadGuestU32(device + 10112 + d * 4);
    // Pending-at-Begin: set since the last flush on this device, the XDK's
    // first recorded-draw flush bakes these.
    // No flush shadow yet (device never drew) = nothing provably pending.
    if (shadow_it != g_boolFlushShadow.end() &&
        shadow_it->second[d] != g_recordBoolBaseline[d]) {
      g_recordBoolPending |= 1ull << d;
    }
  }
}
// Depth-stencil binding per guest device (the XDK's dev+12832 shadow; the
// recompiled setters that maintained it are replaced by these hooks).
std::unordered_map<uint32_t, uint32_t> g_dsBoundPerDevice;

bool Recording(GuestAddr device) {
  return g_recordingCb != 0 && device == g_recordingDevice;
}

void Record(CbOp op, std::initializer_list<uint32_t> args, double f = 0.0,
            bool is_draw = false, std::vector<uint32_t> payload = {}) {
  CbCall call{};
  call.op = op;
  call.f = f;
  call.is_draw = is_draw;
  call.data = std::move(payload);
  size_t i = 0;
  for (uint32_t v : args) call.a[i++] = v;
  if ((is_draw || op == CbOp::kRunNested) && g_recordingDevice) {
    call.st[0] = LoadGuestU32(g_recordingDevice + 10548);  // RB_DEPTHCONTROL
    call.st[1] = LoadGuestU32(g_recordingDevice + 10552);  // RB_BLENDCONTROL0
    call.st[2] = LoadGuestU32(g_recordingDevice + 10556);  // RB_COLORCONTROL
    call.st[3] = LoadGuestU32(g_recordingDevice + 10568);  // PA_SU_SC_MODE_CNTL
    call.st[4] = LoadGuestU32(g_recordingDevice + 10580);  // RB_COLOR_MASK
    call.st[5] = LoadGuestU32(g_recordingDevice + 10500);  // RB_ALPHA_REF
    // Effective z-enable composed at record time like the XDK does: raw
    // request && DS surface bound on the recording device (the recorded PM4
    // DepthControl carries this composition; live replay-time bindings are
    // irrelevant).
    const uint32_t raw_z = LoadGuestU32(g_recordingDevice + 12308);
    call.st[6] = (raw_z & 1u) && g_dsBoundPerDevice[g_recordingDevice] ? 1 : 0;
    call.st[7] = LoadGuestU32(g_recordingDevice + 10584);  // RB_BLENDCONTROL1
    call.st[8] = LoadGuestU32(g_recordingDevice + 10588);  // RB_BLENDCONTROL2
    call.st[9] = LoadGuestU32(g_recordingDevice + 10592);  // RB_BLENDCONTROL3
    // Authoritative-slot mask: the recording device's live dirty bits at
    // capture time, drained at capture, exactly the XDK flush cycle.
    // Recording bypasses the game's wrapper redundancy caches, so every
    // state request during recording emits and dirties; the live bits are
    // the baked-state truth.
    call.st_dirty = 0xFFFFu & ~((1u << 1) | (1u << 4) | (1u << 7) |
                                (1u << 8) | (1u << 9));
    {
      const uint64_t q = (uint64_t(LoadGuestU32(g_recordingDevice + 16)) << 32) |
                         LoadGuestU32(g_recordingDevice + 20);
      if (q & (1ull << 10)) call.st_dirty |= 1u << 1;  // RB_BLENDCONTROL0
      if (q & (1ull << 3)) call.st_dirty |= 1u << 4;   // RB_COLOR_MASK
      if (q & (1ull << 2)) call.st_dirty |= 1u << 7;   // RB_BLENDCONTROL1
      if (q & (1ull << 1)) call.st_dirty |= 1u << 8;   // RB_BLENDCONTROL2
      if (q & (1ull << 0)) call.st_dirty |= 1u << 9;   // RB_BLENDCONTROL3
      const uint64_t drained =
          q & ~((1ull << 10) | (1ull << 3) | (1ull << 2) | (1ull << 1) | 1ull);
      StoreGuestU32(g_recordingDevice + 16, uint32_t(drained >> 32));
      StoreGuestU32(g_recordingDevice + 20, uint32_t(drained));
    }
    // Bool/loop constants of the recording device (host order), with a
    // dirty mask = pending-at-Begin plus changed-since-baseline (the exact
    // float-capture model), replay overlays only these dwords onto the run
    // device's live shadow (see CbCall::bool_dirty / g_recordBoolPending).
    call.bool_dirty = g_recordBoolPending;
    for (uint32_t d = 0; d < 40; d++) {
      call.bool_loop[d] = LoadGuestU32(g_recordingDevice + 10112 + d * 4);
      if (call.bool_loop[d] != g_recordBoolBaseline[d]) {
        call.bool_dirty |= 1ull << d;
        g_recordBoolBaseline[d] = call.bool_loop[d];
      }
    }
    // Fetch-constant shadow capture: bake slots whose current
    // recording-device shadow differs from the baseline; the XDK flush
    // emits exactly the dirty fetch constants (incl. the direct pokes).
    for (uint32_t slot = 0; slot < 32; slot++) {
      uint32_t cur[6];
      bool changed = false;
      for (uint32_t i = 0; i < 6; i++) {
        cur[i] = LoadGuestU32(g_recordingDevice + 1152 + (slot * 6 + i) * 4);
        if (cur[i] != g_recordFetchBaseline[slot * 6 + i]) changed = true;
      }
      if (!changed) continue;
      call.fetch_data.push_back(slot);
      call.fetch_data.insert(call.fetch_data.end(), cur, cur + 6);
      std::memcpy(&g_recordFetchBaseline[slot * 6], cur, sizeof(cur));
    }
    // The recorded flush emitted the pending set, cleared like the XDK;
    // this flush is also a flush point for the recording device's shadow.
    g_recordBoolPending = 0;
    {
      auto& s = g_boolFlushShadow[g_recordingDevice];
      std::memcpy(s.data(), call.bool_loop, sizeof(call.bool_loop));
    }
    // Dirty float constants (see CbCall::data): capture every register of
    // the recording device's float shadow that changed since Begin / the
    // last recorded draw, and advance the baseline, exactly the registers
    // the XDK's draw flush would bake into the CB.
    auto* mem = rex::system::kernel_memory();
    for (int stage = 0; stage < 2; stage++) {
      const uint32_t* shadow = mem->TranslateVirtual<const uint32_t*>(
          g_recordingDevice + (stage ? 6016 : 1920));
      uint32_t* baseline = g_recordFloatBaseline[stage];
      for (uint32_t reg = 0; reg < 256; reg++) {
        uint32_t v[4];
        // Dirty = changed since Begin / the last recorded draw, or carried
        // dirty into Begin (set on the recording device before recording
        // started and not flushed by a draw since, the XDK flush bakes
        // those into the CB too; see g_recordPendingDirty).
        bool dirty =
            (g_recordPendingDirty[stage][reg >> 5] >> (reg & 31)) & 1u;
        for (int j = 0; j < 4; j++) {
          v[j] = __builtin_bswap32(shadow[reg * 4 + j]);
          dirty |= v[j] != baseline[reg * 4 + j];
        }
        if (!dirty) continue;
        call.data.push_back(reg | (stage ? 0x10000u : 0u));
        call.data.insert(call.data.end(), v, v + 4);
        std::memcpy(&baseline[reg * 4], v, 16);
      }
    }
    // The draw's flush emitted every pending range, clear like the XDK.
    std::memset(g_recordPendingDirty, 0, sizeof(g_recordPendingDirty));
  }
  g_commandBuffers[g_recordingCb].push_back(call);
}

void ReplayCall(GuestAddr device, const CbCall& c);  // after the public API

bool g_replaying = false;

// vsync / resolution_scale are registered by the rexgpu-xenos plugin. On a
// plugin-less native boot nobody registers them, so their config values sit
// deferred in the cvar registry and the renderer's by-name queries (see
// renderer.cpp) would only see compiled-in defaults. Register compatible
// entries here when absent: late registration replays the deferred config
// values onto them. By Init() time the plugin, if configured, has already
// loaded, so presence of the names is settled.
void RegisterVideoCvarFallbacks() {
  static bool s_vsync = true;
  static int32_t s_resScale = 1;
  if (!rex::cvar::GetFlagInfo("vsync")) {
    rex::cvar::RegisterFlag({"vsync",
                             rex::cvar::FlagType::Boolean,
                             "GPU",
                             "Enable vertical sync",
                             [](std::string_view v) {
                               s_vsync = rex::string::from_string<bool>(v);
                               return true;
                             },
                             []() -> std::string {
                               return s_vsync ? "true" : "false";
                             },
                             [](std::string_view) {},
                             rex::cvar::Lifecycle::kHotReload,
                             {},
                             "true",
                             false});
  }
  if (!rex::cvar::GetFlagInfo("resolution_scale")) {
    rex::cvar::RegisterFlag({"resolution_scale",
                             rex::cvar::FlagType::Int32,
                             "GPU",
                             "Draw resolution scale for both X and Y axes",
                             [](std::string_view v) {
                               int32_t val = 0;
                               auto [p, ec] = std::from_chars(
                                   v.data(), v.data() + v.size(), val);
                               if (ec != std::errc()) return false;
                               s_resScale = val;
                               return true;
                             },
                             []() -> std::string {
                               return std::to_string(s_resScale);
                             },
                             [](std::string_view) {},
                             rex::cvar::Lifecycle::kRequiresRestart,
                             {},
                             "1",
                             false});
  }
}

}  // namespace

namespace detail {
plume::RenderDevice* Device() { return g_device.get(); }
plume::RenderCommandQueue* Queue() { return g_queue.get(); }
// Render-queue worker override: thread-local so the worker never writes
// the guest-thread globals (a global write races the guest thread
// resolving the next draw while the worker executes an older one).
thread_local const ResolvedShader* t_ovVs = nullptr;
thread_local const ResolvedShader* t_ovPs = nullptr;
thread_local uint32_t t_ovPsBound = 0;
thread_local bool t_ovValid = false;
const ResolvedShader* CurrentResolvedVertexShader() {
  return t_ovValid ? t_ovVs : g_currentResolvedVs;
}
const ResolvedShader* CurrentResolvedPixelShader() {
  return t_ovValid ? t_ovPs : g_currentResolvedPs;
}
bool CurrentDrawPixelShaderBound() {
  return t_ovValid ? t_ovPsBound != 0 : g_currentPixelShader != 0;
}
void OverrideResolvedShaders(const ResolvedShader* vs,
                             const ResolvedShader* ps, bool ps_bound) {
  t_ovVs = vs;
  t_ovPs = ps;
  t_ovPsBound = ps_bound ? 1u : 0u;
  t_ovValid = true;
}
void ClearResolvedShaderOverride() { t_ovValid = false; }
const std::vector<uint8_t>& RectExpandGsDxil() {
  EnsureShaderPackLoaded();
  return g_shaderCache.rect_expand_gs();
}
const std::vector<uint8_t>& PointExpandGsDxil() {
  EnsureShaderPackLoaded();
  return g_shaderCache.point_expand_gs();
}
const std::vector<uint8_t>& BlitVsDxil() {
  EnsureShaderPackLoaded();
  return g_shaderCache.blit_vs();
}
const std::vector<uint8_t>& BlitPsDxil() {
  EnsureShaderPackLoaded();
  return g_shaderCache.blit_ps();
}
}  // namespace detail

void SetVfetchPatcher(VfetchPatchFn fn) { g_vfetchPatcher = fn; }

bool Enabled() { return REXCVAR_GET(native_video); }

bool Init() {
  if (g_initialized.load()) return true;

  std::lock_guard lock(g_initMutex);
  if (g_initialized.load()) return true;
  if (g_initFailed) return false;  // don't retry every call

  RegisterVideoCvarFallbacks();

  // Optional guest-heap shift: applied before anything the game allocates
  // after this point (level load happens later).
  const int32_t pad = REXCVAR_GET(native_video_heap_pad);
  if (pad > 0) {
    rex::system::kernel_memory()->SystemHeapAlloc(uint32_t(pad), 0x100);
  }

  // Device creation retries: the first attempt can transiently fail with
  // DEVICE_RESET when racing the ring backend's early GPU init; back off
  // and retry with a fresh factory.
#ifdef _WIN32
  const std::string& adapter_pick = REXCVAR_GET(native_video_adapter);
  if (!adapter_pick.empty()) {
    SetEnvironmentVariableA("REXGLUE_ADAPTER", adapter_pick.c_str());
  }
#endif
  for (int attempt = 0; attempt < 6 && !g_device; attempt++) {
    if (attempt > 0) {
      REXGPU_WARN(
          "videonative: plume device creation retry {} (transient adapter "
          "reset?)",
          attempt);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    g_interface = plume::CreateD3D12Interface();
    if (!g_interface) {
      REXGPU_ERROR("videonative: plume D3D12 interface creation failed");
      continue;
    }
    g_device = g_interface->createDevice();
    if (!g_device) g_interface.reset();
  }
#ifdef _WIN32
  // An adapter filter that matches nothing must not brick the boot: warn
  // loudly and fall back to the default adapter.
  if (!g_device && !adapter_pick.empty()) {
    REXGPU_WARN(
        "videonative: native_video_adapter '{}' matched no adapter, "
        "falling back to the default adapter",
        adapter_pick);
    SetEnvironmentVariableA("REXGLUE_ADAPTER", nullptr);
    g_interface = plume::CreateD3D12Interface();
    if (g_interface) {
      g_device = g_interface->createDevice();
      if (!g_device) g_interface.reset();
    }
  }
#endif
  if (g_device) {
    const std::string& adapter_name = g_device->getDescription().name;
    REXGPU_INFO("videonative: adapter '{}'", adapter_name);
    // Per-vendor gates (native_video_vs_trim mode 2): a hybrid-GPU machine
    // can select either adapter from one toml.
    renderer::SetAdapterIsIntel(adapter_name.find("Intel") !=
                                std::string::npos);
  }
  if (!g_device) {
    REXGPU_ERROR("videonative: plume device creation failed");
    g_interface.reset();
    g_initFailed = true;
    return false;
  }
  g_queue = g_device->createCommandQueue(plume::RenderCommandListType::DIRECT);

  if (!renderer::Init()) {
    REXGPU_ERROR("videonative: renderer init failed");
    g_queue.reset();
    g_device.reset();
    g_interface.reset();
    g_initFailed = true;
    return false;
  }
  rq::Start();  // render-thread offload (no-op unless the cvar is on)

  // Kernel-side guest-memory writers (XamAvatarGetAssets buffer rewrites)
  // call InvalidateGuestTextureRange / FreezeGuestTextureRange directly.

  const auto& desc = g_device->getDescription();
  REXGPU_INFO("videonative: plume D3D12 device up on '{}' ({} MB VRAM)",
              desc.name, desc.dedicatedVideoMemory >> 20);

  // Leak the plume device/queue/interface at CRT exit (see the
  // renderer's swapchain exit guard, same DXGI-teardown deadlock class).
  static bool vn_exit_guard = false;
  if (!vn_exit_guard) {
    vn_exit_guard = true;
    std::atexit([] {
      g_queue.release();
      g_device.release();
      g_interface.release();
    });
  }

  g_initialized.store(true);
  return true;
}

void Shutdown() {
  std::lock_guard lock(g_initMutex);
  if (!g_initialized.load()) return;
  renderer::Shutdown();
  g_queue.reset();
  g_device.reset();
  g_interface.reset();
  g_initialized.store(false);
}

bool IsActive() { return Enabled() && g_initialized.load(); }

// --- device / swap ---

uint32_t CreateDevice(GuestAddr, uint32_t type, uint32_t, uint32_t,
                      GuestAddr present_params, GuestAddr out_device) {
  if (!Init()) return 0x80004005u;  // E_FAIL, caller falls back to guest path

  auto* mem = rex::system::kernel_memory();
  uint32_t guest_device = mem->SystemHeapAlloc(kGuestDeviceSize, 0x100);
  if (!guest_device) return 0x8007000Eu;  // E_OUTOFMEMORY
  std::memset(mem->TranslateVirtual<void*>(guest_device), 0, kGuestDeviceSize);

  // Shadow-register defaults the real XDK device init would have written
  // (a zero RB_COLOR_MASK would mask every color write). Also fabricate
  // the XDK's g_pDevice/g_pDeviceAux process globals at 0x82000864/68;
  // the page can be mapped read-only, so unprotect before seeding.
  if (auto* heap = const_cast<rex::memory::BaseHeap*>(
          mem->LookupHeap(0x82000864u))) {
    heap->Protect(0x82000000u, 0x1000,
                  rex::memory::kMemoryProtectRead |
                      rex::memory::kMemoryProtectWrite,
                  nullptr);
  }
  // Double indirection: the header slots hold pointers to the XDK's
  // g_pDevice/g_pDeviceAux variables (readers do [slot] -> [ptr] -> device).
  {
    const uint32_t devvars = mem->SystemHeapAlloc(16, 0x10);
    StoreGuestU32(devvars + 0, guest_device);
    StoreGuestU32(devvars + 4, guest_device);
    StoreGuestU32(0x82000864u, devvars + 0);
    StoreGuestU32(0x82000868u, devvars + 4);
  }
  StoreGuestU32(guest_device + 10580, 0xFFFF);  // RB_COLOR_MASK: RT0-3 RGBA on
  // RB_BLENDCONTROL0-3 default = one/ADD/zero both pipes (blending disabled).
  // A zeroed shadow decodes as srcblend zero (0) = "multiply everything by
  // zero" for any RT the game never programs.
  StoreGuestU32(guest_device + 10552, 0x00010001);  // RB_BLENDCONTROL0
  StoreGuestU32(guest_device + 10584, 0x00010001);  // RB_BLENDCONTROL1
  StoreGuestU32(guest_device + 10588, 0x00010001);  // RB_BLENDCONTROL2
  StoreGuestU32(guest_device + 10592, 0x00010001);  // RB_BLENDCONTROL3

  // Real CB scratch segment + fence pool. The XDK appender idiom
  // is `p=*(dev+48); if (p > *(dev+56)) grow; write at p+4; *(dev+48)=end`;
  // with both zero the packets land in guest page zero, which must stay
  // zero (another title's benign null-object reads depend on it; ring keeps it
  // pristine because its dev+48 points into a real ring segment).
  if (REXCVAR_GET(native_video_device_ring_scratch)) {
    constexpr uint32_t kScratchSize = 64 * 1024;
    const uint32_t scratch = mem->SystemHeapAlloc(kScratchSize, 0x100);
    const uint32_t pool = mem->SystemHeapAlloc(256, 0x100);
    if (scratch && pool) {
      std::memset(mem->TranslateVirtual<void*>(scratch), 0, kScratchSize);
      std::memset(mem->TranslateVirtual<void*>(pool), 0, 256);
      StoreGuestU32(guest_device + 48, scratch);
      StoreGuestU32(guest_device + 56, scratch + kScratchSize - 0x200);
      StoreGuestU32(guest_device + 11024, pool);
      // XDK init idiom: current fence 3, completed = current-2.
      StoreGuestU32(guest_device + 11036, 3);
      // Instant-completion pool: with native_video_fence_pool_max the
      // completed counter starts (and stays) above any fence value any path
      // can issue, so a loader blocking on 'completed >= resource fence'
      // resumes immediately. The default keeps the retail idiom (pool[0]=1,
      // advanced per present).
      StoreGuestU32(pool, REXCVAR_GET(native_video_fence_pool_max)
                              ? 0xFFFFFF00u
                              : 1u);
      g_deviceScratch[guest_device] = scratch;
    }
  }
  StoreGuestU32(out_device, guest_device);
  if (type == 1) g_mainGuestDevice = guest_device;
  REXGPU_INFO(
      "videonative: CreateDevice type={} presentParams={:#x} -> guest device "
      "{:#x}",
      type, present_params, guest_device);
  return 0;  // S_OK
}

// GPU fence writeback emulation: the XDK inserts EVENT_WRITE_SHD fence
// packets that nothing consumed natively (swap throttle, release
// deferral, VB reuse and the FX aging clock all read them). The insert
// function is hooked; queued writebacks execute at the next Swap with an
// incrementing counter, one frame behind like real hardware.
std::mutex g_fenceMutex;
std::vector<GuestAddr> g_fencePendingThisFrame;
std::vector<GuestAddr> g_fenceDueNextSwap;
uint32_t g_fenceCounter = 0x1000;

void QueueFenceWriteback(GuestAddr addr) {
  if (!addr) return;
  std::lock_guard lock(g_fenceMutex);
  g_fencePendingThisFrame.push_back(addr);
}

void Swap(GuestAddr device, GuestAddr front_buffer_tex, GuestAddr) {
  g_frameCount.fetch_add(1);
  // Resource-fence progression: D3DResource_Release defers frees while a
  // resource is in flight by comparing its fence against dev+11024, so the
  // counters have to advance. Also maintains the skipped Swap body's
  // per-frame fence ring.
  if (device && REXCVAR_GET(native_video_swap_bookkeeping)) {
    const uint32_t pool = LoadGuestU32(device + 11024);
    const uint32_t f = LoadGuestU32(device + 22036);
    if (pool) {
      uint32_t cur = LoadGuestU32(device + 11036);
      StoreGuestU32(pool + 4 * (16 + (f & 7)), ++cur);
      StoreGuestU32(pool + 4 * (16 + ((f + 1) & 7)), ++cur);
      StoreGuestU32(pool + 8, cur);      // VdSwap frontbuffer writeback slot
      StoreGuestU32(device + 11036, cur);
      if (!REXCVAR_GET(native_video_fence_pool_max)) {
        StoreGuestU32(pool, cur - 1);    // completed-fence idiom
      }
    }
    StoreGuestU32(device + 22036, f + 2);
    StoreGuestU32(device + 22032, f + 2);  // complete instantly (no GPU lag)
  }
  // Re-arm the CB scratch write pointer each present (nothing consumes the
  // scratch; it only absorbs stray PM4 appends).
  if (device) {
    auto it_scratch = g_deviceScratch.find(device);
    if (it_scratch != g_deviceScratch.end()) {
      StoreGuestU32(device + 48, it_scratch->second);
    }
  }
  if (device && REXCVAR_GET(native_video_fence_emulation)) {
    const uint32_t cur = LoadGuestU32(device + 11036) + 1;
    StoreGuestU32(device + 11036, cur);
    const uint32_t wb = LoadGuestU32(device + 11024);
    // Guard: only write through wb when it plausibly is a guest pointer;
    // natively the Swap body that initializes it is skipped, and writing
    // through a non-pointer dword corrupts guest memory.
    if (wb >= 0x10000 && wb < 0xC0000000u) StoreGuestU32(wb, cur - 1);
  }
  {
    // Complete last frame's fences, promote this frame's to next.
    std::lock_guard lock(g_fenceMutex);
    for (GuestAddr a : g_fenceDueNextSwap) {
      StoreGuestU32(a & ~3u, ++g_fenceCounter);
    }
    g_fenceDueNextSwap.swap(g_fencePendingThisFrame);
    g_fencePendingThisFrame.clear();
  }
  // Display gamma ramp: the recompiled SetGammaRamp maintains a
  // D3DGAMMARAMP in the device block; forward it to the present blit's
  // LUT whenever the block changes. The offset is title-dependent
  // (+15408 vs +15216), so score both candidates as a plausible ramp and
  // use the winner; the cvar forces one when needed.
  if (g_mainGuestDevice) {
    static std::array<uint16_t, 768> cached{};
    static bool have_ramp = false;
    static uint32_t chosen_offset = 0;
    auto read_ramp = [&](uint32_t off, std::array<uint16_t, 768>& out) -> bool {
      const uint8_t* p = rex::system::kernel_memory()
                             ->TranslateVirtual<const uint8_t*>(
                                 g_mainGuestDevice + off);
      if (!p) return false;
      for (uint32_t i = 0; i < 768; i++) {
        out[i] = uint16_t((p[i * 2] << 8) | p[i * 2 + 1]);
      }
      return true;
    };
    // A valid ramp: each 256-entry channel is non-decreasing and ends far
    // above where it starts. Garbage device bytes fail both tests.
    auto score_ramp = [](const std::array<uint16_t, 768>& r) -> int {
      int score = 0;
      for (uint32_t ch = 0; ch < 3; ch++) {
        const uint16_t* c = r.data() + ch * 256;
        uint32_t monotonic = 0;
        for (uint32_t i = 1; i < 256; i++) {
          if (c[i] >= c[i - 1]) monotonic++;
        }
        if (monotonic >= 250) score += 2;      // near-monotonic
        if (c[255] > c[0] + 4096) score += 1;  // real dynamic range
      }
      return score;  // 9 = perfect on all three channels
    };
    if (!chosen_offset) {
      const int32_t forced = REXCVAR_GET(native_video_gamma_ramp_offset);
      if (forced > 0) {
        chosen_offset = uint32_t(forced);
        REXGPU_INFO("videonative: gamma ramp offset forced to +{}", forced);
      } else {
        std::array<uint16_t, 768> a{}, b{};
        const bool ok_a = read_ramp(15408, a);
        const bool ok_b = read_ramp(15216, b);
        const int sa = ok_a ? score_ramp(a) : -1;
        const int sb = ok_b ? score_ramp(b) : -1;
        if (sa >= 6 || sb >= 6) {
          chosen_offset = (sb > sa) ? 15216u : 15408u;
          REXGPU_INFO(
              "videonative: gamma ramp offset AUTO -> +{} (score +15408={} "
              "+15216={})",
              chosen_offset, sa, sb);
        }
        // Neither scores yet: the game has not written a ramp, retry next
        // present rather than latching a wrong offset.
      }
    }
    std::array<uint16_t, 768> host{};
    bool nonzero = false;
    if (chosen_offset && read_ramp(chosen_offset, host)) {
      for (uint32_t i = 0; i < 768; i++) nonzero |= host[i] != 0;
    }
    // All-zero = the game has not set a ramp yet (device block starts
    // zeroed), leave the LUT unbound rather than rendering black.
    if (nonzero && (!have_ramp || host != cached)) {
      cached = host;
      have_ramp = true;
      renderer::UpdateGammaRamp(host.data());
    }
  }
  // The frontbuffer texture (fetch header at +0x1C) carries the frame the
  // game resolved its final composite into, blit it onto the swapchain.
  if (front_buffer_tex) {
    renderer::SwapFrontbuffer(front_buffer_tex + 0x1C);
  }
  renderer::EndFrameAndPresent();
}

void AcquireThreadOwnership(GuestAddr) {}
void ReleaseThreadOwnership(GuestAddr) {}

// --- render targets / tiling ---

void SetRenderTarget(GuestAddr device, uint32_t index, GuestAddr surface) {
  if (Recording(device)) {
    Record(CbOp::kSetRenderTarget, {index, surface});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  // All four MRT slots are forwarded; slots 1-3 carry their own formats
  // (the lighting pass binds an 8888 specular accumulator on RT1).
  renderer::SetRenderTargetSurface(index, surface);
}
void SetDepthStencilSurface(GuestAddr device, GuestAddr surface, uint32_t) {
  g_dsBoundPerDevice[device] = surface;
  if (Recording(device)) {
    Record(CbOp::kSetDepthStencilSurface, {surface});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  renderer::SetDepthSurface(surface);
}
// D3DDevice_SetSurfaces(device, {DS, RT0..RT3}, flags); the unbind idiom is
// SetSurfaces(all-NULL, 1) after ResolveAndUnbind.
void SetSurfaces(GuestAddr device, GuestAddr surfaces, uint32_t) {
  const uint32_t ds = surfaces ? LoadGuestU32(surfaces + 0) : 0;
  uint32_t rt[4] = {};
  if (surfaces) {
    for (uint32_t i = 0; i < 4; i++) rt[i] = LoadGuestU32(surfaces + 4 + i * 4);
  }
  g_dsBoundPerDevice[device] = ds;
  if (Recording(device)) {
    Record(CbOp::kSetSurfaces, {ds, rt[0], rt[1], rt[2], rt[3]});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  renderer::SetDepthSurface(ds);
  for (uint32_t i = 0; i < 4; i++) renderer::SetRenderTargetSurface(i, rt[i]);
}
// D3DDevice_CreateSurface (0x8252BB30): (width, height, format, msaa,
// params) -> surface pointer; NO device arg, NO out param. The object gets
// the real XDK field layout (GetDesc and the engine EDRAM math read it back).
uint32_t CreateSurface(uint32_t width, uint32_t height, uint32_t format,
                       uint32_t msaa, GuestAddr params) {
  if (!width || !height || width > 8192 || height > 8192) return 0;
  auto* mem = rex::system::kernel_memory();
  const uint32_t obj = mem->SystemHeapAlloc(48, 0x10);
  if (!obj) return 0;
  std::memset(mem->TranslateVirtual<void*>(obj), 0, 48);
  StoreGuestU32(obj + 0, 0x100000u | (params ? 0 : 0x80000000u));
  StoreGuestU32(obj + 4, 1);           // refcount
  StoreGuestU32(obj + 20, 0xFFFF0000u);
  StoreGuestU32(obj + 24, msaa & 3);   // pitch/msaa dword (msaa bits only)
  StoreGuestU32(obj + 36, ((width - 1) << 18) | ((height - 1) << 3));
  StoreGuestU32(obj + 40, format);
  StoreGuestU32(obj + 44, 5120);
  renderer::RegisterSurface(obj, width, height, format);
  REXGPU_INFO("videonative: CreateSurface {}x{} fmt={:#x} msaa={} -> {:#x}",
              width, height, format, msaa, obj);
  return obj;
}
// D3DDevice_BeginTiling: r4=flags, r5=tile count, r6=rects, r7=clear
// color, f1=clearZ. Record the frame extent so the renderer expands the
// tile-height RT, honor the implicit clear, and capture tile state for
// EndTiling-driven resolves.
uint32_t g_tilingRectCount = 0;
GuestAddr g_tilingRects = 0;

void BeginTiling(GuestAddr, uint32_t flags, uint32_t count, GuestAddr rects,
                 GuestAddr clear_color, float clear_z, uint32_t) {
  g_tilingRectCount = count;
  g_tilingRects = rects;
  uint32_t max_right = 0, max_bottom = 0;
  for (uint32_t i = 0; i < count && i < 16; i++) {
    const uint32_t right = LoadGuestU32(rects + i * 16 + 8);
    const uint32_t bottom = LoadGuestU32(rects + i * 16 + 12);
    if (right > max_right && right <= 8192) max_right = right;
    if (bottom > max_bottom && bottom <= 8192) max_bottom = bottom;
  }
  renderer::SetTilingExtent(max_right, max_bottom);
  if ((flags & 1) == 0) {
    uint32_t color = 0;
    if (clear_color) {
      // D3DVECTOR4 {r,g,b,a} floats -> packed ARGB.
      float rgba[4];
      for (int i = 0; i < 4; i++) {
        const uint32_t bits = LoadGuestU32(clear_color + i * 4);
        std::memcpy(&rgba[i], &bits, 4);
      }
      auto to8 = [](float f) {
        return uint32_t(std::clamp(f, 0.0f, 1.0f) * 255.0f + 0.5f);
      };
      color = (to8(rgba[3]) << 24) | (to8(rgba[0]) << 16) |
              (to8(rgba[1]) << 8) | to8(rgba[2]);
    }
    renderer::Clear(0, 0x3F, color, clear_z);
  }
}
void EndTiling(GuestAddr device, uint32_t flags, GuestAddr rects,
               GuestAddr dest_texture) {
  // XDK D3DDevice_EndTiling (dec_0049 0x82531708): with a dest texture it
  // resolves every tile rect into it before ending the bracket. Engines that
  // resolve explicitly pass a null dest; others reach the frontbuffer only
  // through this path, so the dest argument has to be honored.
  if (dest_texture) {
    if (!rects) rects = g_tilingRects;
    const uint32_t count = g_tilingRectCount ? g_tilingRectCount : 1;
    for (uint32_t i = 0; i < count && i < 16; i++) {
      const GuestAddr rect = rects ? rects + i * 16 : 0;
      Resolve(device, flags, rect, dest_texture, rect);
    }
  }
  g_tilingRectCount = 0;
  g_tilingRects = 0;
  renderer::SetTilingExtent(0, 0);
}
// D3DDevice_SetPredication writes the bin mask for live commands; the
// game wraps each per-tile Run in it. Natively there is one full-frame
// pass, so the live mask at Run time is the effective bin select for
// that Run's spans (0 resets to all-ones).
uint32_t g_predicationSelectLo = 0xFFFFFFFFu;

// force=true skips the pending gate: under the render queue the push
// happens on the worker when the enqueued resolve executes, so a guest-side
// pending check right after enqueuing the resolve races to zero. The
// resolve path forces (the worker-side flush is a no-op when empty); the
// fence-hook path stays gated (its callers spin).
void FlushResolveWritebacksSyncEx(bool force) {
  if (!force && !renderer::ResolveWritebacksPending()) return;
  if (rq::Active()) {
    const uint64_t ticket = rq::EnqFlushResolveWriteback();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (rq::WbFlushDone() < ticket) {
      std::this_thread::yield();
      if (std::chrono::steady_clock::now() > deadline) {
        static uint64_t timeout_logs = 8;
        if (timeout_logs) {
          timeout_logs--;
          REXGPU_WARN(
              "videonative: [writeback] sync flush timed out (worker busy)");
        }
        break;
      }
    }
  } else {
    renderer::FlushResolveWritebacksInline();
  }
}

void FlushResolveWritebacksSync() { FlushResolveWritebacksSyncEx(false); }

bool WritebackSyncEnabled() { return renderer::WritebackSyncEnabled(); }

void NoteRawDevice(uint32_t device_ea) {
  renderer::NoteRawDevice(device_ea);
}

void InvalidateTextureByHeader(uint32_t header_addr) {
  renderer::InvalidateTextureByHeader(header_addr);
}

void InvalidateTextureAfterUnlock(uint32_t header_addr) {
  renderer::InvalidateTextureAfterUnlock(header_addr);
}

void InvalidateSmallAlphaTextures() {
  renderer::InvalidateSmallAlphaTextures();
}

void InvalidateGuestTextureRange(uint32_t guest_address, uint32_t size) {
  renderer::QueueGuestTextureInvalidate(guest_address, size);
}

void FreezeGuestTextureRange(uint32_t guest_address, uint32_t size) {
  renderer::QueueGuestTextureFreeze(guest_address, size);
}

void NoteRttBegin(uint32_t texbase, uint32_t width, uint32_t height,
                  uint32_t color_surf, uint32_t depth_surf, uint32_t dest_tex,
                  uint32_t format, uint32_t msaa, uint32_t tiling) {
  renderer::NoteRttBegin(texbase, width, height, color_surf, depth_surf,
                         dest_tex, format, msaa, tiling);
}
void SetPredication(GuestAddr device, uint32_t mask) {
  if (Recording(device)) {
    // While recording, SetPredication writes SET_BIN_MASK_LO into the CB,
    // same effect as a SetCommandBufferPredication tile mark.
    Record(CbOp::kPredicationMark, {mask, 0});
    return;
  }
  g_predicationSelectLo = mask ? mask : 0xFFFFFFFFu;
}

namespace {
// Shared resolve body: parameters already read out of guest memory (the
// recorded path must capture the rect/point by value, the guest pointers
// are stack temporaries in RENDERTEXTURE::ResolveAndUnbind).
void ResolveParsed(uint32_t flags, GuestAddr dest_texture, bool has_rect,
                   int32_t l, int32_t t, int32_t r, int32_t b, int32_t dx,
                   int32_t dy) {
  if (!dest_texture) return;
  const uint32_t source = flags & 7;
  if (source == 4) {
    // Depth resolve, feeds the depth-restore draw's VS fetch.
    renderer::ResolveDepthToTexture(dest_texture + 0x1C, has_rect, l, t, r, b,
                                    dx, dy);
    return;
  }
  // Sources 0-3 = the MRT color attachments (cached host RTs carry all four;
  // the lighting/AO bakes write oC0+oC1 and resolve them separately).
  renderer::ResolveToTexture(dest_texture + 0x1C, source, has_rect, l, t, r,
                             b, dx, dy);
  // Write-back delivery at the resolve itself: CPU readers poll the
  // (instant-signaled) fence pool in guest memory directly, so no hookable
  // wait ever runs and fence-time delivery never fires. Real hardware writes
  // RAM at the resolve; match it. Forced, because under the render queue the
  // push happens later on the worker and the pending gate would race to zero
  // here.
  if (renderer::ResolveWritebackEnabled() && renderer::WritebackSyncEnabled()) {
    FlushResolveWritebacksSyncEx(true);
  }
}
}  // namespace

// D3DDevice_Resolve: flags bits 0-2 select the source (0-3 = color RT0-3,
// 4 = depth); r5 = D3DRECT* srcRect (frame coords), r6 = dest texture object
// (fetch header at +0x1C), r7 = D3DPOINT* destPoint. Per-tile resolves pass
// the tile rect + the tile origin, honoring them reassembles the frame.
void Resolve(GuestAddr device, uint32_t flags, GuestAddr rect,
             GuestAddr dest_texture, GuestAddr dest_point) {
  const bool has_rect = rect != 0;
  int32_t l = 0, t = 0, r = 0, b = 0, dx = 0, dy = 0;
  if (rect) {
    l = int32_t(LoadGuestU32(rect + 0));
    t = int32_t(LoadGuestU32(rect + 4));
    r = int32_t(LoadGuestU32(rect + 8));
    b = int32_t(LoadGuestU32(rect + 12));
  }
  if (dest_point) {
    dx = int32_t(LoadGuestU32(dest_point + 0));
    dy = int32_t(LoadGuestU32(dest_point + 4));
  }
  if (Recording(device)) {
    Record(CbOp::kResolve,
           {flags, dest_texture, uint32_t(has_rect), uint32_t(l), uint32_t(t),
            uint32_t(r), uint32_t(b), uint32_t(dx), uint32_t(dy)});
    return;
  }
  ResolveParsed(flags, dest_texture, has_rect, l, t, r, b, dx, dy);
}

// --- clear / viewport / scissor ---
// pRects scope the clear (the per-tile setup command buffers clear only
// their band; a full clear would wipe the other tiles in the full-size RT).
void Clear(GuestAddr device, uint32_t count, GuestAddr rects, uint32_t flags,
           uint32_t color, double z, uint32_t stencil) {
  int32_t rect_vals[16] = {};
  uint32_t n = rects ? std::min(count, 4u) : 0;
  for (uint32_t i = 0; i < n * 4; i++) {
    rect_vals[i] = int32_t(LoadGuestU32(rects + i * 4));
  }
  if (Recording(device)) {
    // Captured by value (the guest rect array may be a stack temporary);
    // only the first rect fits the record, another title passes 0 or 1.
    Record(CbOp::kClear,
           {flags, color, n, uint32_t(rect_vals[0]), uint32_t(rect_vals[1]),
            uint32_t(rect_vals[2]), uint32_t(rect_vals[3])},
           z);
    return;
  }
  // (The CB-record path above does not carry stencil, its arg record is
  // full and no recorded stencil clear has been sighted; live clears do.)
  renderer::Clear(device, flags, color, float(z), rect_vals, n, stencil);
}
void SetViewport(GuestAddr device, GuestAddr viewport) {
  if (!viewport) return;
  const uint32_t x = LoadGuestU32(viewport + 0);
  const uint32_t y = LoadGuestU32(viewport + 4);
  const uint32_t w = LoadGuestU32(viewport + 8);
  const uint32_t h = LoadGuestU32(viewport + 12);
  const uint32_t minz_bits = LoadGuestU32(viewport + 16);
  const uint32_t maxz_bits = LoadGuestU32(viewport + 20);
  if (Recording(device)) {
    // Captured by value like the XDK's PM4 write.
    Record(CbOp::kSetViewport, {x, y, w, h, minz_bits, maxz_bits});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  float min_z, max_z;
  std::memcpy(&min_z, &minz_bits, 4);
  std::memcpy(&max_z, &maxz_bits, 4);
  renderer::SetViewport(x, y, w, h, min_z, max_z);
}
// Float-value variant for the SetViewport tail hook (0x921192D0): both
// D3DDevice_SetViewport and SetViewportF funnel there with the viewport in
// f1..f6, no guest struct to read. Mirrors the struct variant exactly,
// including the command-buffer Record (a recorded preview pass must carry
// its viewport like the XDK's own PM4 write would).
void SetViewportValues(GuestAddr device, float x, float y, float w, float h,
                       float min_z, float max_z) {
  const auto px = [](float v) {
    return v <= 0.0f ? 0u : uint32_t(v + 0.5f);
  };
  const uint32_t xi = px(x), yi = px(y), wi = px(w), hi = px(h);
  if (Recording(device)) {
    uint32_t minz_bits, maxz_bits;
    std::memcpy(&minz_bits, &min_z, 4);
    std::memcpy(&maxz_bits, &max_z, 4);
    Record(CbOp::kSetViewport, {xi, yi, wi, hi, minz_bits, maxz_bits});
    // fall through: apply live too (see SetViewport above).
  }
  renderer::SetViewport(xi, yi, wi, hi, min_z, max_z);
}
void SetScissorRect(GuestAddr device, GuestAddr rect) {
  if (!rect) return;
  const uint32_t l = LoadGuestU32(rect + 0);
  const uint32_t t = LoadGuestU32(rect + 4);
  const uint32_t r = LoadGuestU32(rect + 8);
  const uint32_t b = LoadGuestU32(rect + 12);
  if (Recording(device)) {
    Record(CbOp::kSetScissorRect, {l, t, r, b});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  renderer::SetScissor(int32_t(l), int32_t(t), int32_t(r), int32_t(b));
}

// --- resources ---
uint32_t CreateTexture(GuestAddr, uint32_t, uint32_t, uint32_t, uint32_t,
                       uint32_t, uint32_t, uint32_t, uint32_t) {
  return 0;
}
uint32_t CreateVertexBuffer(uint32_t, uint32_t) { return 0; }
uint32_t CreateIndexBuffer(uint32_t, uint32_t, uint32_t) { return 0; }
void ReleaseResource(GuestAddr) {}
void ReleaseDevice(GuestAddr) {}

// --- binding ---
void SetTexture(GuestAddr device, uint32_t sampler, GuestAddr texture) {
  if (Recording(device)) {
    Record(CbOp::kSetTexture, {sampler, texture});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  // The GPUTEXTURE_FETCH constant lives at +0x1C inside the XDK texture
  // object (after Common/RefCount/Fence/ReadFence/Identifier/BaseFlush/
  // MipFlush).
  // Fence stamp: the skipped XDK SetTexture body's first action,
  // mark the resource as in-flight this frame so Release defers its free.
  if (texture && REXCVAR_GET(native_video_fence_emulation)) {
    StoreGuestU32(texture + 8, LoadGuestU32(device + 11036));
  }
  renderer::SetTexture(sampler, texture ? texture + 0x1C : 0);
  // No shadow write here: the guest SetTexture body runs right after
  // this observe-hook and is the only correct writer (it merges the
  // header dwords preserving per-stage sampler fields, and a null unbind
  // clears just the type bits).
}
void SetStreamSource(GuestAddr device, uint32_t stream, GuestAddr vb,
                     uint32_t offset, uint32_t unk, uint32_t stride) {
  if (Recording(device)) {
    Record(CbOp::kSetStreamSource, {stream, vb, offset, unk, stride});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  if (stream < 16) g_streamStrideDwords[stream] = uint8_t(stride >> 2);
  renderer::SetStream(stream, vb, offset, stride);
}
void SetIndices(GuestAddr device, GuestAddr ib) {
  if (Recording(device)) {
    Record(CbOp::kSetIndices, {ib});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  renderer::SetIndices(ib);
}
void SetVertexDeclaration(GuestAddr device, GuestAddr decl) {
  if (Recording(device)) {
    Record(CbOp::kSetVertexDeclaration, {decl});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  g_currentVertexDecl = decl;
}

// --- shaders ---
uint32_t CreateVertexShader(GuestAddr) { return 0; }
uint32_t CreatePixelShader(GuestAddr) { return 0; }
void SetVertexShader(GuestAddr device, GuestAddr shader) {
  if (Recording(device)) {
    Record(CbOp::kSetVertexShader, {shader});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  g_currentVertexShader = shader;
  // Defaults are not applied while recording, the recorded bind reapplies
  // them at replay. Group 1 (shader-load defaults) applied here; groups 2/3
  // by the recompiled body (hook falls through).
  if (shader && !Recording(device)) {
    // VS object: 872-byte header (physical blob ptr at +32), container at +872.
    ApplyShaderFloatDefaults(device, false, shader + 872,
                             LoadGuestU32(shader + 32), g_replaying);
  }
}
void SetPixelShader(GuestAddr device, GuestAddr shader) {
  g_lastSetDevice = device;
  if (Recording(device)) {
    Record(CbOp::kSetPixelShader, {shader});
    // fall through: apply live too (the engine wrapper caches treat recorded
    // binds as current; skipping them here desyncs later "redundant" sets).
  }
  g_currentPixelShader = shader;
  if (shader && !Recording(device)) {
    // PS object: 40-byte header (physical blob ptr at +0x18), container at +40.
    ApplyShaderFloatDefaults(device, true, shader + 40,
                             LoadGuestU32(shader + 0x18), g_replaying);
  }
}
namespace {
std::vector<uint32_t> CopyConstPayload(GuestAddr data, uint32_t count) {
  std::vector<uint32_t> payload(size_t(count) * 4);
  for (uint32_t i = 0; i < count * 4; i++) {
    payload[i] = LoadGuestU32(data + i * 4);
  }
  return payload;
}
}  // namespace

void SetVertexShaderConstantF(GuestAddr device, uint32_t start_reg,
                              GuestAddr data, uint32_t count) {
  MarkFloatsDirty(device, false, start_reg, count);
  // While recording, capture the values (the guest pointer is a stack
  // temporary); the XDK records constant sets inline into the command
  // buffer. The bake/shadow passes set their per-pass matrices during
  // recording and never re-set them live.
  if (Recording(device)) {
    if (!count || count > 256) return;
    Record(CbOp::kSetVertexShaderConstantsF, {start_reg, count}, 0.0, false,
           CopyConstPayload(data, count));
    return;
  }
  renderer::SetShaderConstantsF(false, start_reg, data, count);
}
void SetPixelShaderConstantF(GuestAddr device, uint32_t start_reg,
                             GuestAddr data, uint32_t count) {
  MarkFloatsDirty(device, true, start_reg, count);
  if (Recording(device)) {
    if (!count || count > 256) return;
    Record(CbOp::kSetPixelShaderConstantsF, {start_reg, count}, 0.0, false,
           CopyConstPayload(data, count));
    return;
  }
  renderer::SetShaderConstantsF(true, start_reg, data, count);
}
void SetShaderGPRAllocation(GuestAddr, uint32_t, uint32_t, uint32_t) {
  /* no-op by design */
}

// --- render states (read back from device shadows at draw time) ---
void SetRenderState_ZEnable(GuestAddr, uint32_t) {}
void SetRenderState_ZWriteEnable(GuestAddr, uint32_t) {}
void SetRenderState_ZFunc(GuestAddr, uint32_t) {}
void SetRenderState_CullMode(GuestAddr, uint32_t) {}
void SetRenderState_AlphaTestEnable(GuestAddr, uint32_t) {}
void SetRenderState_AlphaRef(GuestAddr, uint32_t) {}
void SetRenderState_AlphaFunc(GuestAddr, uint32_t) {}
void SetBlendControl(GuestAddr, uint32_t, uint32_t) {}

// --- draws ---
void DrawVertices(GuestAddr device, uint32_t prim_type, uint32_t start,
                  uint32_t count) {
  if (Recording(device)) {
    Record(CbOp::kDrawVertices, {prim_type, start, count}, 0.0, true);
    ClearFloatDirty(device);  // the XDK recorded-draw flush clears dirty too
    return;
  }
  OnDraw(device, /*is_up=*/false);
  renderer::DrawVertices(device, prim_type, start, count);
  ClearFloatDirty(device);  // draw flush emitted + cleared the dirty ranges
  FoldBoolPendingIntoPersist(device);  // bool flush point
}
void DrawIndexedVertices(GuestAddr device, uint32_t prim_type,
                         uint32_t base_vertex, uint32_t start,
                         uint32_t count) {
  if (Recording(device)) {
    Record(CbOp::kDrawIndexedVertices, {prim_type, base_vertex, start, count},
           0.0, true);
    ClearFloatDirty(device);
    return;
  }
  OnDraw(device, /*is_up=*/false);
  renderer::DrawIndexedVertices(device, prim_type, base_vertex, start, count);
  ClearFloatDirty(device);
  FoldBoolPendingIntoPersist(device);
}
// UP draws feed stream 0's vfetch stride from the draw argument (the XDK
// binds the inline vertex data as a temporary stream 0), so mirror that in
// the patch inputs before resolving.
void DrawVerticesUP(GuestAddr device, uint32_t prim_type, uint32_t count,
                    GuestAddr vertex_data, uint32_t stride) {
  if (Recording(device)) {
    // XDK semantics: UP data is copied into the CB at record time; replay
    // re-reads of the guest scratch would draw rewritten data.
    Record(CbOp::kDrawVerticesUP, {prim_type, count, vertex_data, stride}, 0.0,
           true);
    {
      CbCall& call = g_commandBuffers[g_recordingCb].back();
      const uint32_t bytes = count * stride;
      const uint8_t* src =
          bytes && vertex_data
              ? rex::system::kernel_memory()->TranslateVirtual<const uint8_t*>(
                    vertex_data)
              : nullptr;
      if (src && bytes < (1u << 22)) {
        call.up_data.assign(src, src + bytes);
      }
    }
    ClearFloatDirty(device);
    return;
  }
  if (stride) g_streamStrideDwords[0] = uint8_t(stride >> 2);
  OnDraw(device, /*is_up=*/true);
  renderer::DrawVerticesUP(device, prim_type, count, vertex_data, stride);
  ClearFloatDirty(device);
  FoldBoolPendingIntoPersist(device);
}
void DrawIndexedVerticesUP(GuestAddr device, uint32_t prim_type,
                           uint32_t min_vertex, uint32_t num_vertices,
                           uint32_t index_count, GuestAddr index_data,
                           uint32_t index_format, GuestAddr vertex_data,
                           uint32_t stride) {
  if (Recording(device)) {
    Record(CbOp::kDrawIndexedVerticesUP,
           {prim_type, min_vertex, num_vertices, index_count, index_data,
            index_format, vertex_data, stride},
           0.0, true);
    {
      CbCall& call = g_commandBuffers[g_recordingCb].back();
      const uint32_t vbytes = num_vertices * stride;
      const uint32_t ibytes = index_count * 2;  // 16-bit indices only
      auto* mem = rex::system::kernel_memory();
      const uint8_t* vsrc =
          vbytes && vertex_data
              ? mem->TranslateVirtual<const uint8_t*>(vertex_data)
              : nullptr;
      const uint8_t* isrc =
          ibytes && index_data
              ? mem->TranslateVirtual<const uint8_t*>(index_data)
              : nullptr;
      if (vsrc && isrc && vbytes + ibytes < (1u << 22)) {
        call.up_data.reserve(vbytes + ibytes);
        call.up_data.assign(vsrc, vsrc + vbytes);
        call.up_data.insert(call.up_data.end(), isrc, isrc + ibytes);
        call.up_index_offset = vbytes;
      }
    }
    ClearFloatDirty(device);
    return;
  }
  if (stride) g_streamStrideDwords[0] = uint8_t(stride >> 2);
  OnDraw(device, /*is_up=*/true);
  renderer::DrawIndexedVerticesUP(device, prim_type, min_vertex, num_vertices,
                                  index_count, index_data, index_format,
                                  vertex_data, stride);
  ClearFloatDirty(device);
  FoldBoolPendingIntoPersist(device);
}

// --- command buffers ---
// Nested-recording stack. Avatar_Draw (0x829DFBF0) lazily records its avatar
// CB while the outer per-tile scene CB is being recorded (Begin/End on the
// same aux device, flag 0x10), so the inner End has to restore the outer
// session instead of closing recording altogether.
struct RecordingFrame {
  uint32_t cb;
  GuestAddr device;
  uint32_t float_baseline[2][256 * 4];
  uint32_t pending_dirty[2][8];
  uint32_t bool_baseline[40];
  uint64_t bool_pending;
  uint32_t fetch_baseline[32 * 6];
};
std::vector<std::unique_ptr<RecordingFrame>> g_recordingStack;

uint32_t BeginCommandBuffer(GuestAddr device, GuestAddr cb, uint32_t flags) {
  if (g_recordingCb) {
    auto frame = std::make_unique<RecordingFrame>();
    frame->cb = g_recordingCb;
    frame->device = g_recordingDevice;
    std::memcpy(frame->float_baseline, g_recordFloatBaseline,
                sizeof(g_recordFloatBaseline));
    std::memcpy(frame->pending_dirty, g_recordPendingDirty,
                sizeof(g_recordPendingDirty));
    std::memcpy(frame->bool_baseline, g_recordBoolBaseline,
                sizeof(g_recordBoolBaseline));
    frame->bool_pending = g_recordBoolPending;
    std::memcpy(frame->fetch_baseline, g_recordFetchBaseline,
                sizeof(g_recordFetchBaseline));
    g_recordingStack.push_back(std::move(frame));
  }
  g_recordingCb = cb;
  g_recordingDevice = device;
  if (device) g_lastRecordingDevice = device;
  g_commandBuffers[cb].clear();
  // Baseline for the recorded draws' dirty-float capture (see CbCall::data).
  if (device) SnapshotRecordFloatBaseline(device);
  REXGPU_INFO("videonative: BeginCommandBuffer cb={:#x} flags={:#x} depth={}",
              cb, flags, g_recordingStack.size());
  return 0;
}

uint32_t EndCommandBuffer(GuestAddr) {
  if (g_recordingCb) {
    REXGPU_INFO("videonative: EndCommandBuffer cb={:#x} ({} calls) depth={}",
                g_recordingCb, g_commandBuffers[g_recordingCb].size(),
                g_recordingStack.size());
  }
  if (!g_recordingStack.empty()) {
    auto frame = std::move(g_recordingStack.back());
    g_recordingStack.pop_back();
    g_recordingCb = frame->cb;
    g_recordingDevice = frame->device;
    std::memcpy(g_recordFloatBaseline, frame->float_baseline,
                sizeof(g_recordFloatBaseline));
    std::memcpy(g_recordPendingDirty, frame->pending_dirty,
                sizeof(g_recordPendingDirty));
    std::memcpy(g_recordBoolBaseline, frame->bool_baseline,
                sizeof(g_recordBoolBaseline));
    g_recordBoolPending = frame->bool_pending;
    std::memcpy(g_recordFetchBaseline, frame->fetch_baseline,
                sizeof(g_recordFetchBaseline));
  } else {
    g_recordingCb = 0;
    g_recordingDevice = 0;
  }
  return 0;  // S_OK
}

void SetCommandBufferPredication(GuestAddr device, uint32_t tile_pred,
                                 uint32_t run_pred) {
  if (Recording(device)) {
    Record(CbOp::kPredicationMark, {tile_pred, run_pred});
  }
}

// Direct RB_BLENDCONTROL[rt] PM4 emit (see video_native.h). During recording
// this becomes a predication-gated span (the avatar CB's whole blend model);
// live it changes the effective register file only, the device shadow and
// dirty bits are untouched, exactly like the real emitter.
void SetBlendControlDirect(GuestAddr device, uint32_t rt, uint32_t value) {
  if (Recording(device)) {
    Record(CbOp::kSetBlendControlDirect, {rt, value});
    return;
  }
  renderer::ApplyBlendControlDirect(rt, value);
}

void RunCommandBuffer(GuestAddr device, GuestAddr cb,
                      uint32_t predication_select) {
  if (Recording(device)) {
    Record(CbOp::kRunNested, {cb, predication_select});
    return;
  }
  auto it = g_commandBuffers.find(cb);
  if (it == g_commandBuffers.end()) return;
  const bool was_replaying = g_replaying;
  g_replaying = true;

  // XDK Run pre-flush: live register programming issued since the last flush
  // (the game sets the avatar/lighting blend on the main device between
  // Runs) reaches hardware before the CB's PM4 executes. Fold it into the
  // effective register file now so recorded direct-blend spans inside the CB
  // override it, not the other way around.
  renderer::ConsumeLiveStateDirty(device);
  // State isolation: the XDK dirties every non-persisted state group after a
  // Run and re-emits device shadows, replayed binds must not leak.
  renderer::PushState();
  // Inherit the device's live float constants at Run (XDK contract); recorded
  // in-CB constant sets overlay as encountered. Nested runs keep the outer
  // replay's accumulated state instead (the device shadow hasn't changed
  // mid-replay, and re-seeding would drop the outer CB's recorded overlays).
  if (!was_replaying) {
    renderer::SeedReplayFloatConstants(device, g_lastRecordingDevice);
    // Bool/loop constants inherit the run device's live shadow; recorded
    // dirty dwords overlay per draw. The Run pre-flush folds the run
    // device's pending bool dwords into the persistent file, so
    // replayed-CB state from earlier Runs survives.
    FoldBoolPendingIntoPersist(device);
  }
  const GuestAddr saved_vs = g_currentVertexShader;
  const GuestAddr saved_ps = g_currentPixelShader;
  const GuestAddr saved_decl = g_currentVertexDecl;
  uint8_t saved_strides[16];
  std::memcpy(saved_strides, g_streamStrideDwords, sizeof(saved_strides));

  // Xenos bin-predication semantics (IDA ground truth): a recorded
  // SetCommandBufferPredication(tile, run) emits SET_BIN_MASK_LO=tile /
  // SET_BIN_MASK_HI=run into the CB ((0,0) resets to LO=~0/HI=0); Run emits
  // SET_BIN_SELECT_HI=PredicationSelect, and SELECT_LO is the live
  // D3DDevice_SetPredication mask wrapped around this Run (the game's
  // section/tile culling, see SetPredication above). The CP executes a
  // span iff (select & mask) != 0 (64-bit bitwise and, not equality).
  const uint32_t select_lo = g_predicationSelectLo;
  uint32_t tile_tag = 0xFFFFFFFFu;  // reset state: mask LO = ~0
  uint32_t run_tag = 0;             // reset state: mask HI = 0
  for (const CbCall& c : it->second) {
    if (c.op == CbOp::kPredicationMark) {
      if (c.a[0] == 0 && c.a[1] == 0) {
        tile_tag = 0xFFFFFFFFu;
        run_tag = 0;
      } else {
        tile_tag = c.a[0];
        run_tag = c.a[1];
      }
      continue;
    }
    if ((tile_tag & select_lo) == 0 &&
        (run_tag & predication_select) == 0) {
      continue;
    }
    // Replayed draws use the render state captured at record time (the
    // recording device's shadow block), the live main-device shadows carry
    // unrelated (e.g. menu/UI) state. Same for bool/loop constants (set by
    // unhooked recompiled setters into the recording device's block).
    if (c.is_draw) {
      // Record-time dirty float constants first (they persist into g_state
      // for the rest of this replay, like the CB's PM4 stream on hardware).
      if (REXCVAR_GET(native_video_cb_const_diffs)) {
        for (size_t i = 0; i + 5 <= c.data.size(); i += 5) {
          renderer::SetShaderConstantsFHost((c.data[i] & 0x10000u) != 0,
                                            c.data[i] & 0xFFFFu,
                                            &c.data[i + 1], 1);
        }
      }
      renderer::SetReplayDrawState(c.st, c.st_dirty);
      // Recorded dirty bool/loop dwords persist into the replay base for
      // the rest of this Run (like the CB's PM4 register writes); untouched
      // dwords keep the live-seeded values.
      for (uint32_t d = 0; d < 40; d++) {
        if (c.bool_dirty & (1ull << d)) g_boolPersist[d] = c.bool_loop[d];
      }
      renderer::SetReplayBoolConstants(g_boolPersist);
      renderer::SetReplayFetchConstants(
          c.fetch_data.empty() ? nullptr : c.fetch_data.data(),
          uint32_t(c.fetch_data.size() / 7));
    }
    ReplayCall(device, c);
    if (c.is_draw) {
      renderer::SetReplayDrawState(nullptr, 0);
      renderer::SetReplayBoolConstants(nullptr);
      renderer::SetReplayFetchConstants(nullptr, 0);
    }
  }
  g_currentVertexShader = saved_vs;
  g_currentPixelShader = saved_ps;
  g_currentVertexDecl = saved_decl;
  std::memcpy(g_streamStrideDwords, saved_strides, sizeof(saved_strides));
  renderer::PopState();
  g_replaying = was_replaying;
}

namespace {
void ReplayCall(GuestAddr device, const CbCall& c) {
  switch (c.op) {
    case CbOp::kSetRenderTarget:
      SetRenderTarget(device, c.a[0], c.a[1]);
      break;
    case CbOp::kSetDepthStencilSurface:
      SetDepthStencilSurface(device, c.a[0], 0);
      break;
    case CbOp::kSetSurfaces:
      renderer::SetDepthSurface(c.a[0]);
      for (uint32_t i = 0; i < 4; i++) {
        renderer::SetRenderTargetSurface(i, c.a[1 + i]);
      }
      break;
    case CbOp::kClear: {
      const int32_t rect_vals[4] = {int32_t(c.a[3]), int32_t(c.a[4]),
                                    int32_t(c.a[5]), int32_t(c.a[6])};
      renderer::Clear(device, c.a[0], c.a[1], float(c.f), rect_vals, c.a[2]);
      break;
    }
    case CbOp::kSetViewport: {
      float min_z, max_z;
      std::memcpy(&min_z, &c.a[4], 4);
      std::memcpy(&max_z, &c.a[5], 4);
      renderer::SetViewport(c.a[0], c.a[1], c.a[2], c.a[3], min_z, max_z);
      break;
    }
    case CbOp::kSetScissorRect:
      renderer::SetScissor(int32_t(c.a[0]), int32_t(c.a[1]), int32_t(c.a[2]),
                           int32_t(c.a[3]));
      break;
    case CbOp::kSetTexture:
      SetTexture(device, c.a[0], c.a[1]);
      break;
    case CbOp::kSetStreamSource:
      SetStreamSource(device, c.a[0], c.a[1], c.a[2], c.a[3], c.a[4]);
      break;
    case CbOp::kSetIndices:
      SetIndices(device, c.a[0]);
      break;
    case CbOp::kSetVertexDeclaration:
      SetVertexDeclaration(device, c.a[0]);
      break;
    case CbOp::kSetVertexShader:
      SetVertexShader(device, c.a[0]);
      break;
    case CbOp::kSetPixelShader:
      SetPixelShader(device, c.a[0]);
      break;
    case CbOp::kSetVertexShaderConstantsF:
      renderer::SetShaderConstantsFHost(false, c.a[0], c.data.data(), c.a[1]);
      break;
    case CbOp::kSetPixelShaderConstantsF:
      renderer::SetShaderConstantsFHost(true, c.a[0], c.data.data(), c.a[1]);
      break;
    case CbOp::kResolve:
      ResolveParsed(c.a[0], c.a[1], c.a[2] != 0, int32_t(c.a[3]),
                    int32_t(c.a[4]), int32_t(c.a[5]), int32_t(c.a[6]),
                    int32_t(c.a[7]), int32_t(c.a[8]));
      break;
    case CbOp::kDrawVertices:
      DrawVertices(device, c.a[0], c.a[1], c.a[2]);
      break;
    case CbOp::kDrawIndexedVertices:
      DrawIndexedVertices(device, c.a[0], c.a[1], c.a[2], c.a[3]);
      break;
    case CbOp::kDrawVerticesUP:
      if (!c.up_data.empty()) {
        renderer::SetUPDataOverride(c.up_data.data(), nullptr);
      }
      DrawVerticesUP(device, c.a[0], c.a[1], c.a[2], c.a[3]);
      renderer::SetUPDataOverride(nullptr, nullptr);
      break;
    case CbOp::kDrawIndexedVerticesUP:
      if (!c.up_data.empty()) {
        renderer::SetUPDataOverride(c.up_data.data(),
                                    c.up_data.data() + c.up_index_offset);
      }
      DrawIndexedVerticesUP(device, c.a[0], c.a[1], c.a[2], c.a[3], c.a[4],
                            c.a[5], c.a[6], c.a[7]);
      renderer::SetUPDataOverride(nullptr, nullptr);
      break;
    case CbOp::kRunNested:
      // XDK semantics: state pending on the recording device is flushed
      // into the outer CB before a nested-run token, the avatar clusters'
      // blend reaches hardware this way (Avatar_Draw's CreateDrawCommands
      // deliberately re-raises the blend0 dirty bit so this flush bakes the
      // shadow around the token; IDA 0x829DFA48). Apply the captured
      // authoritative slots to the persistent file so the nested CB's draws
      // inherit them.
      renderer::ApplyReplayStatePersistent(c.st, c.st_dirty);
      RunCommandBuffer(device, c.a[0], c.a[1]);
      break;
    case CbOp::kPredicationMark:
      break;  // handled by the replay loop
    case CbOp::kSetBlendControlDirect:
      // Predication-gated (the replay loop already skipped this record if
      // the span's bin mask missed this Run's select). Persists in the
      // register file like the PM4 write it models; the following draws
      // (whose recorded st carries no authoritative blend) consume it.
      renderer::ApplyBlendControlDirect(c.a[0], c.a[1]);
      break;
  }
}
}  // namespace

}  // namespace rex::videonative
