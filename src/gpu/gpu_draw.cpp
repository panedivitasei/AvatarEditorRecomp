// Draw capture: strong overrides of the D3D draw entry points call through first, so the runtime has flushed its
// pending state and patched the vertex shader, then build an immutable DrawRecord from the device struct and args.
// Begin/EndVertices data is written by the game between the two calls, so that record is finished at End.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <rex/cvar.h>
#include <rex/graphics/format/ucode.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>
#include <rex/hash.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc/context.h>
#include <rex/system/kernel_state.h>

#include <span>

#include <xenosrecomp.h>

#include "gpu/gpu.h"
#include "gpu/gpu_burst_timing.h"
#include "gpu/gpu_cmdbuf.h"
#include "gpu/gpu_constant_snapshot.h"
#include "gpu/gpu_draw.h"
#include "gpu/gpu_memory.h"
#include "gpu/gpu_prepare.h"
#include "gpu/gpu_stream_snapshot.h"
#include "gpu/gpu_shader_pack.h"
#include "gpu/gpu_texture_decode.h"
#include "gpu/gpu_tracker.h"
#include "gpu/guest/d3d_device.h"

REXCVAR_DEFINE_BOOL(gpu_draws, true, "GPU", "Render captured draws; off leaves only clears, resolves and present");
REXCVAR_DEFINE_BOOL(gpu_parallel_prepare, true, "GPU", "Prepare captured draw state, samplers, constants and indices on workers");
REXCVAR_DEFINE_BOOL(gpu_stream_snapshot_pool, true, "GPU", "Reuse worker-prepared stream snapshot buffers");
REXCVAR_DEFINE_BOOL(gpu_replay_cull, true, "GPU", "Apply the recorded cull mode to command-buffer replays; off draws both faces");
REXCVAR_DECLARE(int32_t, gpu_trace_frame);
REXCVAR_DECLARE(int32_t, gpu_trace_frames);

extern "C" REX_FUNC(__imp__sub_92121380);
extern "C" REX_FUNC(__imp__sub_92121798);
extern "C" REX_FUNC(__imp__sub_92120D58);
extern "C" REX_FUNC(__imp__sub_921212E0);
extern "C" REX_FUNC(__imp__sub_92120898);
extern "C" REX_FUNC(__imp__sub_92120DA0);
extern "C" REX_FUNC(__imp__sub_92120888);
extern "C" REX_FUNC(sub_92136018);

namespace ae::gpu::draw {

namespace {

namespace xenos = rex::graphics::xenos;
namespace reg = rex::graphics::reg;
namespace ucode = rex::graphics::ucode;
using guest::D3DDevice;

template <typename T>
const T* Guest(uint32_t address) {
  return REX_KERNEL_MEMORY()->TranslateVirtual<const T*>(address);
}

uint32_t Load32(uint32_t address) { return *Guest<rex::be<uint32_t>>(address); }

bool Tracing() {
  const int32_t frame = REXCVAR_GET(gpu_trace_frame);
  const int32_t count = std::max<int32_t>(REXCVAR_GET(gpu_trace_frames), 1);
  return frame >= 0 && ae::gpu::FrameIndex() >= uint64_t(frame) && ae::gpu::FrameIndex() < uint64_t(frame) + count;
}

// One log line per distinct key, for pack defects and other per-object problems.
bool FirstTime(uint64_t key) {
  static std::mutex mutex;
  static std::unordered_set<uint64_t> seen;
  thread_local std::unordered_set<uint64_t> seen_here;  // per-draw callers skip the lock after the first hit
  if (!seen_here.insert(key).second) return false;
  std::lock_guard lock(mutex);
  return seen.insert(key).second;
}

void Once(std::atomic<bool>& flag, const char* what) {
  if (!flag.exchange(true)) REXGPU_WARN("[gpu] {}", what);
}

// ------------------------------------------------------------------------------------------------ shaders

struct VsInfo {
  uint64_t hash = 0;
  std::array<uint32_t, 32 * 4> vfetch{};  // contract 2.4 layout records, filled for fp2-keyed packs
  const ShaderEntry* entry = nullptr;
  std::vector<uint8_t> fetches;  // vertex fetch constants 0-95 the patched microcode reads
};

struct PsInfo {
  uint64_t hash = 0;
  const ShaderEntry* entry = nullptr;
};

struct DeviceState {
  FloatConstantSnapshots float_snapshots;
  uint32_t last_vs = 0;
  uint64_t vs_raw_hash = 0;
  uint32_t last_ps = 0;
  uint64_t ps_hash = 0;
};

std::mutex g_mutex;
std::unordered_map<uint32_t, DeviceState> g_devices;
std::unordered_map<uint64_t, VsInfo> g_vs_cache;
std::unordered_map<uint64_t, PsInfo> g_ps_cache;
uint32_t g_scratch = 0;
uint32_t g_scratch_size = 0;
uint32_t g_scratch_strides = 0;
uint64_t g_captured_bytes = 0;  // bytes copied out of guest memory since the last timer closed

// Adds the time spent in one capture and the bytes it copied to the frame statistics.
struct CaptureTimer {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  ~CaptureTimer() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    AddCaptureStats(uint64_t(ns.count()), std::exchange(g_captured_bytes, 0));
  }
};


// Every vfetch_full in the exec clauses, walked the way the translator walks control flow; mini fetches reuse the
// preceding full fetch's constant.
std::vector<uint8_t> VertexFetchConstants(const uint8_t* code, uint32_t size) {
  const uint32_t count = size / 4;
  std::vector<uint32_t> words(count);
  for (uint32_t i = 0; i < count; ++i) {
    words[i] = (uint32_t(code[i * 4]) << 24) | (uint32_t(code[i * 4 + 1]) << 16) | (uint32_t(code[i * 4 + 2]) << 8) |
               code[i * 4 + 3];
  }
  std::vector<uint8_t> out;
  uint32_t cf_end = count;
  for (uint32_t pass = 0; pass < 2; ++pass) {
    for (uint32_t addr = 0; addr + 3 <= cf_end; addr += 3) {
      ucode::ControlFlowInstruction cf[2];
      ucode::UnpackControlFlowInstructions(&words[addr], cf);
      for (const auto& c : cf) {
        uint32_t clause = 0, n = 0, sequence = 0;
        switch (c.opcode()) {
          case ucode::ControlFlowOpcode::kExec:
          case ucode::ControlFlowOpcode::kExecEnd:
            clause = c.exec.address(), n = c.exec.count(), sequence = c.exec.sequence();
            break;
          case ucode::ControlFlowOpcode::kCondExec:
          case ucode::ControlFlowOpcode::kCondExecEnd:
          case ucode::ControlFlowOpcode::kCondExecPredClean:
          case ucode::ControlFlowOpcode::kCondExecPredCleanEnd:
            clause = c.cond_exec.address(), n = c.cond_exec.count(), sequence = c.cond_exec.sequence();
            break;
          case ucode::ControlFlowOpcode::kCondExecPred:
          case ucode::ControlFlowOpcode::kCondExecPredEnd:
            clause = c.cond_exec_pred.address(), n = c.cond_exec_pred.count(), sequence = c.cond_exec_pred.sequence();
            break;
          default:
            continue;
        }
        if (pass == 0) {
          if (clause && clause * 3 < cf_end) cf_end = clause * 3;
          continue;
        }
        for (uint32_t i = 0; i < n; ++i, sequence >>= 2) {
          const uint32_t slot = (clause + i) * 3;
          if (slot + 3 > count || !(sequence & 1)) continue;
          ucode::VertexFetchInstruction fetch;
          std::memcpy(&fetch, &words[slot], sizeof(fetch));
          if (fetch.opcode() != ucode::FetchOpcode::kVertexFetch || fetch.is_mini_fetch()) continue;
          const uint8_t index = uint8_t(fetch.fetch_constant_index());
          if (std::find(out.begin(), out.end(), index) == out.end()) out.push_back(index);
        }
      }
    }
  }
  return out;
}

// A pack miss is a pack defect even when the runtime translator covers it (contract 5.4); logged once per key.
const ShaderEntry* HandleMiss(ShaderStage stage, uint64_t hash, uint64_t fp2, const uint8_t* code, uint32_t size,
                              uint32_t object, uint32_t variant) {
  const char* source = "failed";
  const auto started = std::chrono::steady_clock::now();
  const ShaderEntry* entry =
      TranslateMiss(stage, fp2, std::span<const uint32_t>(reinterpret_cast<const uint32_t*>(code), size / 4), &source);
  AddGuestTiming(0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                                 started).count()));
  const char* name = stage == ShaderStage::kPixel ? "ps" : "vs";
  if (FirstTime(hash ^ (stage == ShaderStage::kPixel ? 0x5053000000000000ull : 0))) {
    REXGPU_ERROR("[gpu] pack defect: {} fp2={:016X} rt={:016X} variant={} obj={:#010x} source={}", name, fp2, hash,
                 variant, object, source);
  }
  return entry;
}

// The bound vertex shader object's selected variant: its address, flag, code address and size; false when unusable.
bool LocateVs(const D3DDevice* device, uint32_t& vs, bool& variant1, uint32_t& code, uint32_t& size) {
  vs = device->vertex_shader;
  if (!vs) return false;
  variant1 = (Load32(vs + guest::kVsHeader) & guest::kVsVariantFlag) != 0;
  const uint32_t desc = Load32(vs + guest::kVsDescriptor + (variant1 ? 8 : 0));
  if (desc >= 0x10000) return false;
  code = Load32(vs + guest::kVsCodeBase) + Load32(vs + desc + guest::kVsCodeOffset);
  size = Load32(vs + desc + guest::kVsCodeOffset + 4);
  return code && size && size <= 0x100000;
}

// The runtime's own vfetch patch (PatchVertexShaderToMatchVertexDeclaration 0x92136018), run on a scratch copy so the
// key is the patched microcode the pack was built from; `snap` replaces the object's current code with the bytes a
// command buffer recorded. Strides come from the device, with stream 0 overridden for inline draws as BeginVertices
// does before its flush.
const VsInfo* ResolveVs(PPCContext& ctx, uint8_t* base, DeviceState& ds, const D3DDevice* device, int up_stride_dwords,
                        const VsSnapshot* snap) {
  BurstTiming timing(7);
  uint32_t vs, code, size;
  bool variant1;
  const uint8_t* raw = nullptr;
  auto* mem = REX_KERNEL_MEMORY();
  if (snap) {
    if (snap->code.empty()) return nullptr;
    vs = snap->object;
    variant1 = snap->variant1;
    code = 0;
    size = uint32_t(snap->code.size());
    raw = snap->code.data();
    ds.vs_raw_hash = XXH3_64bits(raw, size);
    ds.last_vs = 0;
  } else {
    if (!LocateVs(device, vs, variant1, code, size)) return nullptr;
    raw = mem->TranslateVirtual<const uint8_t*>(code);
    // The game link-patches microcode in place after load, so the raw bytes are hashed again whenever the bound
    // object changes; paranoid mode hashes them on every draw.
    if (ds.last_vs != vs || memory::Paranoid()) {
      ds.vs_raw_hash = XXH3_64bits(raw, size);
      ds.last_vs = vs;
    }
  }
  uint8_t strides[16];
  std::memcpy(strides, device->stream_stride_dwords, 16);
  if (up_stride_dwords >= 0) strides[0] = uint8_t(up_stride_dwords);
  const uint32_t decl = device->vertex_declaration;
  if (Tracing()) {
    REXGPU_INFO("[gpu]   vs object {:#010x} variant {} code {:#010x}+{:#x} decl {:#010x} strides {} {} {} {}{}", vs,
                variant1 ? 1 : 0, code, size, decl, strides[0], strides[1], strides[2], strides[3],
                snap ? " (recorded)" : "");
  }
  const uint64_t key = ds.vs_raw_hash ^ XXH3_64bits(strides, 16) * 31 ^ (uint64_t(vs) << 32 | decl) * 17 ^
                       (variant1 ? 0x9E3779B97F4A7C15ull : 0);
  auto it = g_vs_cache.find(key);
  if (it != g_vs_cache.end()) return &it->second;

  if (g_scratch_size < size) {
    const uint32_t alloc = (size + 0xFFF) & ~0xFFFu;
    g_scratch = mem->SystemHeapAlloc(alloc, 0x100);
    g_scratch_size = g_scratch ? alloc : 0;
  }
  if (!g_scratch_strides) g_scratch_strides = mem->SystemHeapAlloc(16, 0x10);
  if (!g_scratch || !g_scratch_strides) return nullptr;
  std::memcpy(mem->TranslateVirtual<void*>(g_scratch), raw, size);
  std::memcpy(mem->TranslateVirtual<void*>(g_scratch_strides), strides, 16);
  if (decl) {
    rex::CallFrame frame(ctx);
    frame.ctx.r1.u64 = ctx.r1.u64 - 0x200;
    frame.ctx.r3.u64 = vs;
    frame.ctx.r4.u64 = g_scratch;
    frame.ctx.r5.u64 = decl;
    frame.ctx.r6.u64 = g_scratch_strides;
    frame.ctx.r7.u64 = variant1 ? 1 : 0;
    sub_92136018(frame.ctx, base);
  }
  const auto* patched = mem->TranslateVirtual<const uint8_t*>(g_scratch);
  VsInfo info;
  info.hash = XXH3_64bits(patched, size);
  uint64_t fp2 = 0;
  if (ShaderPackUsesFp2()) {
    const std::span<const uint32_t> words(reinterpret_cast<const uint32_t*>(patched), size / 4);
    fp2 = xenosrecomp::Fingerprint(words, true);
    std::array<xenosrecomp::XeVfetchRecord, 32> records{};
    const uint32_t n = std::min<uint32_t>(xenosrecomp::ReadVfetchRecords(words, true, records), 32);
    std::memcpy(info.vfetch.data(), records.data(), n * sizeof(records[0]));
  }
  info.entry = FindShader(ShaderStage::kVertex, info.hash, fp2);
  info.fetches = VertexFetchConstants(patched, size);
  if (!info.entry) info.entry = HandleMiss(ShaderStage::kVertex, info.hash, fp2, patched, size, vs, variant1 ? 1 : 0);
  return &(g_vs_cache[key] = std::move(info));
}

const PsInfo* ResolvePs(DeviceState& ds, const D3DDevice* device) {
  BurstTiming timing(7);
  const uint32_t ps = device->pixel_shader;
  if (!ps) return nullptr;
  const uint32_t header = ps + guest::kPsHeader;
  const uint32_t shader = Load32(header + guest::kPsShaderOffset);
  if (!shader || shader > 0x8000) return nullptr;
  const uint32_t code = Load32(ps + guest::kPsCodeBase) + Load32(header + shader);
  const uint32_t size = Load32(header + shader + 4);
  if (!code || !size || size > 0x100000) return nullptr;
  if (ds.last_ps != ps || memory::Paranoid()) {
    ds.ps_hash = XXH3_64bits(REX_KERNEL_MEMORY()->TranslateVirtual<const void*>(code), size);
    ds.last_ps = ps;
  }
  auto it = g_ps_cache.find(ds.ps_hash);
  if (it != g_ps_cache.end()) return &it->second;
  PsInfo info;
  info.hash = ds.ps_hash;
  uint64_t fp2 = 0;
  if (ShaderPackUsesFp2()) {
    fp2 = xenosrecomp::Fingerprint(
        std::span<const uint32_t>(REX_KERNEL_MEMORY()->TranslateVirtual<const uint32_t*>(code), size / 4), true);
  }
  info.entry = FindShader(ShaderStage::kPixel, info.hash, fp2);
  if (!info.entry) {
    info.entry = HandleMiss(ShaderStage::kPixel, info.hash, fp2,
                            REX_KERNEL_MEMORY()->TranslateVirtual<const uint8_t*>(code), size, ps, 0);
  }
  return &(g_ps_cache[info.hash] = info);
}

// Literal (def) constants: the runtime's SetLiteralShaderConstants has the GPU load them straight from the shader's
// memory into the ALU constant file, so they never reach m_Constants. The same table is applied to the copy here.
struct LiteralConstant {
  uint32_t dword, bits;
};

void CaptureLiterals(uint32_t header, uint32_t code_base, std::vector<LiteralConstant>& literals) {
  const uint32_t table_offset = Load32(header + guest::kUcodeLiteralOffset);
  if (!table_offset) return;
  const uint32_t table = header + table_offset;
  const uint32_t end = table + 20 + Load32(table + 16);
  for (uint32_t at = table + 20; at + 8 <= end; at += 8) {
    const uint16_t reg = *Guest<rex::be<uint16_t>>(at);
    const uint16_t dwords = *Guest<rex::be<uint16_t>>(at + 2);
    if (!dwords) break;
    const auto* src = Guest<rex::be<uint32_t>>(code_base + Load32(at + 4));
    for (uint32_t i = 0; i < dwords; ++i) {
      const uint32_t dword = reg * 4 + i;
      if (dword >= 512 * 4) break;
      literals.push_back({dword, uint32_t(src[i])});
    }
  }
}

// ------------------------------------------------------------------------------------------------ state

uint8_t SamplerIndex(const xenos::xe_gpu_texture_fetch_t& f, const PackBinding& b) {
  auto clamp = f.clamp_x;
  bool linear = (b.mag != 0xFF ? b.mag : uint8_t(f.mag_filter)) == uint8_t(xenos::TextureFilter::kLinear) ||
                (b.min != 0xFF ? b.min : uint8_t(f.min_filter)) == uint8_t(xenos::TextureFilter::kLinear);
  const bool mip_linear = (b.mip != 0xFF ? b.mip : uint8_t(f.mip_filter)) == uint8_t(xenos::TextureFilter::kLinear);
  const uint8_t aniso = b.aniso != 0xFF ? b.aniso : uint8_t(f.aniso_filter);
  if (xenos::ClampModeUsesBorder(clamp)) return uint8_t(14 + (linear ? 1 : 0));
  uint8_t address = 2;
  if (clamp == xenos::ClampMode::kRepeat) address = 0;
  else if (clamp == xenos::ClampMode::kMirroredRepeat || clamp == xenos::ClampMode::kMirrorClampToEdge ||
           clamp == xenos::ClampMode::kMirrorClampToHalfway) address = 1;
  if (aniso != uint8_t(xenos::AnisoFilter::kDisabled) && aniso != uint8_t(xenos::AnisoFilter::kUseFetchConst)) {
    return uint8_t(12 + (address == 0 ? 0 : 1));
  }
  return uint8_t(address * 4 + (linear ? 2 : 0) + (mip_linear ? 1 : 0));
}

// A translator override, or the fetch constant's value when the override is absent or says "use the fetch constant".
uint8_t Pick(uint8_t override_value, uint32_t fetch_value, uint8_t use_fetch) {
  return override_value == 0xFF || override_value == use_fetch ? uint8_t(fetch_value) : override_value;
}

// The full sampler state in the SamplerBinding::state packing; anisotropy forces linear filtering, as the
// translator's normalization does.
uint32_t SamplerState(const xenos::xe_gpu_texture_fetch_t& f, const PackBinding& b) {
  const uint8_t use_fetch = uint8_t(xenos::TextureFilter::kUseFetchConst);
  uint32_t mag = Pick(b.mag, uint32_t(f.mag_filter), use_fetch);
  uint32_t min = Pick(b.min, uint32_t(f.min_filter), use_fetch);
  uint32_t mip = Pick(b.mip, uint32_t(f.mip_filter), use_fetch);
  uint32_t aniso = Pick(b.aniso, uint32_t(f.aniso_filter), uint8_t(xenos::AnisoFilter::kUseFetchConst));
  if (aniso == uint32_t(xenos::AnisoFilter::kUseFetchConst)) aniso = 0;
  aniso = std::min<uint32_t>(aniso, 5);
  if (aniso > 1) mag = min = mip = uint32_t(xenos::TextureFilter::kLinear);
  else aniso = 0;
  const uint32_t mip_mode = mip == uint32_t(xenos::TextureFilter::kBaseMap)  ? 2u
                            : mip == uint32_t(xenos::TextureFilter::kLinear) ? 1u
                                                                             : 0u;
  return (f.dword_0 >> 10 & 0x1FF) | (mag == 1 ? 1u << 9 : 0) | (min == 1 ? 1u << 10 : 0) | mip_mode << 11 |
         aniso << 13 | uint32_t(f.border_color) << 16 | uint32_t(f.mip_min_level) << 18 |
         uint32_t(f.mip_max_level) << 22;
}

bool BlendIsCopy(uint32_t blend) {
  reg::RB_BLENDCONTROL b;
  b.value = blend;
  return b.color_srcblend == xenos::BlendFactor::kOne && b.color_destblend == xenos::BlendFactor::kZero &&
         b.color_comb_fcn == xenos::BlendOp::kAdd && b.alpha_srcblend == xenos::BlendFactor::kOne &&
         b.alpha_destblend == xenos::BlendFactor::kZero && b.alpha_comb_fcn == xenos::BlendOp::kAdd;
}

struct TextureCacheEntry {
  ResourceDesc desc;
  memory::QueryCache range_cache;
  GuestTextureDesc guest;     // layout from the fetch constant, computed once per key
  uint32_t version = 0;       // host version sent last
  uint32_t generation = 0;    // tracker generation at that upload
  uint64_t frame = UINT64_MAX;  // frame of the last content check
  uint64_t sample = 0;        // stripe hash of the guest bytes at that upload
  uint64_t checked = UINT64_MAX;  // frame of the last stripe check, one per texture per frame
  uint64_t uploaded = UINT64_MAX;  // frame of the last upload
  // Host textures this entry rotates through when its memory is rewritten more than once in a frame, with the host
  // version each one holds; slot 0 is desc.id itself.
  static constexpr int kRing = 4;
  uint32_t ring_id[kRing] = {};
  uint32_t ring_version[kRing] = {};
  int ring_at = 0;
};
std::unordered_map<uint64_t, TextureCacheEntry> g_textures;
std::unordered_map<uint32_t, uint64_t> g_texture_ids;  // host id -> g_textures key, for eviction notices

uint32_t NewTextureId(uint64_t key) {
  const uint32_t id = tracker::NewId();
  g_texture_ids[id] = key;
  return id;
}

// The host destroyed this sampled texture: whichever ring slot held it starts over at version 0.
void ForgetHostTexture(uint32_t id) {
  const auto it = g_texture_ids.find(id);
  if (it == g_texture_ids.end()) return;
  const auto entry_it = g_textures.find(it->second);
  if (entry_it == g_textures.end()) return;
  TextureCacheEntry& entry = entry_it->second;
  if (entry.desc.id == id) entry.version = 0;
  for (int i = 0; i < TextureCacheEntry::kRing; ++i) {
    if (entry.ring_id[i] == id) entry.ring_version[i] = 0;
  }
}
TextureBinding FinishBinding(TextureBinding out, uint64_t key, TextureCacheEntry& entry);

// What a fetch constant slot resolved to last time: the same six words with the same pack overrides bind the same
// host texture and swizzle, so the census, resolve lookup and key hash run once per distinct constant.
struct BindingSlotCache {
  bool valid = false;
  bool resolved = false;
  uint32_t words[6] = {};
  uint64_t overrides = 0;
  uint32_t resolve_epoch = 0;
  ResourceDesc texture;
  std::array<uint8_t, 4> swizzle{};
  uint64_t key = 0;
  TextureCacheEntry* entry = nullptr;
};
thread_local BindingSlotCache g_binding_cache[32];

uint64_t BindingOverrides(const PackBinding& b) {
  return uint64_t(b.dimension) | uint64_t(b.mag) << 8 | uint64_t(b.min) << 16 | uint64_t(b.mip) << 24 |
         uint64_t(b.aniso) << 32;
}

// Draw constants come from 64-record chunks; a record's shared_ptr aliases its chunk, so a frame of draws costs a
// handful of allocations instead of one 12 KB block per draw.
std::shared_ptr<DrawConstants> AllocateConstants() {
  constexpr size_t kChunk = 64;
  struct Chunk {
    std::array<DrawConstants, kChunk> items;
  };
  thread_local std::vector<std::shared_ptr<Chunk>> pool;
  thread_local size_t cursor = 0;
  thread_local std::shared_ptr<Chunk> chunk;
  thread_local size_t next = kChunk;
  if (next == kChunk) {
    chunk.reset();
    // The pool can reuse a chunk only after every aliased draw reference has been released.
    for (size_t checked = 0; checked < pool.size(); ++checked) {
      cursor = (cursor + 1) % pool.size();
      if (pool[cursor].use_count() == 1) {
        chunk = pool[cursor];
        break;
      }
    }
    if (!chunk) {
      chunk = std::make_shared<Chunk>();
      if (pool.size() < 128) pool.push_back(chunk);
    }
    next = 0;
  }
  DrawConstants* item = &chunk->items[next++];
  // Build overwrites every array in full; only the system block keeps defaults for fields a draw may not set.
  item->system = SystemConstants{};
  return std::shared_ptr<DrawConstants>(chunk, item);
}

PreparationQueue* g_prepare_queue = nullptr;

PreparationQueue& PrepareQueue() {
  static PreparationQueue queue;
  g_prepare_queue = &queue;
  return queue;
}

void ConvertConstants(DrawConstants& c) {
  for (auto* values : {&c.vs_float, &c.ps_float}) {
    for (float& value : *values) {
      uint32_t word;
      std::memcpy(&word, &value, sizeof(word));
      word = __builtin_bswap32(word);
      std::memcpy(&value, &word, sizeof(word));
    }
  }
}

void ApplyIndices(DrawRecord& r, const PreparedIndices& prepared) {
  r.count = prepared.count;
  if (prepared.topology) r.state.topology = prepared.topology;
  r.index.size = uint32_t(prepared.bytes->size());
  if (r.send_prepared_indices) {
    r.index.bytes = prepared.bytes;
    g_captured_bytes += prepared.bytes->size();
  }
}

void FinishIndices(DrawRecord& r) {
  if (!r.prepared_indices.valid()) return;
  ApplyIndices(r, r.prepared_indices.get());
  r.prepared_indices = {};
}

// Hashes 32 stripes of 128 bytes spread over the texture's base and mip ranges: the per-frame change check for
// every texture, since level textures have no lifetime hook and owned ones can be written without Lock.
uint64_t SampleTexture(const GuestTextureDesc& desc) {
  uint8_t stripes[32 * 128];
  size_t n = 0;
  auto take = [&](uint32_t address, uint32_t size, uint32_t count) {
    if (!address || !size) return;
    const auto* bytes = REX_KERNEL_MEMORY()->TranslatePhysical<const uint8_t*>(address);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t at = uint32_t((uint64_t(size) * i) / count) & ~15u;
      const uint32_t len = std::min<uint32_t>(128, size - at);
      std::memcpy(stripes + n, bytes + at, len);
      n += len;
    }
  };
  take(desc.base_address, desc.base_size, desc.mip_size ? 24 : 32);
  take(desc.mip_address, desc.mip_size, desc.mip_size ? 8 : 0);
  return XXH3_64bits(stripes, n) ^ desc.base_size;
}

// Everything that changes the decoded contents: pitch, tiling and signs, format and address, size, mip range and
// dimension.
uint64_t TextureKey(const xenos::xe_gpu_texture_fetch_t& f) {
  BurstTiming timing(5);
  const uint32_t words[6] = {f.dword_0 & 0xFFC003FCu, f.dword_1, f.dword_2, 0, f.dword_4 & 0x3FCu,
                             f.dword_5 & 0xFFFFFE00u};
  return XXH3_64bits(words, sizeof(words));
}

std::atomic<bool> g_warn_resolved_sign{false};

// One info line per distinct texture shape the draws sample.
void CensusTexture(const xenos::xe_gpu_texture_fetch_t& f, uint8_t binding_dimension, bool resolved) {
  const uint32_t signs = (f.dword_0 >> 2) & 0xFF;
  const uint64_t key = 0x7465780000000000ull | (uint64_t(f.format) << 32) | (uint64_t(f.dimension) << 28) |
                       (uint64_t(f.stacked) << 27) | (uint64_t(signs) << 16) | (uint64_t(f.packed_mips) << 15) |
                       (uint64_t(f.mip_max_level != 0) << 14) | (uint64_t(f.tiled) << 13) |
                       (uint64_t(binding_dimension) << 8) | (resolved ? 1 : 0);
  if (!FirstTime(key)) return;
  REXGPU_INFO("[gpu] texture census: format {} dimension {} stacked {} signs {}{}{}{} mips {}-{} packed {} tiled {} "
              "endian {} binding {} source {}",
              uint32_t(f.format), uint32_t(f.dimension), uint32_t(f.stacked), signs & 3, (signs >> 2) & 3,
              (signs >> 4) & 3, (signs >> 6) & 3, uint32_t(f.mip_min_level), uint32_t(f.mip_max_level),
              uint32_t(f.packed_mips), uint32_t(f.tiled), uint32_t(f.endianness), binding_dimension,
              resolved ? "resolve" : "memory");
}

TextureBinding BindTexture(const D3DDevice* device, const PackBinding& b, uint8_t slot) {
  BurstTiming timing(9);
  TextureBinding out;
  out.slot = slot;
  out.dimension = b.dimension ? b.dimension : 1;
  xenos::xe_gpu_texture_fetch_t f;
  const auto group = device->m_Constants.Fetch[b.fetch_constant & 31].get();
  std::memcpy(&f, &group, sizeof(f));
  if (f.type != xenos::FetchConstantType::kTexture) return out;
  BindingSlotCache& cache = g_binding_cache[b.fetch_constant & 31];
  const uint64_t overrides = BindingOverrides(b);
  const uint32_t epoch = tracker::ResolveEpoch();
  if (cache.valid && cache.overrides == overrides && cache.resolve_epoch == epoch &&
      std::memcmp(cache.words, &f, sizeof(cache.words)) == 0) {
    out.texture = cache.texture;
    out.swizzle = cache.swizzle;
    if (cache.resolved || cache.entry->desc.format == HostFormat::kUnsupported) return out;
    return FinishBinding(out, cache.key, *cache.entry);
  }
  cache.valid = false;
  for (uint32_t i = 0; i < 4; ++i) out.swizzle[i] = uint8_t((f.swizzle >> (3 * i)) & 7);
  const uint32_t base_address = (uint32_t(f.base_address) & 0x1FFFF) << 12;
  const ResourceDesc resolved = f.dimension == xenos::DataDimension::k2DOrStacked && !f.stacked
      ? tracker::FindResolved(base_address, f.size_2d.width + 1, f.size_2d.height + 1,
                              uint32_t(f.format))
      : ResourceDesc{};
  CensusTexture(f, out.dimension, resolved.id != 0);
  // A resolve destination binds its host render texture by exact address. The host copy keeps the channels the
  // shader wrote, while memory has them in the texture format's order: D3D's resolve swaps red and blue for formats
  // whose swizzle reads red from Z, and the fetch swizzle swaps them back, so the two are composed here.
  if (resolved.id) {
    out.texture = resolved;
    if (out.swizzle[0] == 2) {
      for (auto& c : out.swizzle) c = c == 0 ? 2 : c == 2 ? 0 : c;
    }
    out.swizzle = ComposeSwizzle(resolved.format, out.swizzle);
    // The host copy holds what the shader wrote, so a sign conversion on the fetch has nothing to undo.
    if ((f.dword_0 >> 2) & 0xFF) {
      Once(g_warn_resolved_sign, "a signed, biased or gamma fetch of a resolve destination samples it unconverted");
    }
    cache = BindingSlotCache{};
    cache.valid = true;
    cache.resolved = true;
    std::memcpy(cache.words, &f, sizeof(cache.words));
    cache.overrides = overrides;
    cache.resolve_epoch = epoch;
    cache.texture = out.texture;
    cache.swizzle = out.swizzle;
    return out;
  }
  const uint64_t key = TextureKey(f);
  auto& entry = g_textures[key];
  GuestTextureDesc& desc = entry.guest;
  if (!entry.desc.id) {
    if (!DescribeTexture(f, desc)) {
      entry.desc.id = NewTextureId(key);
      entry.desc.kind = ResourceKind::kSampledTexture;
      entry.desc.format = HostFormat::kUnsupported;
      entry.desc.width = entry.desc.height = 1;
      out.texture = entry.desc;
      return out;
    }
    entry.desc.id = NewTextureId(key);
    entry.desc.kind = ResourceKind::kSampledTexture;
    entry.desc.width = desc.width;
    entry.desc.height = desc.height;
    entry.desc.format = desc.host_format;
    entry.desc.shape = desc.shape;
    entry.desc.depth = desc.shape == TextureShape::k3D ? desc.depth : desc.shape == TextureShape::kStacked ? desc.layers
                                                                                                          : 1;
    entry.desc.mip_levels = desc.mip_levels;
    if (Tracing()) {
      REXGPU_INFO("[gpu] texture {:#010x} {}x{} guest format {} -> host {} {}", base_address, desc.width, desc.height,
                  uint32_t(f.format), entry.desc.id, HostFormatName(desc.host_format));
    }
  }
  out.texture = entry.desc;
  out.swizzle = ComposeSwizzle(entry.desc.format, out.swizzle);
  cache = BindingSlotCache{};
  cache.valid = true;
  std::memcpy(cache.words, &f, sizeof(cache.words));
  cache.overrides = overrides;
  cache.resolve_epoch = epoch;
  cache.texture = out.texture;
  cache.swizzle = out.swizzle;
  cache.key = key;
  cache.entry = &entry;
  if (entry.desc.format == HostFormat::kUnsupported) return out;
  return FinishBinding(out, key, entry);
}

// The per-draw part of a texture binding: whether the host's copy is current, and the decode when it is not.
TextureBinding FinishBinding(TextureBinding out, uint64_t key, TextureCacheEntry& entry) {
  GuestTextureDesc& desc = entry.guest;
  const auto state = memory::Query(desc.base_address, std::max(desc.base_size, 1u), entry.range_cache);
  const uint64_t frame = FrameIndex();
  uint64_t sample = 0;
  bool upload = !entry.version;
  if (!upload) {
    if (state.registered) upload = state.generation != entry.generation;
    // Owned textures are checked too: a streamer that writes texture memory without Lock/Unlock leaves the
    // generation untouched, and the stale copy shows as a blurred or wrong texture after minutes of play.
    if (!upload && entry.checked != frame) {
      entry.checked = frame;
      sample = SampleTexture(desc);
      upload = sample != entry.sample;
    }
  }
  const bool paranoid = memory::Paranoid();
  const auto* bytes = paranoid ? REX_KERNEL_MEMORY()->TranslatePhysical<const uint8_t*>(desc.base_address) : nullptr;
  if (!upload && paranoid) upload = !memory::CheckUnchanged(key, desc.base_address, desc.base_size, bytes);
  if (!state.registered) memory::NoteUnregistered(desc.base_address, desc.base_size, "texture");
  if (upload && entry.uploaded == frame) {
    // The host applies a frame's uploads before its draws, so a second rewrite of the same memory within one frame
    // (XUI decodes one icon after another into a scratch texture and bakes each) moves to the next host texture of
    // the ring; a frame that rewrites more often than the ring is deep wraps onto the oldest copy.
    entry.ring_id[entry.ring_at] = entry.desc.id;
    entry.ring_version[entry.ring_at] = entry.version;
    entry.ring_at = (entry.ring_at + 1) % TextureCacheEntry::kRing;
    if (!entry.ring_id[entry.ring_at]) entry.ring_id[entry.ring_at] = NewTextureId(key);
    entry.desc.id = entry.ring_id[entry.ring_at];
    entry.version = entry.ring_version[entry.ring_at];
    out.texture = entry.desc;
  }
  if (upload) {
    entry.uploaded = frame;
    entry.sample = sample ? sample : SampleTexture(desc);
    entry.checked = frame;
    const auto started = std::chrono::steady_clock::now();
    out.upload = DecodeTextureAsync(desc);
    AddGuestTiming(1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                                   started).count()));
    if (out.upload) g_captured_bytes += uint64_t(desc.base_size) + desc.mip_size;
    entry.version++;
    entry.generation = state.generation;
    if (paranoid) memory::NoteUploaded(key, desc.base_address, desc.base_size, bytes);
  }
  entry.frame = frame;
  out.version = entry.version;
  out.texture = entry.desc;  // the slot cache's copy predates any id change
  return out;
}

// Vertex ranges the host holds: registered ones persist across frames at the tracker generation; unregistered ones
// are uploaded once per frame and shared by that frame's draws.
struct SentRange {
  uint32_t version = 0;
  uint64_t frame = UINT64_MAX;
  memory::QueryCache range_cache;
};
std::unordered_map<uint64_t, SentRange> g_sent_streams;
struct StreamSlot {
  uint64_t key = 0;
  SentRange* sent = nullptr;
};
std::array<StreamSlot, 96> g_stream_slots;
void ForgetIndex(uint64_t key);

// Drops the host's evicted keys from both sent tables so the next draw over them carries bytes again.
void ForgetEvicted() {
  if (!AnyEvicted()) return;
  std::vector<uint64_t> keys;
  TakeEvicted(keys);
  g_stream_slots.fill({});
  for (uint64_t key : keys) {
    if (key & kEvictedTextureTag) {
      const uint32_t id = uint32_t(key & 0xFFFFFFFFu);
      ForgetHostTexture(id);
      tracker::ForgetResolved(id);
      continue;
    }
    g_sent_streams.erase(key);
    ForgetIndex(key);
  }
}

SentRange& StreamRange(uint32_t fetch_index, uint64_t key) {
  if (fetch_index >= g_stream_slots.size()) return g_sent_streams[key];
  auto& slot = g_stream_slots[fetch_index];
  if (!slot.sent || slot.key != key) slot = {key, &g_sent_streams[key]};
  return *slot.sent;
}

StreamData CaptureStream(uint32_t fetch_index, uint32_t guest_base, uint32_t size) {
  BurstTiming timing(3);
  StreamData s;
  s.fetch_index = fetch_index;
  s.guest_base = guest_base;
  s.size = size;
  const uint64_t frame = FrameIndex();
  s.key = (uint64_t(guest_base) << 32) | size;
  ForgetEvicted();
  SentRange& sent = StreamRange(fetch_index, s.key);
  const auto state = [&] {
    BurstTiming query_timing(12);
    return memory::Query(guest_base, size, sent.range_cache);
  }();
  s.persistent = state.registered;
  s.version = state.generation;
  const bool paranoid = memory::Paranoid();
  const uint8_t* bytes = paranoid ? REX_KERNEL_MEMORY()->TranslatePhysical<const uint8_t*>(guest_base) : nullptr;
  bool send;
  if (state.registered) {
    send = sent.frame == UINT64_MAX || sent.version != state.generation;
    if (!send && paranoid && !memory::CheckUnchanged(s.key, guest_base, size, bytes)) {
      send = true;
      s.version = state.generation + 0x80000000u;
    }
  } else {
    BurstTiming unowned_timing(13);
    memory::NoteUnregistered(guest_base, size, "vertex range");
    s.version = uint32_t(frame);
    send = sent.frame != frame;
  }
  if (send) {
    if (!bytes) bytes = REX_KERNEL_MEMORY()->TranslatePhysical<const uint8_t*>(guest_base);
    {
      BurstTiming copy_timing(11);
      if (REXCVAR_GET(gpu_stream_snapshot_pool) && REXCVAR_GET(gpu_parallel_prepare)) {
        static StreamSnapshotPool snapshots;
        s.bytes = snapshots.Copy(bytes, size);
      } else {
        s.bytes = std::make_shared<const std::vector<uint8_t>>(bytes, bytes + size);
      }
    }
    if (BurstTimingEnabled()) AddGuestTiming(14, size);
    g_captured_bytes += size;
    sent.version = s.version;
    sent.frame = frame;
    if (paranoid) memory::NoteUploaded(s.key, guest_base, size, bytes);
  }
  return s;
}

// Reads indices with the buffer's Xenos endian swap applied, the same swap the shaders apply to a raw index.
std::vector<uint32_t> ReadIndices(const uint8_t* src, uint32_t count, bool index32, uint32_t endian) {
  std::vector<uint32_t> out(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t v;
    if (index32) {
      std::memcpy(&v, src + i * 4, 4);
      if (endian == 1 || endian == 2) v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
      if (endian == 2 || endian == 3) v = (v << 16) | (v >> 16);
    } else {
      v = uint32_t(src[i * 2]) | (uint32_t(src[i * 2 + 1]) << 8);
      if (endian == 1 || endian == 2) v = ((v & 0xFF) << 8) | (v >> 8);
    }
    out[i] = v;
  }
  return out;
}

// Fans and quad lists have no host topology and become triangle lists; everything else draws as is.
bool NeedsExpansion(xenos::PrimitiveType type) {
  return type == xenos::PrimitiveType::kTriangleFan || type == xenos::PrimitiveType::kQuadList;
}

std::vector<uint32_t> Expand(xenos::PrimitiveType type, const std::vector<uint32_t>& in) {
  std::vector<uint32_t> out;
  if (type == xenos::PrimitiveType::kTriangleFan) {
    for (size_t i = 2; i < in.size(); ++i) out.insert(out.end(), {in[0], in[i - 1], in[i]});
  } else if (type == xenos::PrimitiveType::kQuadList) {
    for (size_t i = 0; i + 3 < in.size(); i += 4) {
      out.insert(out.end(), {in[i], in[i + 1], in[i + 2], in[i], in[i + 2], in[i + 3]});
    }
  }
  return out;
}

std::atomic<bool> g_warn_ucp_shape{false};
std::atomic<bool> g_warn_ucp_cull{false};
std::atomic<bool> g_warn_ucp_scale{false};
std::atomic<bool> g_warn_blend_color{false};

// PA_SU_POLY_OFFSET converted as the SDK's pipeline cache does (draw_util GetPreferredFacePolygonOffset and
// GetD3D10IntegerPolygonOffset): the face most likely drawn, in host integer bias units and 1/16-subpixel slopes.
template <typename Device>
void PolygonOffset(const Device& device, const reg::PA_SU_SC_MODE_CNTL& mode, bool polygonal, PipelineState& s) {
  const auto& p = device.m_PointPacket;
  float scale = 0.0f, offset = 0.0f;
  if (polygonal) {
    if (mode.poly_offset_front_enable && !mode.cull_front) {
      scale = p.PolyOffsetFrontScale, offset = p.PolyOffsetFrontOffset;
    }
    if (mode.poly_offset_back_enable && !mode.cull_back && scale == 0.0f && offset == 0.0f) {
      scale = p.PolyOffsetBackScale, offset = p.PolyOffsetBackOffset;
    }
  } else if (mode.poly_offset_para_enable) {
    scale = p.PolyOffsetFrontScale, offset = p.PolyOffsetFrontOffset;
  }
  if (scale == 0.0f && offset == 0.0f) return;
  // Float24 depth moves in steps of 2^3 float32 ULPs; unorm24 in 1/(2^24 - 1) of the range.
  const bool float24 =
      device.m_DestinationPacket.DepthInfo.get().depth_format == xenos::DepthRenderTargetFormat::kD24FS8;
  int32_t bias = int32_t(std::ceil(std::abs(offset) * (float24 ? float(1u << 21) : float((1u << 24) - 1))));
  if (float24) bias <<= 3;
  s.depth_bias = offset < 0.0f ? -bias : bias;
  s.slope_bias = scale * xenos::kPolygonOffsetScaleSubpixelUnit;
}

// PA_CL_UCP planes act on the position the VS wrote; the pack's clip_planes GS sees it after the ndc transform, so
// each enabled plane is carried through that transform. Disabled planes stay zero, which never clips.
template <typename Device>
bool ClipPlanes(const Device& device, uint32_t enabled, SystemConstants& sys) {
  for (uint32_t i = 0; i < 6; ++i) {
    if (!(enabled & (1u << i))) continue;
    const float p[4] = {device.m_ClipPlanes[i][0], device.m_ClipPlanes[i][1], device.m_ClipPlanes[i][2],
                        device.m_ClipPlanes[i][3]};
    float w = p[3];
    for (uint32_t c = 0; c < 3; ++c) {
      if (sys.ndc_scale[c] == 0.0f) return false;
      sys.user_clip_planes[i][c] = p[c] / sys.ndc_scale[c];
      w -= p[c] * sys.ndc_offset[c] / sys.ndc_scale[c];
    }
    sys.user_clip_planes[i][3] = w;
  }
  return true;
}

// VGT_MULTI_PRIM_IB_RESET_INDX when the draw is an indexed strip with PA_SU_SC_MODE_CNTL.multi_prim_ib_ena set,
// else UINT32_MAX; the check uses the low 24 bits of the index, as the SDK's register notes say.
uint32_t RestartIndex(const D3DDevice& device, xenos::PrimitiveType type) {
  if (type != xenos::PrimitiveType::kTriangleStrip && type != xenos::PrimitiveType::kLineStrip) return UINT32_MAX;
  if (!device.m_ControlPacket.ModeControl.get().multi_prim_ib_ena) return UINT32_MAX;
  return device.m_ValuesPacket.MultiPrimIbResetIndx.get().reset_indx;
}

// Strips with the multi-primitive reset index split into independent lists; D3D12 only cuts at all-ones.
bool SplitAtRestart(xenos::PrimitiveType type, uint32_t reset, std::vector<uint32_t>& indices) {
  if (std::find_if(indices.begin(), indices.end(), [reset](uint32_t v) { return (v & 0xFFFFFF) == reset; }) ==
      indices.end()) {
    return false;
  }
  std::vector<uint32_t> out;
  size_t start = 0;
  for (size_t i = 0; i <= indices.size(); ++i) {
    if (i < indices.size() && (indices[i] & 0xFFFFFF) != reset) continue;
    const uint32_t* run = indices.data() + start;
    const size_t n = i - start;
    if (type == xenos::PrimitiveType::kTriangleStrip) {
      // Odd triangles swap their first two vertices to keep the strip's winding.
      for (size_t k = 2; k < n; ++k) {
        if (k & 1) out.insert(out.end(), {run[k - 1], run[k - 2], run[k]});
        else out.insert(out.end(), {run[k - 2], run[k - 1], run[k]});
      }
    } else {
      for (size_t k = 1; k < n; ++k) out.insert(out.end(), {run[k - 1], run[k]});
    }
    start = i + 1;
  }
  indices = std::move(out);
  return true;
}

// A converted index run: the little-endian, expanded, restart-split indices of one (buffer, start, count) at one
// version of that memory, kept so the conversion and the upload happen once per change rather than per frame.
struct IndexRun {
  uint32_t version = 0;
  SentRange sent;
  std::shared_future<PreparedIndices> prepared;
  std::optional<PreparedIndices> ready;
};
std::unordered_map<uint64_t, IndexRun> g_index_runs;

void ForgetIndex(uint64_t key) {
  const auto it = g_index_runs.find(key);
  if (it != g_index_runs.end()) it->second.sent = {};
}

// Places the run named by `key` in the record, converting through `convert` (which returns the indices and whether
// a restart split changed the topology) only when this version has not been converted yet.
template <typename GetState, typename MakeConvert>
void PlaceIndices(DrawRecord& r, uint64_t key, GetState get_state, MakeConvert make_convert,
                  xenos::PrimitiveType type, size_t input_bytes) {
  BurstTiming timing(2);
  const uint64_t frame = FrameIndex();
  ForgetEvicted();
  if (g_index_runs.size() > 8192) g_index_runs.clear();
  IndexRun& run = g_index_runs[key];
  const auto state = get_state(run.sent.range_cache);
  const uint32_t version = state.registered ? state.generation : uint32_t(frame);
  if ((!run.prepared.valid() && !run.ready) || run.version != version) {
    run.ready.reset();
    run.prepared = {};
    run.version = version;
    auto work = [convert = make_convert(), type] {
      auto [indices, split] = convert();
      PreparedIndices prepared;
      prepared.count = uint32_t(indices.size());
      prepared.topology = split ? uint8_t(type == xenos::PrimitiveType::kLineStrip ? xenos::PrimitiveType::kLineList
                                                                            : xenos::PrimitiveType::kTriangleList)
                         : 0;
      auto bytes = std::make_shared<std::vector<uint8_t>>(indices.size() * 4);
      std::memcpy(bytes->data(), indices.data(), bytes->size());
      prepared.bytes = std::move(bytes);
      return prepared;
    };
    if (REXCVAR_GET(gpu_parallel_prepare) && !Tracing()) {
      run.prepared = PrepareQueue().Post(std::move(work), input_bytes).share();
    } else {
      run.ready = work();
    }
  }
  if (!run.ready && run.prepared.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    run.ready = run.prepared.get();
    run.prepared = {};
  }
  r.index32 = true;
  if (!run.ready) r.prepared_indices = run.prepared;
  r.index.key = key;
  r.index.version = version;
  r.index.persistent = state.registered;
  SentRange& sent = run.sent;
  const bool send = state.registered ? (sent.frame == UINT64_MAX || sent.version != version) : sent.frame != frame;
  if (send) {
    r.send_prepared_indices = true;
    sent.version = version;
    sent.frame = frame;
  }
  if (run.ready) ApplyIndices(r, *run.ready);
}

struct DrawStateSnapshot {
  decltype(D3DDevice::m_ControlPacket) m_ControlPacket;
  decltype(D3DDevice::m_ValuesPacket) m_ValuesPacket;
  decltype(D3DDevice::m_DestinationPacket) m_DestinationPacket;
  decltype(D3DDevice::m_WindowPacket) m_WindowPacket;
  decltype(D3DDevice::m_MiscPacket) m_MiscPacket;
  decltype(D3DDevice::m_TessellatorPacket) m_TessellatorPacket;
  decltype(D3DDevice::m_PointPacket) m_PointPacket;
  decltype(D3DDevice::m_ClipPlanes) m_ClipPlanes;
};
struct DrawWork : PreparedDrawState {
  DrawStateSnapshot device;
  std::array<uint32_t, 32 * 6> sampler_fetch;
  DrawArgs args;
  std::array<ResourceDesc, 4> color;
  ResourceDesc depth;
  const ShaderEntry* vs = nullptr;
  const ShaderEntry* ps = nullptr;
  std::shared_ptr<DrawConstants> constants;
  std::shared_ptr<const FloatConstantBlock> vs_constants, ps_constants;
  std::vector<LiteralConstant> literals;
  bool disable_cull = false;
  bool convert_constants = false;
  uint8_t topology = 0, geometry = 0;
};

void PrepareDraw(DrawWork& work) {
  const auto* device = &work.device;
  const auto& a = work.args;
  DrawRecord r;
  r.color = work.color;
  r.depth = work.depth;
  r.vs = work.vs;
  r.ps = work.ps;
  auto c = work.constants;
  for (size_t i = 0; i < 256 * 4; ++i) {
    const uint32_t vs = __builtin_bswap32(work.vs_constants->words[i]);
    const uint32_t ps = __builtin_bswap32(work.ps_constants->words[i]);
    std::memcpy(&c->vs_float[i], &vs, sizeof(vs));
    std::memcpy(&c->ps_float[i], &ps, sizeof(ps));
  }
  for (const auto& literal : work.literals) {
    float* destination = literal.dword < 256 * 4 ? &c->vs_float[literal.dword]
                                                : &c->ps_float[literal.dword - 256 * 4];
    std::memcpy(destination, &literal.bits, sizeof(literal.bits));
  }
  const ResourceDesc& size_source = r.color[0].id ? r.color[0] : r.depth;
  const auto geometry = GeometryExpansion(work.geometry);
  const auto host_type = xenos::PrimitiveType(work.topology);
  const auto& cp = device->m_ControlPacket;
  const auto mode = cp.ModeControl.get();
  const auto depth = cp.DepthControl.get();
  const auto color_control = cp.ColorControl.get();
  const auto vte = cp.VteControl.get();
  const auto clip = cp.ClipControl.get();
  const auto& vp = device->m_ValuesPacket;
  auto& s = r.state;
  s.cull = mode.cull_front ? 1 : mode.cull_back ? 2 : 0;
  s.front_ccw = mode.face ? 0 : 1;
  if (work.disable_cull) s.cull = 0;
  s.wireframe = mode.poly_mode != xenos::PolygonModeEnable::kDisabled &&
                mode.polymode_front_ptype == xenos::PolygonType::kLines;
  if (r.depth.id) {
    const bool polygonal = a.type != xenos::PrimitiveType::kPointList && a.type != xenos::PrimitiveType::kLineList &&
                           a.type != xenos::PrimitiveType::kLineStrip && a.type != xenos::PrimitiveType::kLineLoop;
    PolygonOffset(*device, mode, polygonal, s);
  }
  if (r.depth.id) {
    s.depth_enable = depth.z_enable;
    s.depth_write = depth.z_write_enable;
    s.depth_func = uint8_t(depth.zfunc);
    s.stencil_enable = depth.stencil_enable;
    s.stencil_func = uint8_t(depth.stencilfunc);
    s.stencil_fail = uint8_t(depth.stencilfail);
    s.stencil_zfail = uint8_t(depth.stencilzfail);
    s.stencil_pass = uint8_t(depth.stencilzpass);
    const bool bf = depth.backface_enable;
    s.stencil_func_bf = uint8_t(bf ? depth.stencilfunc_bf : depth.stencilfunc);
    s.stencil_fail_bf = uint8_t(bf ? depth.stencilfail_bf : depth.stencilfail);
    s.stencil_zfail_bf = uint8_t(bf ? depth.stencilzfail_bf : depth.stencilzfail);
    s.stencil_pass_bf = uint8_t(bf ? depth.stencilzpass_bf : depth.stencilzpass);
    const auto ref = vp.StencilRefMask.get();
    s.stencil_ref = uint8_t(ref.stencilref);
    s.stencil_read_mask = uint8_t(ref.stencilmask);
    s.stencil_write_mask = uint8_t(ref.stencilwritemask);
    s.depth_format = uint8_t(r.depth.format);
  }
  const uint32_t color_mask = vp.ColorMask.raw;
  const uint32_t shadow_blend[4] = {cp.BlendControl0.raw, cp.BlendControl1.raw, cp.BlendControl2.raw,
                                    cp.BlendControl3.raw};
  bool constant_color = false, constant_alpha = false;
  for (uint32_t i = 0; i < 4; ++i) {
    if (!r.color[i].id) continue;
    s.color_count = uint8_t(i + 1);
    s.color_format[i] = uint8_t(r.color[i].format);
    uint32_t blend = shadow_blend[i];
    s.blend[i] = blend;
    s.blend_enable[i] = BlendIsCopy(blend) ? 0 : 1;
    s.write_mask[i] = uint8_t((color_mask >> (4 * i)) & 0xF);
    reg::RB_BLENDCONTROL b;
    b.value = blend;
    if (s.blend_enable[i]) {
      auto is = [](xenos::BlendFactor f, uint32_t lo) { return uint32_t(f) == lo || uint32_t(f) == lo + 1; };
      constant_color |= is(b.color_srcblend, 12) || is(b.color_destblend, 12);
      constant_alpha |= is(b.color_srcblend, 14) || is(b.color_destblend, 14);
      r.blend_constant_used |= constant_color || constant_alpha || is(b.alpha_srcblend, 12) ||
                               is(b.alpha_destblend, 12) || is(b.alpha_srcblend, 14) || is(b.alpha_destblend, 14);
    }
  }
  // The host has one blend-factor colour: colour channels read its RGB and the alpha channel its A.
  if (r.blend_constant_used) {
    const float red = vp.BlendRed, green = vp.BlendGreen, blue = vp.BlendBlue, alpha = vp.BlendAlpha;
    const bool alpha_only = constant_alpha && !constant_color;
    if (constant_alpha && constant_color) {
      Once(g_warn_blend_color, "a draw blends with both the constant colour and the constant alpha; colour wins");
    }
    r.blend_constant[0] = alpha_only ? alpha : red;
    r.blend_constant[1] = alpha_only ? alpha : green;
    r.blend_constant[2] = alpha_only ? alpha : blue;
    r.blend_constant[3] = alpha;
  }
  // Colour targets must be contiguous from slot 0 for the host framebuffer.
  for (uint32_t i = 0; i < s.color_count; ++i) {
    if (!r.color[i].id) {
      s.color_count = uint8_t(i);
      break;
    }
  }

  s.topology = work.topology;
  s.geometry = work.geometry;
  s.vs_entry = r.vs->id;
  s.ps_entry = r.ps ? r.ps->id : 0;
  auto& sys = c->system;
  sys.flags = (vte.vtx_xy_fmt ? 1u << 1 : 0) | (vte.vtx_z_fmt ? 1u << 2 : 0) | (vte.vtx_w0_fmt ? 1u << 3 : 0);
  const uint32_t alpha_func = color_control.alpha_test_enable ? uint32_t(color_control.alpha_func) : 7u;
  sys.flags |= alpha_func << 7;
  sys.alpha_test_reference = vp.AlphaRef;
  sys.vertex_index_offset = uint32_t(a.base_vertex);
  sys.vertex_index_min = vp.MinVtxIndx.get().min_indx;
  sys.vertex_index_max = vp.MaxVtxIndx.get().max_indx;
  const reg::RB_COLOR_INFO infos[4] = {device->m_DestinationPacket.Color0Info.get(),
                                       device->m_DestinationPacket.Color1Info.get(),
                                       device->m_DestinationPacket.Color2Info.get(),
                                       device->m_DestinationPacket.Color3Info.get()};
  for (uint32_t i = 0; i < 4; ++i) sys.color_exp_bias[i] = std::ldexp(1.0f, infos[i].color_exp_bias);
  // Host viewport = the whole target; the guest viewport, window offset and half-pixel rule go into ndc scale/offset.
  const float w = float(size_source.width), h = float(size_source.height);
  float sx = vte.vport_x_scale_ena ? float(vp.VportXScale) : 1.0f;
  float ox = vte.vport_x_offset_ena ? float(vp.VportXOffset) : 0.0f;
  float sy = vte.vport_y_scale_ena ? float(vp.VportYScale) : 1.0f;
  float oy = vte.vport_y_offset_ena ? float(vp.VportYOffset) : 0.0f;
  float sz = vte.vport_z_scale_ena ? float(vp.VportZScale) : 1.0f;
  float oz = vte.vport_z_offset_ena ? float(vp.VportZOffset) : 0.0f;
  const auto window = device->m_WindowPacket.WindowOffset.get();
  if (mode.vtx_window_offset_enable) {
    ox += float(window.window_x_offset);
    oy += float(window.window_y_offset);
  }
  if (device->m_MiscPacket.VtxControl.get().pix_center == xenos::PixelCenter::kD3DZero) {
    ox += 0.5f;
    oy += 0.5f;
  }
  sys.ndc_scale[0] = 2.0f * sx / w;
  sys.ndc_offset[0] = 2.0f * ox / w - 1.0f;
  sys.ndc_scale[1] = -2.0f * sy / h;
  sys.ndc_offset[1] = 1.0f - 2.0f * oy / h;
  sys.ndc_scale[2] = sz;
  sys.ndc_offset[2] = oz;
  const auto point = device->m_TessellatorPacket.PointSize.get();
  const auto minmax = device->m_TessellatorPacket.PointMinMax.get();
  sys.point_size[0] = float(point.width) / 8.0f * 2.0f;
  sys.point_size[1] = float(point.height) / 8.0f * 2.0f;
  sys.point_size[2] = float(minmax.min_size) / 8.0f * 2.0f;
  sys.point_size[3] = float(minmax.max_size) / 8.0f * 2.0f;
  r.viewport[0] = 0;
  r.viewport[1] = 0;
  r.viewport[2] = w;
  r.viewport[3] = h;
  r.viewport[4] = 0;
  r.viewport[5] = 1;
  r.depth_clip = !clip.clip_disable;

  // User clip planes go through the pack's clip_planes GS, which takes triangles only.
  if (clip.ucp_ena && !clip.clip_disable) {
    const bool triangles = geometry == GeometryExpansion::kNone &&
                           (host_type == xenos::PrimitiveType::kTriangleList ||
                            host_type == xenos::PrimitiveType::kTriangleStrip);
    if (clip.ucp_cull_only_ena) Once(g_warn_ucp_cull, "cull-only user clip planes are applied as clipping planes");
    if (!triangles) {
      Once(g_warn_ucp_shape, "user clip planes on point, line or rectangle draws are not applied");
    } else if (!ClipPlanes(*device, clip.ucp_ena, sys)) {
      Once(g_warn_ucp_scale, "user clip planes with a zero viewport scale are not applied");
    } else {
      s.clip_planes = 1;
    }
  }
  // The trimmed VS links only when it writes every input the PS declares; a container-less PS declares all 16.
  s.use_trimmed_vs = geometry == GeometryExpansion::kNone && !s.clip_planes && r.vs->has_trim() &&
                     (!r.ps || (r.ps->read_ivar & ~r.vs->written_ovar) == 0);

  // Scissor: the window scissor intersected with the screen scissor, in target pixels.
  const auto tl = device->m_WindowPacket.WindowScissorTL.get();
  const auto br = device->m_WindowPacket.WindowScissorBR.get();
  int32_t x0 = int32_t(tl.tl_x), y0 = int32_t(tl.tl_y), x1 = int32_t(br.br_x), y1 = int32_t(br.br_y);
  if (!tl.window_offset_disable) {
    x0 += window.window_x_offset, x1 += window.window_x_offset;
    y0 += window.window_y_offset, y1 += window.window_y_offset;
  }
  const auto stl = device->m_DestinationPacket.ScreenScissorTL.get();
  const auto sbr = device->m_DestinationPacket.ScreenScissorBR.get();
  x0 = std::max(x0, int32_t(stl.tl_x)), y0 = std::max(y0, int32_t(stl.tl_y));
  x1 = std::min(x1, int32_t(sbr.br_x)), y1 = std::min(y1, int32_t(sbr.br_y));
  r.scissor = Rect{std::clamp(x0, 0, int32_t(w)), std::clamp(y0, 0, int32_t(h)), std::clamp(x1, 0, int32_t(w)),
                   std::clamp(y1, 0, int32_t(h))};


  r.samplers = std::move(work.samplers);
  r.samplers.clear();
  size_t sampler_count = 0;
  for (const ShaderEntry* e : {r.vs, r.ps}) {
    if (e) for (const auto& b : e->bindings) sampler_count += bool(b.sampler);
  }
  r.samplers.reserve(sampler_count);
  for (const ShaderEntry* e : {r.vs, r.ps}) {
    if (!e) continue;
    const uint8_t stage = e == r.vs ? 0 : 1;
    for (uint32_t slot = 0; slot < e->bindings.size(); ++slot) {
      const PackBinding& b = e->bindings[slot];
      if (!b.sampler) continue;
        xenos::xe_gpu_texture_fetch_t f;
        std::memcpy(&f, work.sampler_fetch.data() + (b.fetch_constant & 31) * 6, sizeof(f));
        // The sampler index and state depend on the fetch words and the pack overrides alone.
        struct SamplerSlotCache {
          bool valid = false;
          uint32_t words[6] = {};
          uint64_t overrides = 0;
          uint8_t index = 0;
          uint32_t state = 0;
        };
        thread_local SamplerSlotCache sampler_cache[32];
        SamplerSlotCache& sc = sampler_cache[b.fetch_constant & 31];
        const uint64_t overrides = BindingOverrides(b);
        if (sc.valid && sc.overrides == overrides && std::memcmp(sc.words, &f, sizeof(sc.words)) == 0) {
          r.samplers.push_back(SamplerBinding{uint8_t(slot), sc.index, stage, sc.state});
          continue;
        }
        const uint32_t census[5] = {(f.dword_0 >> 10) & 0x1FF, (f.dword_3 >> 19) & 0x3FF, f.dword_5 & 3,
                                    (f.dword_4 >> 2) & 0xFF,
                                    uint32_t(b.mag) | uint32_t(b.min) << 8 | uint32_t(b.mip) << 16 |
                                        uint32_t(b.aniso) << 24};
        if (FirstTime(XXH3_64bits(census, sizeof(census)) ^ 0x73616D706C657200ull)) {
          REXGPU_INFO("[gpu] sampler census: clamp {}/{}/{} filter mag {} min {} mip {} aniso {} border {} mips {}-{} "
                      "overrides {}/{}/{}/{}",
                      uint32_t(f.clamp_x), uint32_t(f.clamp_y), uint32_t(f.clamp_z), uint32_t(f.mag_filter),
                      uint32_t(f.min_filter), uint32_t(f.mip_filter), uint32_t(f.aniso_filter),
                      uint32_t(f.border_color), uint32_t(f.mip_min_level), uint32_t(f.mip_max_level), b.mag, b.min,
                      b.mip, b.aniso);
        }
        sc.valid = true;
        std::memcpy(sc.words, &f, sizeof(sc.words));
        sc.overrides = overrides;
        sc.index = SamplerIndex(f, b);
        sc.state = SamplerState(f, b);
        r.samplers.push_back(SamplerBinding{uint8_t(slot), sc.index, stage, sc.state});

    }
  }
  work.state = r.state;
  std::copy_n(r.viewport, 6, work.viewport.begin());
  work.scissor = r.scissor;
  work.depth_clip = r.depth_clip;
  work.blend_constant_used = r.blend_constant_used;
  std::copy_n(r.blend_constant, 4, work.blend_constant.begin());
  work.samplers = std::move(r.samplers);
  work.constants.reset();
  work.vs_constants.reset();
  work.ps_constants.reset();
  work.literals.clear();
}

std::shared_ptr<DrawWork> AllocateDrawWork() {
  struct Chunk { std::array<DrawWork, 64> items; };
  thread_local std::vector<std::shared_ptr<Chunk>> pool;
  thread_local std::shared_ptr<Chunk> chunk;
  thread_local size_t next = 64, cursor = 0;
  if (next == 64) {
    chunk.reset();
    for (size_t checked = 0; checked < pool.size(); ++checked) {
      cursor = (cursor + 1) % pool.size();
      if (pool[cursor].use_count() == 1) { chunk = pool[cursor]; break; }
    }
    if (!chunk) {
      chunk = std::make_shared<Chunk>();
      if (pool.size() < 64) pool.push_back(chunk);
    }
    next = 0;
  }
  return std::shared_ptr<DrawWork>(chunk, &chunk->items[next++]);
}

std::shared_ptr<DrawWork> g_draw_first;
size_t g_draw_count = 0;

void FlushDrawBatch() {
  if (!g_draw_first) return;
  const size_t count = std::exchange(g_draw_count, 0);
  PrepareQueue().PostDetached([first = std::move(g_draw_first), count] {
    for (size_t i = 0; i < count; ++i) PrepareDraw(first.get()[i]);
  }, count * (sizeof(DrawWork) + sizeof(DrawConstants)));
}

void QueueDrawWork(const std::shared_ptr<PreparedDrawState>& state) {
  auto work = std::static_pointer_cast<DrawWork>(state);
  if (!work->convert_constants) { PrepareDraw(*work); return; }
  if (g_draw_first && (g_draw_first.owner_before(work) || work.owner_before(g_draw_first) ||
                       work.get() != g_draw_first.get() + g_draw_count)) FlushDrawBatch();
  if (!g_draw_first) g_draw_first = std::move(work);
  if (++g_draw_count == 32) FlushDrawBatch();
}

void FinishDrawState(DrawRecord& r) {
  if (!r.prepared_state) return;
  auto prepared = std::move(r.prepared_state);
  const auto topology = r.state.topology;
  r.state = prepared->state;
  r.state.topology = topology;
  std::copy_n(prepared->viewport.begin(), 6, r.viewport);
  r.scissor = prepared->scissor;
  r.depth_clip = prepared->depth_clip;
  r.blend_constant_used = prepared->blend_constant_used;
  std::copy_n(prepared->blend_constant.begin(), 4, r.blend_constant);
  r.samplers = std::move(prepared->samplers);
}

// Builds everything but the inline vertex and index bytes; false drops the draw with the reason logged. `device` is
// the struct to read, normally the one at a.device, or a merged copy when a command buffer replays.
bool Build(PPCContext& ctx, uint8_t* base, const DrawArgs& a, DrawRecord& r, const D3DDevice* device,
           const VsSnapshot* snap = nullptr) {
  DeviceState& ds = g_devices[a.device];
  // Section timing for the statistics window; each mark closes the section named before it.
  auto section_start = std::chrono::steady_clock::now();
  auto mark = [&](int section) {
    const auto now = std::chrono::steady_clock::now();
    AddCaptureBreakdown(section, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - section_start).count()));
    section_start = now;
  };

  // Targets.
  const uint32_t edram_mode = uint32_t(device->m_ControlPacket.EdramModeControl.raw) & 7;
  const bool depth_only = edram_mode == 5;
  for (uint32_t i = 0; i < 4 && !depth_only; ++i) {
    if (device->render_targets[i]) r.color[i] = tracker::Surface(device->render_targets[i], ResourceKind::kColorTarget);
  }
  if (device->depth_stencil) r.depth = tracker::Surface(device->depth_stencil, ResourceKind::kDepthTarget);
  const ResourceDesc& size_source = r.color[0].id ? r.color[0] : r.depth;
  if (!size_source.id) return false;

  // Shaders.
  const VsInfo* vs = ResolveVs(ctx, base, ds, device, a.inline_data ? int(a.stride >> 2) : -1, snap);
  if (!vs || !vs->entry) return false;
  const PsInfo* ps = ResolvePs(ds, device);
  if (device->pixel_shader && (!ps || !ps->entry)) return false;
  r.vs = vs->entry;
  r.ps = ps ? ps->entry : nullptr;

  const auto mode = device->m_ControlPacket.ModeControl.get();
  if (mode.cull_front && mode.cull_back) return true;
  auto& s = r.state;
  const auto& control = device->m_ControlPacket;
  uint32_t blends[4] = {control.BlendControl0.raw, control.BlendControl1.raw,
                       control.BlendControl2.raw, control.BlendControl3.raw};
  // Topology.
  GeometryExpansion geometry = GeometryExpansion::kNone;
  xenos::PrimitiveType host_type = a.type;
  switch (a.type) {
    case xenos::PrimitiveType::kPointList: geometry = GeometryExpansion::kPointList; break;
    case xenos::PrimitiveType::kRectangleList:
      geometry = GeometryExpansion::kRectList;
      host_type = xenos::PrimitiveType::kTriangleList;
      break;
    case xenos::PrimitiveType::kTriangleFan:
    case xenos::PrimitiveType::kQuadList: host_type = xenos::PrimitiveType::kTriangleList; break;
    case xenos::PrimitiveType::kLineList:
    case xenos::PrimitiveType::kLineStrip:
    case xenos::PrimitiveType::kTriangleList:
    case xenos::PrimitiveType::kTriangleStrip: break;
    default:
      if (FirstTime(0x7072696D00000000ull | uint32_t(a.type))) {
        REXGPU_ERROR("[gpu] primitive type {} is not handled; draw skipped", uint32_t(a.type));
      }
      return false;
  }
  s.topology = uint8_t(host_type);
  s.geometry = uint8_t(geometry);
  s.vs_entry = r.vs->id;
  s.ps_entry = r.ps ? r.ps->id : 0;

  mark(0);
  // Constants: the shadow is byte-swapped per dword; literals override what the shader defines itself.
  auto c = AllocateConstants();
  const auto& k = device->m_Constants;
  const bool parallel_constants = REXCVAR_GET(gpu_parallel_prepare) && !Tracing();
  static_assert(sizeof(k.VertexShaderF) == 256 * 16 && sizeof(k.PixelShaderF) == 256 * 16);
  auto work = AllocateDrawWork();
  work->vs_constants = ds.float_snapshots.Capture(0, &k.VertexShaderF[0][0]);
  work->ps_constants = ds.float_snapshots.Capture(1, &k.PixelShaderF[0][0]);
  work->literals.clear();
  CaptureLiterals(device->vertex_shader + guest::kVsHeader, Load32(device->vertex_shader + guest::kVsCodeBase),
                  work->literals);
  if (device->pixel_shader) {
    CaptureLiterals(device->pixel_shader + guest::kPsHeader, Load32(device->pixel_shader + guest::kPsCodeBase),
                    work->literals);
  }
  for (uint32_t i = 0; i < 4; ++i) {
    c->bool_loop[i] = k.VertexShaderB[i];
    c->bool_loop[4 + i] = k.PixelShaderB[i];
  }
  for (uint32_t i = 0; i < 16; ++i) {
    c->bool_loop[8 + i] = k.VertexShaderI[i];
    c->bool_loop[24 + i] = k.PixelShaderI[i];
  }
  for (uint32_t g = 0; g < 32; ++g) {
    for (uint32_t d = 0; d < 6; ++d) c->fetch[g * 6 + d] = k.Fetch[g].dword[d];
  }
  c->vfetch = vs->vfetch;

  const float w = float(size_source.width), h = float(size_source.height);
  const auto window = device->m_WindowPacket.WindowOffset.get();
  // Scissor: the window scissor intersected with the screen scissor, in target pixels.
  const auto tl = device->m_WindowPacket.WindowScissorTL.get();
  const auto br = device->m_WindowPacket.WindowScissorBR.get();
  int32_t x0 = int32_t(tl.tl_x), y0 = int32_t(tl.tl_y), x1 = int32_t(br.br_x), y1 = int32_t(br.br_y);
  if (!tl.window_offset_disable) {
    x0 += window.window_x_offset, x1 += window.window_x_offset;
    y0 += window.window_y_offset, y1 += window.window_y_offset;
  }
  const auto stl = device->m_DestinationPacket.ScreenScissorTL.get();
  const auto sbr = device->m_DestinationPacket.ScreenScissorBR.get();
  x0 = std::max(x0, int32_t(stl.tl_x)), y0 = std::max(y0, int32_t(stl.tl_y));
  x1 = std::min(x1, int32_t(sbr.br_x)), y1 = std::min(y1, int32_t(sbr.br_y));
  r.scissor = Rect{std::clamp(x0, 0, int32_t(w)), std::clamp(y0, 0, int32_t(h)), std::clamp(x1, 0, int32_t(w)),
                   std::clamp(y1, 0, int32_t(h))};

  if (r.scissor.right <= r.scissor.left || r.scissor.bottom <= r.scissor.top) return true;

  mark(1);
  // Vertex data: every fetch constant the patched microcode reads.
  r.streams.reserve(vs->fetches.size());
  for (uint8_t index : vs->fetches) {
    if (a.inline_data && index == guest::kInlineVertexFetch) continue;
    const uint32_t d0 = c->fetch[(index / 3) * 6 + (index % 3) * 2];
    const uint32_t d1 = c->fetch[(index / 3) * 6 + (index % 3) * 2 + 1];
    if ((d0 & 3) != uint32_t(xenos::FetchConstantType::kVertex)) continue;
    const uint32_t guest_base = d0 & ~3u;
    const uint32_t size = d1 & 0x3FFFFFC;
    if (Tracing()) REXGPU_INFO("[gpu]   vfetch {} base {:#010x} size {:#x}", index, guest_base, size);
    if (!guest_base || !size) continue;
    r.streams.push_back(CaptureStream(index, guest_base & 0x1FFFFFFF, size));
  }

  // Resource ownership and guest reads stay at the commit point; sampler assembly uses the captured fetch words.
  size_t texture_count = 0;
  for (const ShaderEntry* e : {r.vs, r.ps}) {
    if (e) for (const auto& b : e->bindings) texture_count += !b.sampler;
  }
  r.textures.reserve(texture_count);
  for (const ShaderEntry* e : {r.vs, r.ps}) {
    if (!e) continue;
    const uint8_t stage = e == r.vs ? 0 : 1;
    for (uint32_t slot = 0; slot < e->bindings.size(); ++slot) {
      const PackBinding& b = e->bindings[slot];
      if (b.sampler) continue;
      TextureBinding t = BindTexture(device, b, uint8_t(slot));
      t.stage = stage;
      r.textures.push_back(std::move(t));
    }
  }

  mark(2);
  // Geometry.
  r.count = a.count;
  r.indexed = a.indexed;
  if (a.indexed && !a.inline_data) {
    const uint32_t ib = device->index_buffer;
    if (!ib) return false;
    const auto* header = Guest<guest::D3DIndexBuffer>(ib);
    const uint32_t common = header->resource.Common;
    const bool index32 = (common & guest::kIndexBufferIndex32) != 0;
    const uint32_t endian = (common & guest::kIndexBufferEndianMask) >> guest::kIndexBufferEndianShift;
    const uint32_t address = uint32_t(header->Address) + a.start * (index32 ? 4 : 2);
    const uint32_t reset = RestartIndex(*device, a.type);
    const uint64_t parts[4] = {address, uint64_t(a.count) | uint64_t(reset) << 32,
                               uint64_t(index32) | uint64_t(endian) << 8 | uint64_t(a.type) << 16, 0x494E4458};
    PlaceIndices(r, XXH3_64bits(parts, sizeof(parts)) | 1, [&](memory::QueryCache& cached) {
      return memory::Query(address, a.count * (index32 ? 4 : 2), cached);
    }, [&] {
      const auto* src = Guest<uint8_t>(address);
      std::vector<uint8_t> snapshot(src, src + size_t(a.count) * (index32 ? 4 : 2));
      return [snapshot = std::move(snapshot), count = a.count, type = a.type, index32, endian, reset] {
        auto indices = ReadIndices(snapshot.data(), count, index32, endian);
        if (NeedsExpansion(type)) indices = Expand(type, indices);
        const bool split = reset != UINT32_MAX && SplitAtRestart(type, reset, indices);
        return std::make_pair(std::move(indices), split);
      };
    }, a.type, size_t(a.count) * 32);
  } else if (!a.indexed && NeedsExpansion(a.type)) {
    const uint64_t parts[2] = {uint64_t(a.count) | uint64_t(a.type) << 32, 0x455850};
    r.indexed = true;
    PlaceIndices(r, XXH3_64bits(parts, sizeof(parts)) | 1, [](memory::QueryCache&) {
      return memory::RangeState{true, 1};
    }, [&] {
      return [count = a.count, type = a.type] {
        std::vector<uint32_t> seq(count);
        for (uint32_t i = 0; i < count; ++i) seq[i] = i;
        return std::make_pair(Expand(type, seq), false);
      };
    }, a.type, size_t(a.count) * 32);
  }
  std::memcpy(&work->device.m_ControlPacket, &device->m_ControlPacket, sizeof(device->m_ControlPacket));
  std::memcpy(&work->device.m_ValuesPacket, &device->m_ValuesPacket, sizeof(device->m_ValuesPacket));
  std::memcpy(&work->device.m_DestinationPacket, &device->m_DestinationPacket, sizeof(device->m_DestinationPacket));
  std::memcpy(&work->device.m_WindowPacket, &device->m_WindowPacket, sizeof(device->m_WindowPacket));
  std::memcpy(&work->device.m_MiscPacket, &device->m_MiscPacket, sizeof(device->m_MiscPacket));
  std::memcpy(&work->device.m_TessellatorPacket, &device->m_TessellatorPacket, sizeof(device->m_TessellatorPacket));
  std::memcpy(&work->device.m_PointPacket, &device->m_PointPacket, sizeof(device->m_PointPacket));
  std::memcpy(&work->device.m_ClipPlanes, &device->m_ClipPlanes, sizeof(device->m_ClipPlanes));
  work->args = a;
  work->sampler_fetch = c->fetch;
  work->color = r.color;
  work->depth = r.depth;
  work->vs = r.vs;
  work->ps = r.ps;
  work->constants = c;
  work->disable_cull = snap && !REXCVAR_GET(gpu_replay_cull);
  work->convert_constants = parallel_constants;
  work->topology = uint8_t(host_type);
  work->geometry = uint8_t(geometry);
  auto& cp = work->device.m_ControlPacket;
  cp.BlendControl0.raw = blends[0];
  cp.BlendControl1.raw = blends[1];
  cp.BlendControl2.raw = blends[2];
  cp.BlendControl3.raw = blends[3];
  r.prepared_state = std::move(work);
  r.constants = std::move(c);
  mark(3);
  return true;
}

// A Begin*Vertices record waiting for the game to write its data.
struct PendingInline {
  bool active = false;
  bool valid = false;
  DrawArgs args;
  DrawRecord record;
  uint32_t vertex_data = 0;
  uint32_t index_data = 0;
  xenos::PrimitiveType type = xenos::PrimitiveType::kNone;
};
thread_local PendingInline t_pending;
thread_local int t_depth = 0;

void Submit(DrawRecord&& r) {
  if (r.prepared_state) QueueDrawWork(r.prepared_state);
  if (Tracing()) {
    FinishDrawState(r);
    FinishIndices(r);
    REXGPU_INFO("[gpu] draw vs {} ps {} topo {} count {} idx {} streams {} tex {} rt {} depth {} viewport ({},{} {}x{}) "
                "scissor ({},{})-({},{}) depth {}/{}/{} stencil {}/{}/{} ops {}/{}/{} masks {:#x}/{:#x} blend {:#x}/{} "
                "mask {:#x}",
                r.state.vs_entry, r.state.ps_entry, r.state.topology, r.count, r.indexed, r.streams.size(),
                r.textures.size(), r.color[0].id, r.depth.id, r.viewport[0], r.viewport[1], r.viewport[2], r.viewport[3],
                r.scissor.left, r.scissor.top, r.scissor.right, r.scissor.bottom, r.state.depth_enable,
                r.state.depth_write, r.state.depth_func, r.state.stencil_enable, r.state.stencil_func,
                r.state.stencil_ref, r.state.stencil_fail, r.state.stencil_zfail, r.state.stencil_pass,
                r.state.stencil_read_mask, r.state.stencil_write_mask, r.state.blend[0], r.state.blend_enable[0],
                r.state.write_mask[0]);
    for (const auto& t : r.textures) {
      REXGPU_INFO("[gpu]   tex stage {} slot {} dim {} host {} {} {}x{} upload {}", t.stage, t.slot, t.dimension,
                  t.texture.id, HostFormatName(t.texture.format), t.texture.width, t.texture.height, t.upload != nullptr);
    }
  }
  ae::gpu::Submit(std::move(r));
}

void FinishInline() {
  PendingInline& p = t_pending;
  if (!p.active) return;
  p.active = false;
  if (!p.valid) return;
  std::lock_guard lock(g_mutex);
  CaptureTimer timer;
  DrawRecord& r = p.record;
  const DrawArgs& a = p.args;
  // Inline vertex data goes to the transient buffer with the other streams of the draw.
  if (p.vertex_data && a.vertex_bytes) {
    const auto* bytes = Guest<uint8_t>(p.vertex_data);
    StreamData s;
    s.fetch_index = guest::kInlineVertexFetch;
    s.size = a.vertex_bytes;
    g_captured_bytes += a.vertex_bytes;
    s.bytes = std::make_shared<const std::vector<uint8_t>>(bytes, bytes + a.vertex_bytes);
    r.streams.push_back(std::move(s));
    auto& fetch = r.constants->fetch;
    const uint32_t at = (guest::kInlineVertexFetch / 3) * 6 + (guest::kInlineVertexFetch % 3) * 2;
    fetch[at] = 3;
    fetch[at + 1] = (a.vertex_bytes & 0x3FFFFFC) | 2;  // 8-in-32, as BeginVertices writes it
  }
  if (a.indexed) {
    if (!p.index_data) return;
    auto indices = ReadIndices(Guest<uint8_t>(p.index_data), a.count, a.index32, a.index32 ? 2 : 1);
    if (NeedsExpansion(a.type)) indices = Expand(a.type, indices);
    if (const uint32_t reset = RestartIndex(*Guest<D3DDevice>(a.device), a.type);
        reset != UINT32_MAX && SplitAtRestart(a.type, reset, indices)) {
      r.state.topology = uint8_t(a.type == xenos::PrimitiveType::kLineStrip ? xenos::PrimitiveType::kLineList
                                                                             : xenos::PrimitiveType::kTriangleList);
    }
    r.index32 = true;
    r.count = uint32_t(indices.size());
    auto bytes = std::make_shared<std::vector<uint8_t>>(indices.size() * 4);
    std::memcpy(bytes->data(), indices.data(), bytes->size());
    r.index.key = 0;   // private to this draw, uploaded with it
    r.index.size = uint32_t(bytes->size());
    g_captured_bytes += bytes->size();
    r.index.bytes = std::move(bytes);
  }
  Submit(std::move(r));
}

void Capture(PPCContext& ctx, uint8_t* base, const DrawArgs& a) {
  if (!REXCVAR_GET(gpu_draws) || !a.count) return;
  if (cmdbuf::Recording(a.device)) {
    cmdbuf::RecordDraw(a);
    return;
  }
  std::lock_guard lock(g_mutex);
  CaptureTimer timer;
  DrawRecord r;
  if (Build(ctx, base, a, r, Guest<D3DDevice>(a.device)) && r.constants) Submit(std::move(r));
}

void BeginInline(PPCContext& ctx, uint8_t* base, const DrawArgs& a, uint32_t vertex_data, uint32_t index_data) {
  FinishInline();
  if (cmdbuf::Recording(a.device)) {
    cmdbuf::RecordDraw(a);
    return;
  }
  PendingInline& p = t_pending;
  p = PendingInline{};
  p.active = true;
  p.args = a;
  p.vertex_data = vertex_data;
  p.index_data = index_data;
  if (!REXCVAR_GET(gpu_draws) || !a.count || !vertex_data) return;
  std::lock_guard lock(g_mutex);
  CaptureTimer timer;
  p.valid = Build(ctx, base, a, p.record, Guest<D3DDevice>(a.device)) && p.record.constants;
}

}  // namespace

void FinishPending() { FinishInline(); }

void FinishPreparation(Packet& packet) {
  BurstTiming timing(8);
  std::lock_guard lock(g_mutex);
  CaptureTimer timer;
  const auto start = std::chrono::steady_clock::now();
  FlushDrawBatch();
  if (g_prepare_queue) g_prepare_queue->Drain();
  for (auto& record : packet.records) {
    if (auto* draw = std::get_if<DrawRecord>(&record)) {
      FinishDrawState(*draw);
      FinishIndices(*draw);
    }
  }
  const auto finish_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start).count());
  AddCaptureBreakdown(3, finish_ns);
  static uint64_t frames = 0, finish_total = 0;
  finish_total += finish_ns;
  if (++frames == 300) {
    FloatConstantSnapshots::Stats snapshots;
    for (auto& [device, state] : g_devices) {
      const auto count = state.float_snapshots.TakeStats();
      snapshots.requests += count.requests;
      snapshots.reused += count.reused;
    }
    REXGPU_INFO("[gpu] constants/300 frames: {} of {} snapshots reused, {:.3f} MB/frame copied",
                snapshots.reused, snapshots.requests,
                (snapshots.requests - snapshots.reused) * sizeof(FloatConstantBlock) / 300e6);
    if (g_prepare_queue) {
      const auto stats = g_prepare_queue->TakeStats();
      REXGPU_INFO("[gpu] preparation/300 frames: jobs {} queued, {} inline, work {:.3f} ms/frame, "
                  "guest finish {:.3f} ms/frame", stats.queued, stats.inline_jobs,
                  stats.work_ns / 300e6, finish_total / 300e6);
    }
    frames = finish_total = 0;
  }
}

VsSnapshot SnapshotVs(const D3DDevice* device) {
  VsSnapshot s;
  uint32_t code, size;
  if (!LocateVs(device, s.object, s.variant1, code, size)) return s;
  const auto* raw = Guest<uint8_t>(code);
  s.code.assign(raw, raw + size);
  return s;
}

void Replay(PPCContext& ctx, uint8_t* base, const DrawArgs& a, const D3DDevice* device, const VsSnapshot& vs) {
  if (!REXCVAR_GET(gpu_draws) || !a.count || a.inline_data) return;
  std::lock_guard lock(g_mutex);
  CaptureTimer timer;
  DrawRecord r;
  if (Build(ctx, base, a, r, device, &vs) && r.constants) Submit(std::move(r));
}

}  // namespace ae::gpu::draw

namespace draw = ae::gpu::draw;
namespace xenos = rex::graphics::xenos;

// D3DDevice_DrawVertices 0x92121380 (pDevice, PrimitiveType, StartVertex, VertexCount); the packet writes
// StartVertex into VGT_INDX_OFFSET, not the shadow.
REX_HOOK_RAW(sub_92121380) {
  draw::FinishPending();
  draw::DrawArgs a;
  a.device = ctx.r3.u32;
  a.type = xenos::PrimitiveType(ctx.r4.u32);
  a.start = ctx.r5.u32;
  a.base_vertex = int32_t(ctx.r5.u32);
  a.count = ctx.r6.u32;
  ae::gpu::cmdbuf::NotePending(a.device);
  __imp__sub_92121380(ctx, base);
  draw::Capture(ctx, base, a);
}

// D3DDevice_DrawIndexedVertices 0x92121798 (pDevice, PrimitiveType, BaseVertexIndex, StartIndex, IndexCount).
REX_HOOK_RAW(sub_92121798) {
  draw::FinishPending();
  draw::DrawArgs a;
  a.device = ctx.r3.u32;
  a.type = xenos::PrimitiveType(ctx.r4.u32);
  a.base_vertex = int32_t(ctx.r5.u32);
  a.start = ctx.r6.u32;
  a.count = ctx.r7.u32;
  a.indexed = true;
  ae::gpu::cmdbuf::NotePending(a.device);
  __imp__sub_92121798(ctx, base);
  draw::Capture(ctx, base, a);
}

// D3DDevice_BeginVertices 0x92120898 (pDevice, PrimitiveType, VertexCount, VertexStreamZeroStride) returns the data
// pointer; also reached from DrawVerticesUP.
REX_HOOK_RAW(sub_92120898) {
  draw::DrawArgs a;
  a.device = ctx.r3.u32;
  a.type = xenos::PrimitiveType(ctx.r4.u32);
  a.count = ctx.r5.u32;
  a.stride = ctx.r6.u32;
  a.vertex_bytes = (a.count * a.stride) & ~3u;
  a.inline_data = true;
  __imp__sub_92120898(ctx, base);
  draw::BeginInline(ctx, base, a, ctx.r3.u32, 0);
}

// D3DDevice_BeginIndexedVertices 0x92120DA0 (pDevice, PrimitiveType, BaseVertexIndex, NumVertices, IndexCount,
// IndexDataFormat, VertexStreamZeroStride, ppIndexData, ppVertexData); the last argument is the stack slot r1+0x54.
REX_HOOK_RAW(sub_92120DA0) {
  draw::DrawArgs a;
  a.device = ctx.r3.u32;
  a.type = xenos::PrimitiveType(ctx.r4.u32);
  a.base_vertex = int32_t(ctx.r5.u32);
  a.count = ctx.r7.u32;
  a.index32 = (ctx.r8.u32 & 4) != 0;
  a.stride = ctx.r9.u32;
  a.vertex_bytes = (ctx.r6.u32 * a.stride) & ~3u;
  a.indexed = true;
  a.inline_data = true;
  const uint32_t pp_index = ctx.r10.u32;
  const uint32_t pp_vertex = *REX_KERNEL_MEMORY()->TranslateVirtual<rex::be<uint32_t>*>(ctx.r1.u32 + 0x54);
  __imp__sub_92120DA0(ctx, base);
  const uint32_t index_data = pp_index ? uint32_t(*REX_KERNEL_MEMORY()->TranslateVirtual<rex::be<uint32_t>*>(pp_index)) : 0;
  const uint32_t vertex_data =
      pp_vertex ? uint32_t(*REX_KERNEL_MEMORY()->TranslateVirtual<rex::be<uint32_t>*>(pp_vertex)) : 0;
  draw::BeginInline(ctx, base, a, vertex_data, index_data);
}

// D3DDevice_EndIndexedVertices 0x92120888, which EndVertices shares; the game may also inline it, so every later
// commit point finishes a pending record too.
REX_HOOK_RAW(sub_92120888) {
  __imp__sub_92120888(ctx, base);
  draw::FinishPending();
}

// D3DDevice_DrawVerticesUP 0x92120D58 and DrawIndexedVerticesUP 0x921212E0 run Begin, copy, and End inline.
REX_HOOK_RAW(sub_92120D58) {
  draw::FinishPending();
  __imp__sub_92120D58(ctx, base);
  draw::FinishPending();
}

REX_HOOK_RAW(sub_921212E0) {
  draw::FinishPending();
  __imp__sub_921212E0(ctx, base);
  draw::FinishPending();
}
