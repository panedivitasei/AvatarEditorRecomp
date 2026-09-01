// rexvideonative, render-thread offload queue (see render_queue.h).
// GPL-3.0, see LICENSE in this directory.

#include "render_queue.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>  // guest-thread cycle time attribution

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>

#include "renderer.h"
#include "video_native_internal.h"

REXCVAR_DEFINE_BOOL(native_video_render_thread, false, "GPU",
                    "Run the native renderer on a dedicated worker thread "
                    "(guest thread enqueues), restoring the ring path's "
                    "two-core split on weak-CPU machines. Latched at boot.");
REXCVAR_DEFINE_INT32(native_video_rq_depth, 2, "GPU",
                     "Frames the guest may run ahead of the render worker "
                     "(1-3). 1 = tightest pacing (inline-like present "
                     "timing). 2 lets single-frame guest-thread spikes "
                     "amortize instead of dipping the frame rate. Costs one "
                     "frame of latency per step.");

namespace rex::videonative::rq {

thread_local Win t_win[kMaxWins];

namespace {

thread_local bool t_onWorker = false;

enum Op : uint16_t {
  kSetViewport,
  kSetScissor,
  kSetTexture,
  kSetStream,
  kSetIndices,
  kConstFHost,
  kPatchFHost,
  kMergeFHost,
  kClear,
  kRegisterSurface,
  kSetRT,
  kSetDepth,
  kTiling,
  kNoteRtt,
  kInvalidateTex,
  kResolve,
  kResolveDepth,
  kSwap,
  kEndFrame,
  kGamma,
  kFlushResolveWriteback,
  kPushState,
  kPopState,
  kSeedReplayF,
  kReplayDrawState,
  kReplayBool,
  kReplayFetch,
  kReplayPersist,
  kBlendDirect,
  kLiveDirty,
  kDrawV,
  kDrawIV,
  kDrawVUP,
  kDrawIVUP,
  kStop,
};

struct Rec {
  uint16_t op = 0;
  uint16_t flag = 0;
  uint32_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0;
  float fa = 0.0f, fb = 0.0f;
  uint64_t blob1 = 0, blob2 = 0, blob3 = 0;
  uint32_t len1 = 0, len2 = 0, len3 = 0;
  const void* p0 = nullptr;
  const void* p1 = nullptr;
  uint64_t arena_release = 0;  // arena head after this record's blobs
};

constexpr uint32_t kRecCount = 1u << 16;  // 64K records (power of two)
constexpr uint64_t kArenaSize = 96ull << 20;  // 96MB blob arena

std::vector<Rec> g_recs;
std::vector<uint8_t> g_arena;
std::atomic<uint64_t> g_recHead{0};   // producer (published records)
std::atomic<uint64_t> g_recTail{0};   // consumer
std::atomic<uint64_t> g_arenaHead{0};  // producer bytes (monotonic)
std::atomic<uint64_t> g_arenaTail{0};  // consumer release point
std::atomic<uint64_t> g_framesEnq{0};
std::atomic<uint64_t> g_framesDone{0};
// Resolve write-back flush completion (ticket/counter pair, see the
// kFlushResolveWriteback dispatch note on waiter lifetime).
std::atomic<uint64_t> g_wbFlushTickets{0};
std::atomic<uint64_t> g_wbFlushDone{0};
std::atomic<bool> g_running{false};
std::thread g_worker;
std::mutex g_wakeMutex;
std::condition_variable g_wake;

// Guest (enqueue) thread handle for cycle-time attribution.
// Captured at the first EnqEndFrameAndPresent; read by the worker.
std::atomic<void*> g_guestThreadHandle{nullptr};
// Record-ring producer (the guest render thread).
std::atomic<uint32_t> g_guestThreadId{0};

// Guest-side pending UP override (single guest thread, plain vars).
const void* g_pendUPv = nullptr;
const void* g_pendUPi = nullptr;
// Guest-side mirror of bound VB/IB object handles (g_state itself is
// worker-owned once the queue runs): draw enqueues snapshot the object
// blocks (32B each: flags + data ptr + size).
uint32_t g_boundStreams[16] = {};
uint32_t g_boundIb = 0;
// Push/PopState bracket CB replays and the worker restores
// g_state.streams/index_buffer_object at PopState, so the mirror must restore
// in lockstep; otherwise a post-Run draw reusing pre-Run bindings snapshots
// the CB's objects and its real reads fall through to live memory.
struct BoundState {
  uint32_t streams[16];
  uint32_t ib;
};
std::vector<BoundState> g_boundStack;

// Big-VB mutation classifier: buffers larger than kSnapVbMax stay deferred
// (the worker reads them live, up to a frame late), which is only safe for
// content that never changes. Fingerprint each big VB once per frame (sampled
// head/mid/tail 16KB, the same windows BindStreamData's large-buffer hash
// reads) and track which buffers mutate.
constexpr uint32_t kFpWin = 16 * 1024;
const uint8_t* GuestHost(uint32_t addr);
struct BigVbInfo {
  uint64_t fp = 0;
  uint64_t fp_frame = 0;   // frame the fingerprint was last taken
  uint32_t size = 0;
  bool dynamic_vb = false;
};
std::unordered_map<uint32_t, BigVbInfo> g_bigVb;  // by masked data base
uint64_t g_rqFrame = 0;    // guest-side frame counter (EnqEndFrame)

uint64_t FingerprintBigVb(const uint8_t* host, uint32_t size) {
  XXH3_state_t* st = XXH3_createState();
  XXH3_64bits_reset(st);
  XXH3_64bits_update(st, host, kFpWin);
  XXH3_64bits_update(st, host + (size / 2 & ~15u), kFpWin);
  XXH3_64bits_update(st, host + size - kFpWin, kFpWin);
  const uint64_t fp = XXH3_64bits_digest(st);
  XXH3_freeState(st);
  return fp;
}

void ClassifyBigVb(uint32_t data_base, uint32_t size, const uint8_t* host) {
  BigVbInfo& info = g_bigVb[data_base];
  if (info.fp_frame != g_rqFrame || info.size != size) {
    const uint64_t fp = FingerprintBigVb(host, size);
    if (info.fp_frame && (fp != info.fp || info.size != size)) {
      info.dynamic_vb = true;
    }
    info.fp = fp;
    info.fp_frame = g_rqFrame;
    info.size = size;
  }
}

const uint8_t* GuestHost(uint32_t addr) {
  return rex::system::kernel_memory()->TranslateVirtual<const uint8_t*>(addr);
}

uint8_t* ArenaPtr(uint64_t off) { return g_arena.data() + (off % kArenaSize); }

// Contiguous arena allocation (skip the wrap remainder); blocks on the
// worker when full.
uint64_t ArenaAlloc(uint32_t len) {
  uint64_t head = g_arenaHead.load(std::memory_order_relaxed);
  const uint64_t rem = kArenaSize - (head % kArenaSize);
  if (len > rem) head += rem;  // skip to the wrap point
  uint32_t spins = 0;
  while (head + len - g_arenaTail.load(std::memory_order_acquire) >
         kArenaSize) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    if (++spins == 20000) {  // ~2s
      REXGPU_WARN("videonative: [rthread] guest blocked in arena alloc ~2s "
                  "(len={} head={} tail={})",
                  len, head, g_arenaTail.load());
    }
  }
  g_arenaHead.store(head + len, std::memory_order_relaxed);
  return head;
}

uint64_t Blob(const void* src, uint32_t len, uint64_t* out_off) {
  if (!src || !len) {
    *out_off = 0;
    return 0;
  }
  const uint64_t off = ArenaAlloc(len);
  std::memcpy(ArenaPtr(off), src, len);
  *out_off = off;
  return len;
}

Rec& Begin() {
  const uint64_t head = g_recHead.load(std::memory_order_relaxed);
  uint32_t spins = 0;
  while (head - g_recTail.load(std::memory_order_acquire) >= kRecCount) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    if (++spins == 20000) {
      REXGPU_WARN("videonative: [rthread] guest blocked in record ring ~2s "
                  "(head={} tail={})",
                  head, g_recTail.load());
    }
  }
  Rec& r = g_recs[head % kRecCount];
  r = Rec{};
  return r;
}

void Commit() {
  // The record's arena_release marks everything its blobs consumed.
  Rec& r = g_recs[g_recHead.load(std::memory_order_relaxed) % kRecCount];
  r.arena_release = g_arenaHead.load(std::memory_order_relaxed);
  g_recHead.fetch_add(1, std::memory_order_release);
  g_wake.notify_one();
}

// Object-block snapshot pack: [addr,32B block] entries for the bound IB +
// nonzero VB streams; the worker installs one 32B window per entry so the
// unchanged renderer reads call-time object state (data ptr, flags, size).
constexpr uint32_t kObjSnapBytes = 32;
// Call-time data capture limits: stream VBs at/below this size are copied at
// enqueue (pooled/streamed VBs mutate under the worker's deferred read);
// larger buffers (big static meshes) stay deferred like the ring path. Index
// ranges are always small.
constexpr uint32_t kSnapVbMax = 64 * 1024;
constexpr uint32_t kSnapIbMax = 256 * 1024;

// Variable-entry snapshot pack: [u32 addr][u32 len][len bytes]..., the
// worker installs one override window per entry; renderer code reads
// call-time bytes through the unchanged GuestPtr/LoadGuestU32 helpers.
struct PackBuilder {
  // Two-pass: entries recorded first, one contiguous ArenaAlloc at Finish
  // (per-entry allocs could straddle the arena wrap and orphan the tail).
  struct Entry { uint32_t addr; uint32_t bytes; const uint8_t* host; };
  Entry entries[24];
  uint32_t count = 0;
  uint32_t total = 0;
  uint64_t off = 0;
  uint32_t len = 0;
  bool dropped = false;
  // host: the validated view to copy from at Finish. The default resolves
  // the virtual view, correct for the trusted object/data spans that
  // always copied from it; untrusted (shadow) spans pass their probed view.
  void Add(uint32_t addr, uint32_t bytes, const uint8_t* host = nullptr) {
    if (!addr || !bytes) return;
    if (count >= 24) {
      dropped = true;
      return;
    }
    entries[count++] = {addr, bytes, host ? host : GuestHost(addr)};
    total += 8 + bytes;
  }
  void Finish() {
    if (!count) return;
    off = ArenaAlloc(total);
    uint8_t* dst = ArenaPtr(off);
    uint32_t o = 0;
    for (uint32_t i = 0; i < count; i++) {
      std::memcpy(dst + o, &entries[i].addr, 4);
      std::memcpy(dst + o + 4, &entries[i].bytes, 4);
      std::memcpy(dst + o + 8, entries[i].host, entries[i].bytes);
      o += 8 + entries[i].bytes;
    }
    len = total;
  }
};

// Data-span capture policy shared by the VB-object and fetch-shadow paths:
// small spans always copy at enqueue; big spans stay deferred (worker reads
// live, ring-equivalent) unless the mutation classifier has seen the base
// change, in which case deferral tears the read and the span is copied whole
// up to kSnapDynVbMax.
constexpr uint32_t kSnapDynVbMax = 1024 * 1024;
void AddVbDataSpan(PackBuilder& pk, uint32_t base, uint32_t size,
                   bool untrusted = false) {
  if (!base || !size) return;
  // Shadow-origin spans are untrusted (boot-time junk pairs decode as
  // plausible base|3 dwords): resolve through the probing memoized reader and
  // never touch an unvalidated page (the runtime VEH makes SEH guards
  // unreliable, so a junk-span fault is fatal).
  const uint8_t* host;
  if (untrusted) {
    host = renderer::GuestDataPtrProbe(base, size);
    if (!host) return;
  } else {
    host = GuestHost(base);
  }
  if (size <= kSnapVbMax) {
    pk.Add(base, size, host);
    return;
  }
  ClassifyBigVb(base, size, host);
  if (g_bigVb[base].dynamic_vb && size <= kSnapDynVbMax) {
    pk.Add(base, size, host);
  }
}

uint64_t SnapDrawInputs(uint32_t device, bool indexed, uint32_t index_count,
                        uint32_t start_index, uint32_t* out_len) {
  PackBuilder pk;
  uint32_t data_bases[24];
  uint32_t data_base_count = 0;
  if (g_boundIb) {
    pk.Add(g_boundIb, kObjSnapBytes);
    if (indexed) {
      // Index range: data ptr + format from the live object at call time.
      const uint8_t* obj = GuestHost(g_boundIb);
      uint32_t dw0, data_ptr;
      std::memcpy(&dw0, obj, 4);
      std::memcpy(&data_ptr, obj + 24, 4);
      dw0 = __builtin_bswap32(dw0);
      data_ptr = __builtin_bswap32(data_ptr);
      const uint32_t isz = (dw0 & 0x80000000u) ? 4 : 2;
      const uint32_t bytes = index_count * isz;
      if (data_ptr && bytes && bytes <= kSnapIbMax) {
        pk.Add(data_ptr + start_index * isz, bytes);
      }
    }
  }
  for (uint32_t i = 0; i < 16; i++) {
    const uint32_t vb = g_boundStreams[i];
    if (!vb) continue;
    pk.Add(vb, kObjSnapBytes);
    const uint8_t* obj = GuestHost(vb);
    uint32_t data_ptr, total;
    std::memcpy(&data_ptr, obj + 24, 4);
    std::memcpy(&total, obj + 28, 4);
    data_ptr = __builtin_bswap32(data_ptr);
    total = __builtin_bswap32(total);
    // XDK D3DVertexBuffer flavor (SetupDraw's decode, renderer.cpp): +24 =
    // data | 3 (pre-formatted vfetch type bits), +28 = (size & 0x3FFFFFC) |
    // endian/usage flags. Without the masks the size gate reads flag bits as
    // gigabytes and the captured window's base sits 3 bytes above the data.
    if ((data_ptr & 3u) == 3u) {
      data_ptr &= ~3u;
      total &= 0x3FFFFFCu;
    }
    AddVbDataSpan(pk, data_ptr, total);
    if (data_ptr && data_base_count < 24) {
      data_bases[data_base_count++] = data_ptr;
    }
  }
  // Fetch-shadow streams: SetupDraw's primary stream source in mirror mode is
  // the device shadow pair (dev+1152, dword 190-2*s), not the bound VB
  // objects. The bound-handle mirror never sees those bases, so without this
  // capture the worker reads them live a frame late.
  if (device) {
    const uint8_t* dev = GuestHost(device);
    for (uint32_t s = 0; s < 16; s++) {
      const uint32_t di = 1152 + (190 - 2 * s) * 4;
      uint32_t g0, g1;
      std::memcpy(&g0, dev + di, 4);
      std::memcpy(&g1, dev + di + 4, 4);
      g0 = __builtin_bswap32(g0);
      g1 = __builtin_bswap32(g1);
      if ((g0 & 3u) != 3u) continue;
      const uint32_t base = g0 & ~3u;
      const uint32_t size = g1 & 0x3FFFFFCu;
      if (!base || !size) continue;
      bool dup = false;
      for (uint32_t i = 0; i < data_base_count; i++) {
        if (data_bases[i] == base) { dup = true; break; }
      }
      if (dup) continue;
      AddVbDataSpan(pk, base, size, /*untrusted=*/true);
      if (data_base_count < 24) data_bases[data_base_count++] = base;
    }
  }
  pk.Finish();
  if (pk.dropped) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      REXGPU_WARN(
          "videonative: [rthread] draw snapshot pack entry cap hit, some "
          "bound objects stay deferred this draw");
    }
  }
  *out_len = pk.len;
  return pk.off;
}

void InstallPackWins(uint64_t blob, uint32_t len, int first_slot) {
  const uint8_t* base = ArenaPtr(blob);
  int slot = first_slot;
  uint32_t o = 0;
  while (o + 8 <= len && slot < kMaxWins) {
    uint32_t addr, sz;
    std::memcpy(&addr, base + o, 4);
    std::memcpy(&sz, base + o + 4, 4);
    if (o + 8 + sz > len) break;
    t_win[slot].host = base + o + 8;
    t_win[slot].base = addr;
    t_win[slot].size = sz;
    slot++;
    o += 8 + sz;
  }
}

// Device-block snapshot (kWinSize bytes) into the arena.
uint64_t SnapDevice(uint32_t device, uint64_t* out_off) {
  if (!device) {
    *out_off = 0;
    return 0;
  }
  return Blob(GuestHost(device), kWinSize, out_off);
}

void InstallWin(int slot, uint64_t blob_off, uint32_t len, uint32_t base) {
  if (len) {
    t_win[slot].host = ArenaPtr(blob_off);
    t_win[slot].base = base;
    t_win[slot].size = len;
  } else {
    t_win[slot] = Win{};
  }
}

void ClearWins() {
  for (int i = 0; i < kMaxWins; i++) t_win[i] = Win{};
  detail::ClearResolvedShaderOverride();
}

void Execute(const Rec& r) {
  using namespace rex::videonative::renderer;
  switch (r.op) {
    case kSetViewport:
      SetViewport(r.a, r.b, r.c, r.d, r.fa, r.fb);
      break;
    case kSetScissor:
      SetScissor(int32_t(r.a), int32_t(r.b), int32_t(r.c), int32_t(r.d));
      break;
    case kSetTexture:
      SetTextureWithSnapshot(r.a, r.b,
                             reinterpret_cast<const uint32_t*>(ArenaPtr(r.blob1)));
      break;
    case kSetStream:
      SetStream(r.a, r.b, r.c, r.d);
      break;
    case kSetIndices:
      SetIndices(r.a);
      break;
    case kConstFHost:
      SetShaderConstantsFHost(r.flag != 0, r.a,
                              reinterpret_cast<const uint32_t*>(ArenaPtr(r.blob1)),
                              r.b);
      break;
    case kPatchFHost:
      PatchShaderFloatDwordsHost(
          r.flag != 0, r.a,
          reinterpret_cast<const uint32_t*>(ArenaPtr(r.blob1)), r.b);
      break;
    case kMergeFHost:
      MergeShaderFloatDwordHost(r.flag != 0, r.a, r.b, r.c);
      break;
    case kClear:
      Clear(0, r.a, r.b, r.fa,
            r.len1 ? reinterpret_cast<const int32_t*>(ArenaPtr(r.blob1))
                   : nullptr,
            r.c, r.d);
      break;
    case kRegisterSurface:
      RegisterSurface(r.a, r.b, r.c, r.d);
      break;
    case kSetRT:
      SetRenderTargetSurface(r.a, r.b);
      break;
    case kSetDepth:
      SetDepthSurface(r.a);
      break;
    case kTiling:
      SetTilingExtent(r.a, r.b);
      break;
    case kNoteRtt:
      NoteRttBegin(r.a, r.b, r.c, r.d, r.e, r.f, r.g, r.h, r.flag);
      break;
    case kInvalidateTex:
      InvalidateTextureByHeader(r.a);
      break;
    case kResolve:
      // win: the dest texture header captured at enqueue (recycled-header
      // race, the same class as the SetTexture bind snapshot).
      InstallWin(0, r.blob1, r.len1, r.a);
      ResolveToTexture(r.a, r.b, r.flag != 0, int32_t(r.c), int32_t(r.d),
                       int32_t(r.e), int32_t(r.f), int32_t(r.g),
                       int32_t(r.h));
      ClearWins();
      break;
    case kResolveDepth:
      InstallWin(0, r.blob1, r.len1, r.a);
      ResolveDepthToTexture(r.a, r.flag != 0, int32_t(r.c), int32_t(r.d),
                            int32_t(r.e), int32_t(r.f), int32_t(r.g),
                            int32_t(r.h));
      ClearWins();
      break;
    case kSwap:
      SwapFrontbuffer(r.a);
      break;
    case kEndFrame:
      EndFrameAndPresent();
      g_framesDone.fetch_add(1, std::memory_order_release);
      break;
    case kFlushResolveWriteback:
      FlushResolveWritebacksInline();
      // Monotonic completion counter, not a caller pointer: a timed-out
      // waiter's stack flag dangles by the time a busy worker gets here
      // (a stale-pointer crash class).
      g_wbFlushDone.fetch_add(1, std::memory_order_release);
      break;
    case kGamma:
      UpdateGammaRamp(reinterpret_cast<const uint16_t*>(ArenaPtr(r.blob1)));
      break;
    case kPushState:
      PushState();
      break;
    case kPopState:
      PopState();
      break;
    case kSeedReplayF:
      InstallWin(0, r.blob1, r.len1, r.a);
      InstallWin(1, r.blob2, r.len2, r.b);
      SeedReplayFloatConstants(r.a, r.b);
      ClearWins();
      break;
    case kReplayDrawState:
      SetReplayDrawState(
          r.len1 ? reinterpret_cast<const uint32_t*>(ArenaPtr(r.blob1))
                 : nullptr,
          r.a);
      break;
    case kReplayBool:
      SetReplayBoolConstants(
          r.len1 ? reinterpret_cast<const uint32_t*>(ArenaPtr(r.blob1))
                 : nullptr);
      break;
    case kReplayFetch:
      SetReplayFetchConstants(
          reinterpret_cast<const uint32_t*>(ArenaPtr(r.blob1)), r.a);
      break;
    case kReplayPersist:
      ApplyReplayStatePersistent(
          r.len1 ? reinterpret_cast<const uint32_t*>(ArenaPtr(r.blob1))
                 : nullptr,
          r.a);
      break;
    case kBlendDirect:
      ApplyBlendControlDirect(r.a, r.b);
      break;
    case kLiveDirty: {
      const uint32_t vals[5] = {r.b, r.c, r.d, r.e, r.f};
      ApplyLiveStateDirtyHost(r.a, vals);
      break;
    }
    case kDrawV:
      InstallWin(0, r.blob1, r.len1, r.a);
      InstallPackWins(r.blob3, r.len3, 2);
      detail::OverrideResolvedShaders(
          reinterpret_cast<const ResolvedShader*>(r.p0),
          reinterpret_cast<const ResolvedShader*>(r.p1), r.flag != 0);
      DrawVertices(r.a, r.b, r.c, r.d);
      ClearWins();
      break;
    case kDrawIV:
      InstallWin(0, r.blob1, r.len1, r.a);
      InstallPackWins(r.blob3, r.len3, 2);
      detail::OverrideResolvedShaders(
          reinterpret_cast<const ResolvedShader*>(r.p0),
          reinterpret_cast<const ResolvedShader*>(r.p1), r.flag != 0);
      DrawIndexedVertices(r.a, r.b, r.c, r.d, r.e);
      ClearWins();
      break;
    case kDrawVUP:
      InstallWin(0, r.blob1, r.len1, r.a);
      InstallPackWins(r.blob3, r.len3, 2);
      detail::OverrideResolvedShaders(
          reinterpret_cast<const ResolvedShader*>(r.p0),
          reinterpret_cast<const ResolvedShader*>(r.p1), r.flag != 0);
      SetUPDataOverride(r.len2 ? ArenaPtr(r.blob2) : nullptr, nullptr);
      DrawVerticesUP(r.a, r.b, r.c, r.d, r.e);
      SetUPDataOverride(nullptr, nullptr);
      ClearWins();
      break;
    case kDrawIVUP: {
      InstallWin(0, r.blob1, r.len1, r.a);
      InstallPackWins(r.blob3, r.len3, 2);
      detail::OverrideResolvedShaders(
          reinterpret_cast<const ResolvedShader*>(r.p0),
          reinterpret_cast<const ResolvedShader*>(r.p1), r.flag != 0);
      // blob2 = vertex bytes then index bytes (len2 = vertex byte count,
      // index bytes follow; g = index byte count).
      const uint8_t* vtx = r.len2 ? ArenaPtr(r.blob2) : nullptr;
      const uint8_t* idx = (r.len2 && r.g) ? ArenaPtr(r.blob2) + r.len2 : nullptr;
      SetUPDataOverride(vtx, idx);
      DrawIndexedVerticesUP(r.a, r.b, r.c, r.d, r.e, r.f, 1, r.h,
                            uint32_t(r.fb));
      SetUPDataOverride(nullptr, nullptr);
      ClearWins();
      break;
    }
    default:
      break;
  }
}

void WorkerMain() {
  t_onWorker = true;
  for (;;) {
    const uint64_t tail = g_recTail.load(std::memory_order_relaxed);
    if (tail == g_recHead.load(std::memory_order_acquire)) {
      std::unique_lock<std::mutex> lk(g_wakeMutex);
      g_wake.wait_for(lk, std::chrono::milliseconds(1));
      continue;
    }
    Rec& r = g_recs[tail % kRecCount];
    if (r.op == kStop) {
      g_recTail.fetch_add(1, std::memory_order_release);
      break;
    }
    Execute(r);
    g_arenaTail.store(r.arena_release, std::memory_order_release);
    g_recTail.fetch_add(1, std::memory_order_release);
  }
  t_onWorker = false;
}

}  // namespace

bool Active() {
  return g_running.load(std::memory_order_relaxed) && !t_onWorker;
}

bool Running() { return g_running.load(std::memory_order_relaxed); }

// The record ring is single-producer: only the guest render thread (the one
// presenting) may Begin()/Commit(). Other guest threads (XUI's image worker
// locks textures too) must use the mutex-protected side queues
// (renderer::QueueGuestTextureInvalidate) instead; a second producer here
// clobbers or loses records.
bool OnProducerThread() {
  const uint32_t tid = g_guestThreadId.load(std::memory_order_acquire);
  return tid != 0 && tid == GetCurrentThreadId();
}

void Start() {
  if (g_running.load()) return;
  if (!REXCVAR_GET(native_video_render_thread)) return;
  g_recs.resize(kRecCount);
  g_arena.resize(kArenaSize);
  g_running.store(true);
  g_worker = std::thread(WorkerMain);
}

void StopAndJoin() {
  if (!g_running.load()) return;
  {
    Rec& r = Begin();
    r.op = kStop;
    Commit();
  }
  g_worker.join();
  g_running.store(false);
}

// --- enqueue implementations ------------------------------------------------

void EnqSetViewport(uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                    float min_z, float max_z) {
  Rec& r = Begin();
  r.op = kSetViewport; r.a = x; r.b = y; r.c = w; r.d = h;
  r.fa = min_z; r.fb = max_z;
  Commit();
}

void EnqSetScissor(int32_t l, int32_t t, int32_t rr, int32_t b) {
  Rec& r = Begin();
  r.op = kSetScissor;
  r.a = uint32_t(l); r.b = uint32_t(t); r.c = uint32_t(rr); r.d = uint32_t(b);
  Commit();
}

void EnqSetTexture(uint32_t sampler, uint32_t header, const uint32_t dw[6]) {
  Rec& r = Begin();
  r.op = kSetTexture; r.a = sampler; r.b = header;
  r.len1 = uint32_t(Blob(dw, 24, &r.blob1));
  Commit();
}

void EnqSetStream(uint32_t stream, uint32_t vb, uint32_t off, uint32_t stride) {
  if (stream < 16) g_boundStreams[stream] = vb;
  Rec& r = Begin();
  r.op = kSetStream; r.a = stream; r.b = vb; r.c = off; r.d = stride;
  Commit();
}

void EnqSetIndices(uint32_t ib) {
  g_boundIb = ib;
  Rec& r = Begin();
  r.op = kSetIndices; r.a = ib;
  Commit();
}

void EnqSetShaderConstantsF(bool pixel, uint32_t start_reg,
                            uint32_t guest_data, uint32_t vec4_count) {
  if (start_reg >= 256) return;
  vec4_count = vec4_count > 256 - start_reg ? 256 - start_reg : vec4_count;
  // Call-time capture + byteswap (PM4-equivalent semantics), host-order blob.
  const uint32_t* src = reinterpret_cast<const uint32_t*>(GuestHost(guest_data));
  uint32_t tmp[256 * 4];
  for (uint32_t i = 0; i < vec4_count * 4; i++) {
    tmp[i] = __builtin_bswap32(src[i]);
  }
  Rec& r = Begin();
  r.op = kConstFHost; r.flag = pixel ? 1 : 0; r.a = start_reg; r.b = vec4_count;
  r.len1 = uint32_t(Blob(tmp, vec4_count * 16, &r.blob1));
  Commit();
}

void EnqSetShaderConstantsFHost(bool pixel, uint32_t start_reg,
                                const uint32_t* host_dwords,
                                uint32_t vec4_count) {
  if (start_reg >= 256) return;
  vec4_count = vec4_count > 256 - start_reg ? 256 - start_reg : vec4_count;
  Rec& r = Begin();
  r.op = kConstFHost; r.flag = pixel ? 1 : 0; r.a = start_reg; r.b = vec4_count;
  r.len1 = uint32_t(Blob(host_dwords, vec4_count * 16, &r.blob1));
  Commit();
}

void EnqPatchShaderFloatDwordsHost(bool pixel, uint32_t dword_index,
                                   const uint32_t* host_dwords,
                                   uint32_t count) {
  Rec& r = Begin();
  r.op = kPatchFHost; r.flag = pixel ? 1 : 0; r.a = dword_index; r.b = count;
  r.len1 = uint32_t(Blob(host_dwords, count * 4, &r.blob1));
  Commit();
}

void EnqMergeShaderFloatDwordHost(bool pixel, uint32_t dword_index,
                                  uint32_t mask, uint32_t orv) {
  Rec& r = Begin();
  r.op = kMergeFHost; r.flag = pixel ? 1 : 0;
  r.a = dword_index; r.b = mask; r.c = orv;
  Commit();
}

void EnqClear(uint32_t flags, uint32_t color, float z, const int32_t* rects,
              uint32_t rect_count, uint32_t stencil) {
  Rec& r = Begin();
  r.op = kClear; r.a = flags; r.b = color; r.fa = z; r.c = rect_count;
  r.d = stencil;
  if (rects && rect_count) {
    r.len1 = uint32_t(Blob(rects, rect_count * 16, &r.blob1));
  }
  Commit();
}

void EnqRegisterSurface(uint32_t obj, uint32_t w, uint32_t h, uint32_t fmt) {
  Rec& r = Begin();
  r.op = kRegisterSurface; r.a = obj; r.b = w; r.c = h; r.d = fmt;
  Commit();
}

void EnqSetRenderTargetSurface(uint32_t index, uint32_t obj) {
  Rec& r = Begin();
  r.op = kSetRT; r.a = index; r.b = obj;
  Commit();
}

void EnqSetDepthSurface(uint32_t obj) {
  Rec& r = Begin();
  r.op = kSetDepth; r.a = obj;
  Commit();
}

void EnqSetTilingExtent(uint32_t w, uint32_t h) {
  Rec& r = Begin();
  r.op = kTiling; r.a = w; r.b = h;
  Commit();
}

void EnqInvalidateTexture(uint32_t header_addr) {
  Rec& r = Begin();
  r.op = kInvalidateTex; r.a = header_addr;
  Commit();
}

void EnqNoteRtt(uint32_t texbase, uint32_t w, uint32_t h, uint32_t color_surf,
                uint32_t depth_surf, uint32_t dest_tex, uint32_t format,
                uint32_t msaa, uint32_t tiling) {
  Rec& r = Begin();
  r.op = kNoteRtt; r.a = texbase; r.b = w; r.c = h; r.d = color_surf;
  r.e = depth_surf; r.f = dest_tex; r.g = format; r.h = msaa;
  r.flag = uint16_t(tiling > 0xFFFFu ? 0xFFFFu : tiling);
  Commit();
}

void EnqResolveToTexture(uint32_t dest, uint32_t source, bool has_rect,
                         int32_t l, int32_t t, int32_t rr, int32_t b,
                         int32_t dx, int32_t dy) {
  Rec& r = Begin();
  r.op = kResolve; r.a = dest; r.b = source; r.flag = has_rect ? 1 : 0;
  if (dest) r.len1 = uint32_t(Blob(GuestHost(dest), 24, &r.blob1));
  r.c = uint32_t(l); r.d = uint32_t(t); r.e = uint32_t(rr); r.f = uint32_t(b);
  r.g = uint32_t(dx); r.h = uint32_t(dy);
  Commit();
}

void EnqResolveDepthToTexture(uint32_t dest, bool has_rect, int32_t l,
                              int32_t t, int32_t rr, int32_t b, int32_t dx,
                              int32_t dy) {
  Rec& r = Begin();
  r.op = kResolveDepth; r.a = dest; r.flag = has_rect ? 1 : 0;
  if (dest) r.len1 = uint32_t(Blob(GuestHost(dest), 24, &r.blob1));
  r.c = uint32_t(l); r.d = uint32_t(t); r.e = uint32_t(rr); r.f = uint32_t(b);
  r.g = uint32_t(dx); r.h = uint32_t(dy);
  Commit();
}

void EnqSwapFrontbuffer(uint32_t header) {
  Rec& r = Begin();
  r.op = kSwap; r.a = header;
  Commit();
}

void EnqEndFrameAndPresent() {
  if (!g_guestThreadHandle.load(std::memory_order_relaxed)) {
    g_guestThreadId.store(GetCurrentThreadId(), std::memory_order_release);
    g_guestThreadHandle.store(
        OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE,
                   GetCurrentThreadId()),
        std::memory_order_release);
  }
  {
    Rec& r = Begin();
    r.op = kEndFrame;
    Commit();
  }
  g_rqFrame++;
  const uint64_t enq = g_framesEnq.fetch_add(1, std::memory_order_relaxed) + 1;
  // Back-pressure: how far the guest may run ahead of the worker. Depth 2
  // lets single-frame guest spikes amortize. Warns if parked abnormally
  // long.
  const int32_t depth_cv = REXCVAR_GET(native_video_rq_depth);
  const uint64_t depth =
      uint64_t(depth_cv < 1 ? 1 : (depth_cv > 3 ? 3 : depth_cv));
  uint32_t spins = 0;
  while (enq - g_framesDone.load(std::memory_order_acquire) > depth) {
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    if (++spins == 10000) {  // ~2s
      REXGPU_WARN("videonative: [rthread] guest blocked in frame "
                  "back-pressure ~2s (enq={} done={})",
                  enq, g_framesDone.load());
    }
  }
}

void EnqUpdateGammaRamp(const uint16_t* rgb768) {
  Rec& r = Begin();
  r.op = kGamma;
  r.len1 = uint32_t(Blob(rgb768, 1536, &r.blob1));
  Commit();
}

uint64_t EnqFlushResolveWriteback() {
  const uint64_t ticket =
      g_wbFlushTickets.fetch_add(1, std::memory_order_acq_rel) + 1;
  Rec& r = Begin();
  r.op = kFlushResolveWriteback;
  Commit();
  return ticket;
}

uint64_t WbFlushDone() {
  return g_wbFlushDone.load(std::memory_order_acquire);
}

void DrainForGuestRewrite() {
  if (!g_running.load(std::memory_order_relaxed) || t_onWorker) return;
  const uint64_t head = g_recHead.load(std::memory_order_acquire);
  uint32_t spins = 0;
  while (g_recTail.load(std::memory_order_acquire) < head) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    if (++spins == 20000) {  // ~2s
      REXGPU_WARN(
          "videonative: [rthread] guest-rewrite drain stuck ~2s "
          "(head={} tail={})",
          head, g_recTail.load());
    }
  }
}

void EnqPushState() {
  // Keep the bound-handle mirror in lockstep with the worker's g_state
  // stack (PopState restores streams/IB worker-side; see BoundState).
  BoundState s;
  std::memcpy(s.streams, g_boundStreams, sizeof(s.streams));
  s.ib = g_boundIb;
  g_boundStack.push_back(s);
  Rec& r = Begin();
  r.op = kPushState;
  Commit();
}

void EnqPopState() {
  if (!g_boundStack.empty()) {  // matches the worker's empty-stack no-op
    const BoundState& s = g_boundStack.back();
    std::memcpy(g_boundStreams, s.streams, sizeof(g_boundStreams));
    g_boundIb = s.ib;
    g_boundStack.pop_back();
  }
  Rec& r = Begin();
  r.op = kPopState;
  Commit();
}

void EnqSeedReplayFloatConstants(uint32_t device, uint32_t recording_device) {
  Rec& r = Begin();
  r.op = kSeedReplayF; r.a = device; r.b = recording_device;
  r.len1 = uint32_t(SnapDevice(device, &r.blob1));
  r.blob3 = SnapDrawInputs(device, false, 0, 0, &r.len3);
  if (recording_device && recording_device != device) {
    r.len2 = uint32_t(SnapDevice(recording_device, &r.blob2));
  }
  Commit();
}

void EnqSetReplayDrawState(const uint32_t* state10, uint32_t dirty_mask) {
  Rec& r = Begin();
  r.op = kReplayDrawState; r.a = dirty_mask;
  if (state10) r.len1 = uint32_t(Blob(state10, 40, &r.blob1));
  Commit();
}

void EnqSetReplayBoolConstants(const uint32_t* dwords40) {
  Rec& r = Begin();
  r.op = kReplayBool;
  if (dwords40) r.len1 = uint32_t(Blob(dwords40, 160, &r.blob1));
  Commit();
}

void EnqSetReplayFetchConstants(const uint32_t* records, uint32_t count) {
  Rec& r = Begin();
  r.op = kReplayFetch; r.a = count;
  if (records && count) {
    r.len1 = uint32_t(Blob(records, count * 28, &r.blob1));
  }
  Commit();
}

void EnqApplyReplayStatePersistent(const uint32_t* state10,
                                   uint32_t dirty_mask) {
  Rec& r = Begin();
  r.op = kReplayPersist; r.a = dirty_mask;
  if (state10) r.len1 = uint32_t(Blob(state10, 40, &r.blob1));
  Commit();
}

void EnqApplyBlendControlDirect(uint32_t rt, uint32_t value) {
  Rec& r = Begin();
  r.op = kBlendDirect; r.a = rt; r.b = value;
  Commit();
}

void EnqApplyLiveStateDirty(uint32_t consumed_mask, const uint32_t vals[5]) {
  Rec& r = Begin();
  r.op = kLiveDirty; r.a = consumed_mask;
  r.b = vals[0]; r.c = vals[1]; r.d = vals[2]; r.e = vals[3]; r.f = vals[4];
  Commit();
}

namespace {
void CaptureShaders(Rec& r) {
  r.p0 = detail::CurrentResolvedVertexShader();
  r.p1 = detail::CurrentResolvedPixelShader();
  r.flag = detail::CurrentDrawPixelShaderBound() ? 1 : 0;
}
}  // namespace

void EnqDrawVertices(uint32_t device, uint32_t prim, uint32_t start_vertex,
                     uint32_t count) {
  Rec& r = Begin();
  r.op = kDrawV; r.a = device; r.b = prim; r.c = start_vertex; r.d = count;
  CaptureShaders(r);
  r.len1 = uint32_t(SnapDevice(device, &r.blob1));
  r.blob3 = SnapDrawInputs(device, false, 0, 0, &r.len3);
  Commit();
}

void EnqDrawIndexedVertices(uint32_t device, uint32_t prim,
                            uint32_t base_vertex, uint32_t start_index,
                            uint32_t index_count) {
  Rec& r = Begin();
  r.op = kDrawIV; r.a = device; r.b = prim; r.c = base_vertex;
  r.d = start_index; r.e = index_count;
  CaptureShaders(r);
  r.len1 = uint32_t(SnapDevice(device, &r.blob1));
  r.blob3 = SnapDrawInputs(device, true, index_count, start_index, &r.len3);
  Commit();
}

void EnqDrawVerticesUP(uint32_t device, uint32_t prim, uint32_t vertex_count,
                       uint32_t vertex_data, uint32_t stride) {
  Rec& r = Begin();
  r.op = kDrawVUP; r.a = device; r.b = prim; r.c = vertex_count;
  r.d = vertex_data; r.e = stride;
  CaptureShaders(r);
  r.len1 = uint32_t(SnapDevice(device, &r.blob1));
  // No stream/IB pack: the UP SetupDraw path never reads bound VB/IB
  // objects (vertex bytes travel in blob2).
  const void* src = g_pendUPv ? g_pendUPv : GuestHost(vertex_data);
  r.len2 = uint32_t(Blob(src, vertex_count * stride, &r.blob2));
  Commit();
}

void EnqDrawIndexedVerticesUP(uint32_t device, uint32_t prim, uint32_t min_v,
                              uint32_t num_v, uint32_t index_count,
                              uint32_t index_data, uint32_t index_format,
                              uint32_t vertex_data, uint32_t stride) {
  (void)index_format;  // 16-bit only (matches the renderer)
  Rec& r = Begin();
  r.op = kDrawIVUP; r.a = device; r.b = prim; r.c = min_v; r.d = num_v;
  r.e = index_count; r.f = index_data; r.h = vertex_data;
  r.fb = float(stride);
  CaptureShaders(r);
  r.len1 = uint32_t(SnapDevice(device, &r.blob1));
  // No stream/IB pack: UP draws carry their vertex+index bytes in blob2.
  // One contiguous blob: vertex bytes then index bytes (r.g = index bytes).
  const uint32_t vlen = num_v * stride;
  const uint32_t ilen = index_count * 2;
  const void* vsrc = g_pendUPv ? g_pendUPv : GuestHost(vertex_data);
  const void* isrc = g_pendUPi ? g_pendUPi : GuestHost(index_data);
  const uint64_t off = ArenaAlloc(vlen + ilen);
  std::memcpy(ArenaPtr(off), vsrc, vlen);
  std::memcpy(ArenaPtr(off) + vlen, isrc, ilen);
  r.blob2 = off;
  r.len2 = vlen;
  r.g = ilen;
  Commit();
}

void SetPendingUPOverride(const void* vertex_data, const void* index_data) {
  g_pendUPv = vertex_data;
  g_pendUPi = index_data;
}

uint64_t GuestThreadCycles() {
  void* h = g_guestThreadHandle.load(std::memory_order_acquire);
  if (!h) return 0;
  ULONG64 cycles = 0;
  QueryThreadCycleTime(static_cast<HANDLE>(h), &cycles);
  return cycles;
}

}  // namespace rex::videonative::rq
