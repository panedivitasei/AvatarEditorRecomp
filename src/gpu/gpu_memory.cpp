// Range tracker plus the lifetime observer hooks on the D3D and XG resource functions. Each hook calls through first
// so the runtime's own header writes land before the range is read back out of the header.

#include "gpu/gpu_memory.h"
#include "gpu/gpu_burst_timing.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <rex/cvar.h>
#include <rex/graphics/xenos.h>
#include <rex/hash.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc/context.h>
#include <rex/system/kernel_state.h>

#include "gpu/gpu.h"
#include "gpu/gpu_texture_decode.h"
#include "gpu/gpu_tracker.h"
#include "gpu/guest/d3d_device.h"

REXCVAR_DEFINE_BOOL(gpu_paranoid, false, "GPU",
                    "Hash every referenced vertex and texture range per draw and log changes no tracked write explains");

extern "C" REX_FUNC(__imp__sub_9211E5F8);
extern "C" REX_FUNC(__imp__sub_9211B6E8);
extern "C" REX_FUNC(__imp__sub_9211B808);
extern "C" REX_FUNC(__imp__sub_9211F500);
extern "C" REX_FUNC(__imp__sub_9211E6C0);
extern "C" REX_FUNC(__imp__sub_9211AFB8);
extern "C" REX_FUNC(__imp__sub_9211B090);
extern "C" REX_FUNC(__imp__sub_92119DD0);
extern "C" REX_FUNC(__imp__sub_9211F480);
extern "C" REX_FUNC(__imp__sub_92268B88);
extern "C" REX_FUNC(__imp__sub_92268B20);
extern "C" REX_FUNC(__imp__sub_92268758);
extern "C" REX_FUNC(__imp__sub_922685A8);
extern "C" REX_FUNC(__imp__sub_92268640);

namespace ae::gpu::memory {

struct Range {
  const uint32_t end;
  std::atomic<uint32_t> generation;
  std::atomic<bool> active{true};

  Range(uint32_t end, uint32_t generation) : end(end), generation(generation) {}
};

namespace {

namespace xenos = rex::graphics::xenos;

std::mutex g_mutex;
std::map<uint32_t, std::shared_ptr<Range>> g_ranges;  // registered, keyed by base, non-overlapping
uint32_t g_generation = 1;
std::atomic<uint64_t> g_revision{1};
std::unordered_map<uint64_t, uint64_t> g_paranoid;  // key -> content hash at upload
std::unordered_set<uint64_t> g_reported;

template <typename T>
const T* Guest(uint32_t address) {
  return REX_KERNEL_MEMORY()->TranslateVirtual<const T*>(address);
}

// Erases registered ranges overlapping [base, end) and returns whether any existed.
void EraseOverlaps(uint32_t base, uint32_t end) {
  auto it = g_ranges.upper_bound(base);
  if (it != g_ranges.begin()) {
    auto prev = std::prev(it);
    if (prev->second->end > base) it = prev;
  }
  while (it != g_ranges.end() && it->first < end) {
    it->second->active.store(false, std::memory_order_release);
    it = g_ranges.erase(it);
  }
}

// Physical data ranges of a resource header: up to two (texture base and mips).
int ResourceRanges(uint32_t resource_address, uint32_t (&base)[2], uint32_t (&size)[2]) {
  if (!resource_address) return 0;
  const auto* resource = Guest<guest::D3DResource>(resource_address);
  const uint32_t type = uint32_t(resource->Common) & 0xF;
  if (type == 1) {
    const auto* vb = Guest<guest::D3DVertexBuffer>(resource_address);
    base[0] = uint32_t(vb->Format[0]) & ~3u;
    size[0] = uint32_t(vb->Format[1]) & 0x3FFFFFC;
    // Engine-built headers over XPhysicalAlloc blocks hold the CPU view of the data; the draw reads the fetch
    // constant's physical address, so the range is keyed the way the draw looks it up.
    if (base[0] >= 0x20000000) {
      const uint32_t physical = REX_KERNEL_MEMORY()->GetPhysicalAddress(base[0]);
      base[0] = physical != UINT32_MAX ? physical : base[0] & 0x1FFFFFFF;
    }
    return base[0] && size[0] ? 1 : 0;
  }
  if (type == 2) {
    // Index buffers carry a virtual address (DrawIndexedVertices converts it for the GPU), so they are keyed by it.
    const auto* ib = Guest<guest::D3DIndexBuffer>(resource_address);
    base[0] = ib->Address;
    size[0] = ib->Size;
    return base[0] && size[0] ? 1 : 0;
  }
  if (type == 3) {
    const auto* texture = Guest<guest::D3DBaseTexture>(resource_address);
    xenos::xe_gpu_texture_fetch_t fetch;
    for (int i = 0; i < 6; ++i) (&fetch.dword_0)[i] = texture->Format.dword[i];
    uint32_t n = 0;
    uint32_t b, bs, m, ms;
    if (TextureMemory(fetch, b, bs, m, ms)) {
      if (b && bs) base[n] = b, size[n++] = bs;
      if (m && ms) base[n] = m, size[n++] = ms;
    }
    return int(n);
  }
  return 0;
}

}  // namespace

void Register(uint32_t base, uint32_t size) {
  BurstTiming timing(10);
  if (!base || !size) return;
  std::lock_guard lock(g_mutex);
  // A header for a piece of a registered block keeps the block registered; only a range that does not fit inside
  // an existing one replaces what it overlaps.
  auto it = g_ranges.upper_bound(base);
  if (it != g_ranges.begin()) {
    auto prev = std::prev(it);
    if (prev->first <= base && prev->second->end >= base + size) {
      prev->second->generation.store(++g_generation, std::memory_order_release);
      return;
    }
  }
  EraseOverlaps(base, base + size);
  g_ranges[base] = std::make_shared<Range>(base + size, ++g_generation);
  g_revision.fetch_add(1, std::memory_order_release);
}

void MarkDirty(uint32_t base, uint32_t size) {
  if (!base || !size) return;
  std::lock_guard lock(g_mutex);
  const uint32_t end = base + size;
  auto it = g_ranges.upper_bound(base);
  if (it != g_ranges.begin()) {
    auto prev = std::prev(it);
    if (prev->second->end > base) it = prev;
  }
  for (; it != g_ranges.end() && it->first < end; ++it) {
    it->second->generation.store(++g_generation, std::memory_order_release);
  }
}

RangeState Query(uint32_t base, uint32_t size) {
  thread_local std::array<QueryCache, 512> cache;
  const uint64_t key = (uint64_t(base) << 32) | size;
  auto& cached = cache[XXH3_64bits(&key, sizeof(key)) & (cache.size() - 1)];
  return Query(base, size, cached);
}

RangeState Query(uint32_t base, uint32_t size, QueryCache& cached) {
  const uint64_t key = (uint64_t(base) << 32) | size;
  if (cached.key == key) {
    if (cached.range && cached.range->active.load(std::memory_order_acquire)) {
      return {true, cached.range->generation.load(std::memory_order_acquire)};
    }
    if (!cached.range && cached.revision == g_revision.load(std::memory_order_acquire)) return {};
  }

  // Cached owners publish dirty versions directly; replacement retires the old owner before erasing it.
  // Misses follow the map revision so a new registration cannot leave an unowned result cached.
  std::lock_guard lock(g_mutex);
  auto it = g_ranges.upper_bound(base);
  std::shared_ptr<Range> range;
  if (it != g_ranges.begin()) {
    --it;
    if (it->second->end >= base + size) range = it->second;
  }
  cached = {key, g_revision.load(std::memory_order_relaxed), std::move(range)};
  return cached.range ? RangeState{true, cached.range->generation.load(std::memory_order_relaxed)} : RangeState{};
}

void RegisterResource(uint32_t resource_address) {
  uint32_t base[2], size[2];
  const int n = ResourceRanges(resource_address, base, size);
  for (int i = 0; i < n; ++i) {
    tracker::InvalidateResolved(base[i], size[i]);
    Register(base[i], size[i]);
  }
}

void DirtyResource(uint32_t resource_address) {
  uint32_t base[2], size[2];
  const int n = ResourceRanges(resource_address, base, size);
  for (int i = 0; i < n; ++i) {
    tracker::InvalidateResolved(base[i], size[i]);
    MarkDirty(base[i], size[i]);
  }
}

bool Paranoid() { return REXCVAR_GET(gpu_paranoid); }

void NoteUploaded(uint64_t key, uint32_t, uint32_t size, const void* bytes) {
  const uint64_t hash = XXH3_64bits(bytes, size);
  std::lock_guard lock(g_mutex);
  g_paranoid[key] = hash;
}

bool CheckUnchanged(uint64_t key, uint32_t base, uint32_t size, const void* bytes) {
  const uint64_t hash = XXH3_64bits(bytes, size);
  std::lock_guard lock(g_mutex);
  auto it = g_paranoid.find(key);
  if (it == g_paranoid.end() || it->second == hash) return true;
  it->second = hash;
  if (g_reported.insert(key).second) {
    REXGPU_WARN("[gpu] paranoid: {:#010x}+{:#x} changed with no tracked write; an engine CPU writer is unhooked", base,
                size);
  }
  return false;
}

void NoteUnregistered(uint32_t base, uint32_t size, const char* what) {
  const uint64_t key = (uint64_t(base) << 32) | size;
  // Every draw over an unowned range lands here, so ranges this thread already noted skip the lock.
  thread_local std::unordered_set<uint64_t> noted;
  if (!noted.insert(key).second) return;
  std::lock_guard lock(g_mutex);
  if (g_reported.insert(key ^ 0x8000000000000000ull).second) {
    // The registered neighbours say whether a hook missed the range or a later registration cut it.
    auto next = g_ranges.upper_bound(base);
    std::string near;
    if (next != g_ranges.begin()) {
      auto prev = std::prev(next);
      near += fmt::format(" prev {:#010x}..{:#010x}", prev->first, prev->second->end);
    }
    if (next != g_ranges.end()) near += fmt::format(" next {:#010x}..{:#010x}", next->first, next->second->end);
    REXGPU_INFO("[gpu] {} {:#010x}+{:#x} has no lifetime owner; re-checked every frame ({} registered ranges;{})",
                what, base, size, g_ranges.size(), near);
  }
}

}  // namespace ae::gpu::memory

namespace ae::gpu {

void MarkRangeDirty(uint32_t base, uint32_t size) {
  if (!base || !size) return;
  // Vertex and texture ranges are keyed physical, index buffers virtual, so both spellings get dirtied.
  memory::MarkDirty(base, size);
  tracker::InvalidateResolved(base, size);
  if (base >= 0x20000000) {
    const uint32_t physical = REX_KERNEL_MEMORY()->GetPhysicalAddress(base);
    const uint32_t key = physical != UINT32_MAX ? physical : base & 0x1FFFFFFF;
    memory::MarkDirty(key, size);
    tracker::InvalidateResolved(key, size);
  }
}

}  // namespace ae::gpu

namespace memory = ae::gpu::memory;

namespace {

uint32_t StackArg(const PPCContext& ctx, uint32_t offset) {
  return *REX_KERNEL_MEMORY()->TranslateVirtual<rex::be<uint32_t>*>(ctx.r1.u32 + offset);
}

}  // namespace

// D3DDevice_CreateVertexBuffer 0x9211E5F8 (Length, Usage, Pool) returns the buffer.
REX_HOOK_RAW(sub_9211E5F8) {
  __imp__sub_9211E5F8(ctx, base);
  memory::RegisterResource(ctx.r3.u32);
}

// D3DVertexBuffer_Unlock 0x9211E6C0 (pBuffer).
REX_HOOK_RAW(sub_9211E6C0) {
  const uint32_t resource = ctx.r3.u32;
  __imp__sub_9211E6C0(ctx, base);
  memory::DirtyResource(resource);
}

// D3DResource_Release 0x9211F480 returns the remaining reference count; only the final release retires its memory.
REX_HOOK_RAW(sub_9211F480) {
  uint32_t addresses[2], sizes[2];
  const int count = memory::ResourceRanges(ctx.r3.u32, addresses, sizes);
  __imp__sub_9211F480(ctx, base);
  if (ctx.r3.u32 == 0) {
    for (int i = 0; i < count; ++i) {
      ae::gpu::tracker::InvalidateResolved(addresses[i], sizes[i]);
      memory::MarkDirty(addresses[i], sizes[i]);
    }
  }
}

// D3DVertexBuffer_Lock 0x9211F500 (pBuffer, Offset, Size, Flags).
REX_HOOK_RAW(sub_9211F500) {
  const uint32_t resource = ctx.r3.u32;
  __imp__sub_9211F500(ctx, base);
  memory::DirtyResource(resource);
}

// D3D::LockSurface 0x9211AFB8 and Lock2DSurface 0x9211B090 (the LockRect thunk 0x9211B6A0 lands there), first
// argument the texture.
REX_HOOK_RAW(sub_9211AFB8) {
  const uint32_t resource = ctx.r3.u32;
  __imp__sub_9211AFB8(ctx, base);
  memory::DirtyResource(resource);
}

REX_HOOK_RAW(sub_9211B090) {
  const uint32_t resource = ctx.r3.u32;
  __imp__sub_9211B090(ctx, base);
  memory::DirtyResource(resource);
}

// Texture unlock 0x92119DD0 (pTexture) flushes the base and mip ranges after CPU writes.
REX_HOOK_RAW(sub_92119DD0) {
  const uint32_t resource = ctx.r3.u32;
  __imp__sub_92119DD0(ctx, base);
  memory::DirtyResource(resource);
}

// D3DDevice_CreateTexture 0x9211B6E8 (Width, Height, Depth, Levels, Usage, Format, Pool, Type) returns the texture.
REX_HOOK_RAW(sub_9211B6E8) {
  __imp__sub_9211B6E8(ctx, base);
  memory::RegisterResource(ctx.r3.u32);
}

// D3DDevice_CreateSurface 0x9211B808 returns the header; only a header with texture-typed memory registers anything.
REX_HOOK_RAW(sub_9211B808) {
  __imp__sub_9211B808(ctx, base);
  memory::RegisterResource(ctx.r3.u32);
}

// XGOffsetBaseTextureAddress 0x92268B20 (pTexture, pBase, pMip) and XGOffsetResourceAddress 0x92268B88 (pResource,
// pBase): the header now points at its data, so the range becomes owned.
REX_HOOK_RAW(sub_92268B20) {
  const uint32_t resource = ctx.r3.u32;
  __imp__sub_92268B20(ctx, base);
  memory::RegisterResource(resource);
}

REX_HOOK_RAW(sub_92268B88) {
  const uint32_t resource = ctx.r3.u32;
  __imp__sub_92268B88(ctx, base);
  memory::RegisterResource(resource);
}

// XGRAPHICS::SetBaseTextureHeader 0x92268758 (Width, Height, Depth, Levels, Usage, Format, MipPacking, ExpBias,
// Flags, BaseOffset, MipOffset, Pitch, ResourceType, pTexture, pBaseSize, pMipSize) is the body behind
// XGSetTextureHeader, XGSetTextureHeaderEx, XGSetLineTextureHeaderEx and XGSetArrayTextureHeaderEx; the avatar
// library builds its texture stacks through the array variant, so hooking here (pTexture at r1+0x7C, read at
// 396(r1) under the 272-byte frame) owns every XG-built texture.
REX_HOOK_RAW(sub_92268758) {
  const uint32_t texture = StackArg(ctx, 0x7C);
  __imp__sub_92268758(ctx, base);
  if (texture) memory::RegisterResource(texture);
}

// XGSetVertexBufferHeader 0x922685A8 (Length, Usage, Pool, BaseOffset, pBuffer).
REX_HOOK_RAW(sub_922685A8) {
  const uint32_t buffer = ctx.r7.u32;
  __imp__sub_922685A8(ctx, base);
  memory::RegisterResource(buffer);
}

// XGSetIndexBufferHeader 0x92268640 (Length, Usage, Format, Pool, BaseOffset, pBuffer): the title builds every index
// buffer this way, never through D3D_CreateIndexBuffer.
REX_HOOK_RAW(sub_92268640) {
  const uint32_t buffer = ctx.r8.u32;
  __imp__sub_92268640(ctx, base);
  memory::RegisterResource(buffer);
}
