#include "gpu/gpu_tracker.h"
#include "gpu/gpu_burst_timing.h"
#include "gpu/gpu_resolve_cache.h"
#include "gpu/gpu_texture_decode.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>

#include "gpu/guest/d3d_device.h"

namespace ae::gpu::tracker {

namespace {

namespace xenos = rex::graphics::xenos;

template <typename T>
const T* Guest(uint32_t address) {
  return REX_KERNEL_MEMORY()->TranslateVirtual<const T*>(address);
}

// EDRAM placement: surfaces the game creates at the same base, format and size share one host target, as they
// share the same EDRAM tiles on hardware. A different format at the same base stays separate (aliasing is Phase 4).
struct SurfaceKey {
  ResourceKind kind;
  uint32_t edram_base;
  uint32_t format;      // colour or depth format field
  uint32_t dimensions;
  uint32_t msaa;
  auto operator<=>(const SurfaceKey&) const = default;
};

// A tile-sized surface BeginTiling tiles: the host keeps it at the size of the whole tiled area. The header's info
// and dimensions at registration let a freed and reused surface address drop the entry.
struct TiledSurface {
  uint32_t width;
  uint32_t height;
  uint32_t info;
  uint32_t dimensions;
};

std::mutex g_mutex;
std::map<SurfaceKey, ResourceDesc> g_surfaces;
std::map<uint32_t, TiledSurface> g_tiled;  // keyed by guest surface address
ResolveTextureCache g_textures;
uint32_t g_next_id = 1;

}  // namespace

ResourceDesc Surface(uint32_t surface_address, ResourceKind kind) {
  BurstTiming timing(6);
  if (!surface_address) return {};
  const auto* surface = Guest<guest::D3DSurface>(surface_address);
  // A texture-level surface keeps its parent texture where the EDRAM fields would be; it cannot be bound as a target.
  if (uint32_t(surface->resource.Common) & guest::kSurfaceFromTexture) {
    static std::atomic<bool> once{false};
    if (!once.exchange(true)) REXGPU_ERROR("[gpu] surface {:#010x} bound as a target is a texture level", surface_address);
    return {};
  }
  std::lock_guard lock(g_mutex);
  const TiledSurface* tiled = nullptr;
  if (auto it = g_tiled.find(surface_address); it != g_tiled.end()) {
    if (uint32_t(surface->ColorOrDepthInfo) == it->second.info && uint32_t(surface->Dimensions) == it->second.dimensions) {
      tiled = &it->second;
    } else {
      g_tiled.erase(it);
    }
  }
  const uint32_t info = surface->ColorOrDepthInfo;
  const bool depth = kind == ResourceKind::kDepthTarget;
  // Base is the 12-bit tile index; the format sits at bit 16 (one bit for depth, four for colour).
  // A tiled surface keys on its host size, flagged by the top bit, so its tile-sized twin stays a separate target.
  const uint32_t dimensions = tiled ? 0x80000000u | (tiled->width << 16) | tiled->height : uint32_t(surface->Dimensions);
  const SurfaceKey key{kind, info & 0xFFF, depth ? (info >> 16) & 1 : (info >> 16) & 0xF, dimensions,
                       uint32_t(surface->SurfaceInfo.get().msaa_samples)};
  if (auto it = g_surfaces.find(key); it != g_surfaces.end()) return it->second;
  ResourceDesc desc;
  desc.id = g_next_id++;
  desc.kind = kind;
  desc.width = tiled ? tiled->width : surface->width();
  desc.height = tiled ? tiled->height : surface->height();
  desc.samples = uint8_t(1u << std::min<uint32_t>(key.msaa, 2));
  desc.format = depth ? HostFormatForDepthTarget(key.format) : HostFormatForColorTarget(key.format);
  REXGPU_INFO("[gpu] {} surface {:#010x} -> host {} {}x{} {} (guest format {}, edram base {}, msaa {}{})",
              depth ? "depth" : "colour", surface_address, desc.id, desc.width, desc.height,
              HostFormatName(desc.format), key.format, key.edram_base, key.msaa, tiled ? ", whole tiled area" : "");
  g_surfaces.emplace(key, desc);
  return desc;
}

void RegisterTiledSurface(uint32_t surface_address, uint32_t width, uint32_t height) {
  if (!surface_address || !width || !height) return;
  const auto* surface = Guest<guest::D3DSurface>(surface_address);
  if (uint32_t(surface->resource.Common) & guest::kSurfaceFromTexture) return;
  width = std::max(width, surface->width());
  height = std::max(height, surface->height());
  std::lock_guard lock(g_mutex);
  g_tiled[surface_address] = TiledSurface{width, height, uint32_t(surface->ColorOrDepthInfo), uint32_t(surface->Dimensions)};
}

ResourceDesc Texture(uint32_t texture_address) {
  BurstTiming timing(6);
  if (!texture_address) return {};
  const auto* texture = Guest<guest::D3DBaseTexture>(texture_address);
  xenos::xe_gpu_texture_fetch_t fetch;
  fetch.dword_0 = texture->Format.dword[0];
  fetch.dword_1 = texture->Format.dword[1];
  fetch.dword_2 = texture->Format.dword[2];
  fetch.dword_3 = texture->Format.dword[3];
  fetch.dword_4 = texture->Format.dword[4];
  fetch.dword_5 = texture->Format.dword[5];
  // Headers made by CreateTexture hold a CPU view of the data (0xE0000000 and up is offset by a page); SetTexture
  // converts it, so the draw side sees the physical address and the key here must match it.
  uint32_t base = fetch.base_address << 12;
  if (base >= 0x20000000) {
    const uint32_t physical = REX_KERNEL_MEMORY()->GetPhysicalAddress(base);
    base = physical != UINT32_MAX ? physical : base & 0x1FFFFFFF;
  }
  if (!base) return {};
  if (fetch.dimension != xenos::DataDimension::k2DOrStacked) {
    static std::atomic<bool> once{false};
    if (!once.exchange(true)) {
      REXGPU_ERROR("[gpu] texture {:#010x} has dimension {}; only its first 2D slice is tracked", texture_address,
                   uint32_t(fetch.dimension));
    }
  }
  ResourceDesc want;
  want.kind = ResourceKind::kTexture;
  want.width = fetch.size_2d.width + 1;
  want.height = fetch.size_2d.height + 1;
  const uint32_t guest_format = uint32_t(fetch.format);
  std::lock_guard lock(g_mutex);
  const ResourceDesc have = g_textures.Find(base, want.width, want.height, guest_format);
  if (have.id) return have;
  want.format = HostFormatForTexture(guest_format);
  want.id = g_next_id++;
  REXGPU_DEBUG("[gpu] texture at {:#010x} -> host {} {}x{} {} (guest format {})", base, want.id, want.width,
              want.height, HostFormatName(want.format), guest_format);
  uint32_t physical, size, mip, mip_size;
  TextureMemory(fetch, physical, size, mip, mip_size);
  g_textures.Store(base, std::max(size, 1u), want, guest_format);
  return want;
}

uint32_t ResolveEpoch() { return g_textures.Epoch(); }

ResourceDesc FindResolved(uint32_t base_address, uint32_t width, uint32_t height, uint32_t guest_format) {
  std::lock_guard lock(g_mutex);
  return g_textures.Find(base_address, width, height, guest_format);
}

void ForgetResolved(uint32_t id) {
  std::lock_guard lock(g_mutex);
  g_textures.Erase(id);
}

void TakeRetiredResolves(std::vector<uint32_t>& ids) {
  std::lock_guard lock(g_mutex);
  g_textures.TakeRetired(ids);
}

void InvalidateResolved(uint32_t base, uint32_t size) {
  if (!base || !size) return;
  std::lock_guard lock(g_mutex);
  const uint32_t removed = g_textures.Invalidate(base, size);
  if (removed) {
    REXGPU_DEBUG("[gpu] retired {} resolve aliases for reused/dirty memory {:#010x}+{:#x}", removed, base, size);
  }
}

uint32_t NewId() {
  std::lock_guard lock(g_mutex);
  return g_next_id++;
}

}  // namespace ae::gpu::tracker
