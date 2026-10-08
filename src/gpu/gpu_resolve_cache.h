#pragma once

#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gpu/gpu_records.h"

namespace ae::gpu::tracker {

// Callers serialize the table; the epoch lets draw caches notice retired address aliases.
class ResolveTextureCache {
 public:
  ResourceDesc Find(uint32_t base) const {
    const auto it = entries_.find(base);
    return it == entries_.end() ? ResourceDesc{} : it->second.desc;
  }

  ResourceDesc Find(uint32_t base, uint32_t width, uint32_t height, uint32_t guest_format) const {
    const auto it = entries_.find(base);
    if (it == entries_.end()) return {};
    const auto& entry = it->second;
    return entry.desc.width == width && entry.desc.height == height && entry.guest_format == guest_format
        ? entry.desc : ResourceDesc{};
  }

  void Store(uint32_t base, uint32_t size, const ResourceDesc& desc, uint32_t guest_format) {
    auto& entry = entries_[base];
    if (entry.desc.id && entry.desc.id != desc.id) retired_.push_back(entry.desc.id);
    entry = {desc, size, guest_format};
    epoch_.fetch_add(1, std::memory_order_release);
  }

  // Host ids whose alias went away since the last call; the host may destroy those textures.
  void TakeRetired(std::vector<uint32_t>& ids) { ids.swap(retired_); retired_.clear(); }

  uint32_t Invalidate(uint32_t base, uint32_t size) {
    if (!size) return 0;
    uint32_t removed = 0;
    for (auto it = entries_.begin(); it != entries_.end();) {
      if (uint64_t(base) < uint64_t(it->first) + it->second.size &&
          uint64_t(it->first) < uint64_t(base) + size) {
        retired_.push_back(it->second.desc.id);
        it = entries_.erase(it);
        ++removed;
      } else {
        ++it;
      }
    }
    if (removed) epoch_.fetch_add(1, std::memory_order_release);
    return removed;
  }

  void Erase(uint32_t id) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->second.desc.id != id) continue;
      entries_.erase(it);
      epoch_.fetch_add(1, std::memory_order_release);
      return;
    }
  }

  uint32_t Epoch() const { return epoch_.load(std::memory_order_acquire); }

 private:
  struct Entry {
    ResourceDesc desc;
    uint32_t size;
    uint32_t guest_format;
  };
  std::unordered_map<uint32_t, Entry> entries_;
  std::vector<uint32_t> retired_;
  std::atomic<uint32_t> epoch_{1};
};

}  // namespace ae::gpu::tracker
