// rexvideonative shader cache, resolves guest XDK shader objects to entries
// of the XenosRecomp native shader pack (see docs/native_shaders.md for the
// runtime contract the pack DXIL binds against).
// GPL-3.0, see LICENSE in this directory.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rex::videonative {

class ShaderCache;

struct PackShader {
  std::vector<uint8_t> dxil;
  // Trimmed-signature VS variant (unwritten oVar/oPts outputs removed);
  // empty when the pack predates trimming or the shader writes everything.
  std::vector<uint8_t> dxil_trim;
  // b4 bindings in slot order: kind (sampler/texture), fetch constant,
  // dimension, signedness, mirrors PipelineCache::NativeShaderEntry.
  struct Binding {
    uint8_t is_sampler;
    uint8_t fetch_constant;
    uint8_t dimension;
    uint8_t is_signed;
  };
  std::vector<Binding> bindings;
};

class ShaderCache {
 public:
  // Loads manifest.csv + DXIL blobs from the pack directory. Returns false
  // (and leaves the cache empty) when the pack is missing.
  bool Load(const std::string& pack_dir);

  bool loaded() const { return loaded_; }
  size_t vertex_count() const { return vertex_shaders_.size(); }
  size_t pixel_count() const { return pixel_shaders_.size(); }

  // Rectangle-list expansion GS (fixed interpolator signature, one per pack);
  // empty when the pack predates it.
  const std::vector<uint8_t>& rect_expand_gs() const { return rect_gs_dxil_; }

  // Point-sprite expansion GS (POINTLIST -> screen-aligned quads sized by the
  // VS oPts export); empty when the pack predates it.
  const std::vector<uint8_t>& point_expand_gs() const {
    return point_gs_dxil_;
  }

  // Fullscreen blit pair for the Swap frontbuffer composite; empty when the
  // pack predates them.
  const std::vector<uint8_t>& blit_vs() const { return blit_vs_dxil_; }
  const std::vector<uint8_t>& blit_ps() const { return blit_ps_dxil_; }

  const PackShader* Find(uint64_t ucode_hash, bool pixel) const {
    const auto& map = pixel ? pixel_shaders_ : vertex_shaders_;
    auto it = map.find(ucode_hash);
    return it != map.end() ? &it->second : nullptr;
  }

  // Pack prover: iterate every pixel-shader entry (hash -> PackShader).
  const std::unordered_map<uint64_t, PackShader>& pixel_shaders() const {
    return pixel_shaders_;
  }

 private:
  bool loaded_ = false;
  std::unordered_map<uint64_t, PackShader> vertex_shaders_;
  std::unordered_map<uint64_t, PackShader> pixel_shaders_;
  std::vector<uint8_t> rect_gs_dxil_;
  std::vector<uint8_t> point_gs_dxil_;
  std::vector<uint8_t> blit_vs_dxil_;
  std::vector<uint8_t> blit_ps_dxil_;
};

// Resolution result for a guest XDK shader object, cached by object address.
struct ResolvedShader {
  uint32_t guest_object = 0;
  uint32_t code_ptr = 0;    // guest address of microcode
  uint32_t code_size = 0;   // bytes
  uint64_t ucode_hash = 0;  // XXH3 of the microcode as stored (big-endian)
  const PackShader* pack = nullptr;  // null = pack miss
  bool is_pixel = false;
};

}  // namespace rex::videonative
