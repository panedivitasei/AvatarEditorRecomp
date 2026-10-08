// Translated shader lookup for the draw path (docs/research/pack_contract.md): the X4 container or the loose
// directory pack behind one interface, chosen by gpu_shader_pack; entries are immutable once published.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace ae::gpu {

enum class ShaderStage : uint8_t { kVertex = 0, kPixel = 1, kGeometry = 2 };

// One b4 slot, in slot order (contract 1.7).
struct PackBinding {
  uint8_t sampler = 0;        // 0 = texture, 1 = sampler
  uint8_t fetch_constant = 0;
  uint8_t dimension = 0;      // texture: 1 = 2D/1D, 2 = 3D, 3 = cube
  uint8_t is_signed = 0;
  uint8_t mag = 0xFF, min = 0xFF, mip = 0xFF, aniso = 0xFF;  // translator overrides; 0xFF = from the fetch constant
};

struct ShaderEntry {
  uint32_t id = 0;            // stable per process, 1-based
  ShaderStage stage = ShaderStage::kVertex;
  uint64_t runtime_hash = 0;
  std::vector<uint8_t> dxil, dxil_trim;
  std::vector<PackBinding> bindings;
  uint32_t vfetch_count = 0;  // X3 layout-table records the VS reads from b5; 0 for the directory pack
  uint16_t written_ovar = 0xFFFF;  // VS: oVar outputs the trimmed variant keeps
  uint16_t read_ivar = 0;          // PS: iVar inputs the signature declares
  bool has_trim() const { return !dxil_trim.empty(); }
};

// ABI constants the loaded pack was built against; the host lays out its bindings from them.
struct PackAbi {
  uint32_t system_size = 256;   // b0 bytes
  bool vfetch_table = false;    // b5 present (X3)
  uint32_t texture_heap = 16384;
  uint32_t texture_clamp = 2047;  // highest descriptor index the shaders reach
  uint32_t sampler_heap = 16;
  uint32_t shared_mem_size = 0x8000000;
};

// Picks and opens the pack; safe to call more than once. False leaves every lookup missing.
bool LoadShaderPack();
const PackAbi& ShaderPackAbi();

// Hash of the loaded pack's header (0 for a directory pack), and a file path next to the pack for records that
// belong to exactly this pack, such as the pipeline list the renderer warms at start-up.
uint64_t ShaderPackIdentity();
std::filesystem::path ShaderPackSidecar(const char* name);
std::string ShaderPackName();

// Guest thread: finds the translation by runtime microcode hash (contract 3.1, directory pack) or fp2 (3.3,
// container). Null is a pack miss.
const ShaderEntry* FindShader(ShaderStage stage, uint64_t runtime_hash, uint64_t fp2);

// Guest thread: a pack miss translated at runtime (or read back from the disk cache); source names which. Null when
// translation failed or the loaded pack has no fallback.
const ShaderEntry* TranslateMiss(ShaderStage stage, uint64_t fp2, std::span<const uint32_t> ucode, const char** source);

// True when lookups need fp2 and draws need the b5 vfetch layout table.
bool ShaderPackUsesFp2();

// Render thread only; a missing or incompatible bundle retains the original pixel shader.
const ShaderEntry* SpecializePixelShader(const ShaderEntry* original, uint32_t boolean_word);

// Built-in geometry shaders: "rect_expand" and "point_expand".
const ShaderEntry* BuiltinShader(const char* name);

}  // namespace ae::gpu
