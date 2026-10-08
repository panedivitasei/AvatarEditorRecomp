#include "gpu/gpu_shader_pack.h"
#include "gpu/gpu_burst_timing.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <xenosrecomp.h>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/hash.h>
#include <rex/logging.h>

REXCVAR_DEFINE_STRING(gpu_shader_pack, "auto", "GPU",
                      "Shader pack: a .pack container, a directory holding manifest.csv, or auto (out/shaderpack/ae.pack, "
                      "then shaderpack/)");

REXCVAR_DEFINE_STRING(gpu_shader_cache_dir, "", "GPU",
                      "Runtime translation cache root; empty = gpu_shader_cache next to the executable, none = off");
REXCVAR_DEFINE_BOOL(gpu_shader_prewarm, true, "GPU", "Prepare immutable container shader entries before first draw");
REXCVAR_DEFINE_BOOL(gpu_boolean_variants, true, "GPU", "Use precompiled pixel shader boolean variants when available");

namespace ae::gpu {

namespace {

namespace fs = std::filesystem;

std::vector<uint8_t> ReadFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

uint64_t ParseHex(const std::string& s) {
  uint64_t v = 0;
  for (char c : s) {
    v <<= 4;
    if (c >= '0' && c <= '9') v |= uint64_t(c - '0');
    else if (c >= 'A' && c <= 'F') v |= uint64_t(c - 'A' + 10);
    else if (c >= 'a' && c <= 'f') v |= uint64_t(c - 'a' + 10);
  }
  return v;
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r') {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

// "t:<fetch>:<dimension>:<signed>" and "s:<fetch>:0:0", ';'-separated (contract 1.7).
std::vector<PackBinding> ParseBindings(const std::string& text) {
  std::vector<PackBinding> out;
  if (text.empty()) return out;
  for (const auto& item : Split(text, ';')) {
    const auto f = Split(item, ':');
    if (f.size() != 4) continue;
    PackBinding b;
    b.sampler = f[0] == "s" ? 1 : 0;
    b.fetch_constant = uint8_t(std::stoul(f[1]));
    b.dimension = uint8_t(std::stoul(f[2]));
    b.is_signed = uint8_t(std::stoul(f[3]));
    out.push_back(b);
  }
  return out;
}

class Source {
 public:
  virtual ~Source() = default;
  virtual const ShaderEntry* Find(ShaderStage stage, uint64_t runtime_hash) = 0;
  virtual const ShaderEntry* Builtin(const std::string& name) = 0;
  PackAbi abi;
  std::string name;
  bool uses_fp2 = false;  // keyed by the fp2 fingerprint instead of the runtime microcode hash
  uint64_t identity = 0;
  fs::path sidecar_dir;
};

// shaderpack/manifest.csv plus <hash>.<vs|ps|vst>.dxil; files load on first use.
class DirectorySource : public Source {
 public:
  bool Open(const fs::path& dir) {
    dir_ = dir;
    std::ifstream in(dir / "manifest.csv");
    if (!in) return false;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto f = Split(line, ',');
      if (f.size() < 6 || f[3].empty()) continue;
      Row row;
      row.hash = ParseHex(f[0]);
      row.stage = f[1] == "ps" ? ShaderStage::kPixel : ShaderStage::kVertex;
      row.bindings = f[4];
      rows_[Key(row.stage, row.hash)] = std::move(row);
    }
    // The directory pack predates X3: b0 is 256 bytes, no b5, and the shaders clamp texture indices at 2047.
    abi = PackAbi{};
    name = "directory pack " + dir.string();
    return !rows_.empty();
  }

  const ShaderEntry* Find(ShaderStage stage, uint64_t hash) override {
    std::lock_guard lock(mutex_);
    auto it = rows_.find(Key(stage, hash));
    if (it == rows_.end()) return nullptr;
    Row& row = it->second;
    if (!row.entry) {
      auto e = std::make_unique<ShaderEntry>();
      e->id = next_id_++;
      e->stage = stage;
      e->runtime_hash = hash;
      char stem[32];
      std::snprintf(stem, sizeof(stem), "%016llX", static_cast<unsigned long long>(hash));
      const char* type = stage == ShaderStage::kPixel ? "ps" : "vs";
      e->dxil = ReadFile(dir_ / (std::string(stem) + "." + type + ".dxil"));
      if (stage == ShaderStage::kVertex) {
        e->dxil_trim = ReadFile(dir_ / (std::string(stem) + ".vst.dxil"));
      }
      e->bindings = ParseBindings(row.bindings);
      row.entry = std::move(e);
    }
    return row.entry.get();
  }

  const ShaderEntry* Builtin(const std::string& builtin) override {
    std::lock_guard lock(mutex_);
    auto& slot = builtins_[builtin];
    if (!slot) {
      auto e = std::make_unique<ShaderEntry>();
      e->id = next_id_++;
      e->stage = ShaderStage::kGeometry;
      e->dxil = ReadFile(dir_ / (builtin + ".gs.dxil"));
      slot = std::move(e);
    }
    return slot->dxil.empty() ? nullptr : slot.get();
  }

 private:
  struct Row {
    uint64_t hash = 0;
    ShaderStage stage = ShaderStage::kVertex;
    std::string bindings;
    std::unique_ptr<ShaderEntry> entry;
  };
  static uint64_t Key(ShaderStage stage, uint64_t hash) { return hash ^ (uint64_t(stage) << 63); }

  fs::path dir_;
  std::mutex mutex_;
  std::unordered_map<uint64_t, Row> rows_;
  std::unordered_map<std::string, std::unique_ptr<ShaderEntry>> builtins_;
  uint32_t next_id_ = 1;
};

// Fills a ShaderEntry from translated blobs and a serialized reflection record.
void FillEntry(ShaderEntry& e, std::span<const uint8_t> reflection) {
  xenosrecomp::ReflectionView view;
  if (!xenosrecomp::ParseReflection(reflection, view)) return;
  e.vfetch_count = view.header->vfetch_count;
  if (e.stage == ShaderStage::kVertex) e.written_ovar = view.header->vs_written_ovar;
  if (e.stage == ShaderStage::kPixel) e.read_ivar = view.header->ps_read_ivar;
  for (const auto& rb : view.bindings) {
    PackBinding b;
    b.sampler = rb.kind;
    b.fetch_constant = rb.fetch_constant;
    b.dimension = rb.dimension;
    b.is_signed = rb.is_signed;
    b.mag = rb.mag, b.min = rb.min, b.mip = rb.mip, b.aniso = rb.aniso;
    e.bindings.push_back(b);
  }
}

// The X4 container through the translator library's reader: its load checks cover magic, versions, the ABI
// constants, abi_hash and file_xxh3. Entries are keyed by (stage, fp2) and blobs decompress on first use.
class ContainerSource : public Source {
 public:
  bool Open(const fs::path& path) {
    name = "container " + path.string();
    std::string error;
    if (!pack_.OpenFile(path, &error)) {
      REXGPU_ERROR("[gpu] shader pack {} refused: {}", path.string(), error);
      return false;
    }
    abi.system_size = pack_.header().b0_size;
    abi.vfetch_table = pack_.header().push_slots >= 9;
    abi.texture_heap = pack_.header().texture_heap;
    abi.texture_clamp = abi.texture_heap - 1;
    abi.sampler_heap = pack_.header().sampler_heap;
    abi.shared_mem_size = pack_.header().shared_mem_size;
    uses_fp2 = true;
    identity = XXH3_64bits(&pack_.header(), sizeof(pack_.header()));
    sidecar_dir = path.parent_path();
    if (REXCVAR_GET(gpu_shader_prewarm)) {
      warm_ = std::jthread([this](std::stop_token stop) {
        for (const auto& pe : pack_.entries()) {
          if (stop.stop_requested()) break;
          if (pe.stage <= xenosrecomp::kPackGS) Entry(&pe, ShaderStage(pe.stage));
        }
      });
    }
    return true;
  }

  const ShaderEntry* Find(ShaderStage stage, uint64_t fp2) override {
    return Entry(pack_.Find(uint8_t(stage), fp2), stage);
  }

  const ShaderEntry* Builtin(const std::string& builtin) override {
    return Entry(pack_.FindBuiltin(xenosrecomp::kPackGS, builtin), ShaderStage::kGeometry);
  }

 private:
  const ShaderEntry* Entry(const xenosrecomp::PackEntry* pe, ShaderStage stage) {
    if (!pe) return nullptr;
    {
      std::lock_guard lock(mutex_);
      const auto it = entries_.find(pe);
      if (it != entries_.end()) return it->second.get();
    }
    // Decompression owns its outputs and reads the immutable pack outside the publication lock.
    // A racing lookup may prepare a duplicate, but every caller receives the published entry.
      auto e = std::make_unique<ShaderEntry>();
      e->id = uint32_t(pe - pack_.entries().data()) + 1;
      e->stage = stage;
      e->runtime_hash = pe->fp2;
      std::string error;
      pack_.Decompress(*pe, xenosrecomp::BlobKind::kDxil, e->dxil, &error);
      pack_.Decompress(*pe, xenosrecomp::BlobKind::kDxilTrim, e->dxil_trim, &error);
      FillEntry(*e, pack_.Reflection(*pe));
    std::lock_guard lock(mutex_);
    auto& slot = entries_[pe];
    if (!slot) slot = std::move(e);
    return slot.get();
  }

  xenosrecomp::Pack pack_;
  std::mutex mutex_;
  std::unordered_map<const xenosrecomp::PackEntry*, std::unique_ptr<ShaderEntry>> entries_;
  std::jthread warm_;
};

std::mutex g_mutex;
std::unique_ptr<Source> g_source;
bool g_tried = false;
PackAbi g_default_abi;
fs::path g_variant_root;

}  // namespace

bool LoadShaderPack() {
  std::lock_guard lock(g_mutex);
  if (g_tried) return g_source != nullptr;
  g_tried = true;
  const fs::path root = AE_SOURCE_DIR;
  std::string wanted = REXCVAR_GET(gpu_shader_pack);
  std::vector<fs::path> candidates;
  if (wanted.empty() || wanted == "auto") {
    // A shipped build carries the pack beside the exe; a dev build falls back to the source tree.
    const fs::path exe = rex::filesystem::GetExecutableFolder();
    candidates = {exe / "shaderpack" / "ae.pack", root / "out" / "shaderpack" / "ae.pack", root / "shaderpack"};
  } else {
    candidates = {fs::path(wanted)};
  }
  for (const auto& path : candidates) {
    std::error_code ec;
    if (!fs::exists(path, ec)) continue;
    std::unique_ptr<Source> source;
    if (fs::is_directory(path, ec)) {
      auto dir = std::make_unique<DirectorySource>();
      if (dir->Open(path)) source = std::move(dir);
    } else {
      auto container = std::make_unique<ContainerSource>();
      if (container->Open(path)) source = std::move(container);
    }
    if (source) {
      REXGPU_INFO("[gpu] shaders from {}", source->name);
      g_source = std::move(source);
      g_variant_root = (fs::is_directory(path, ec) ? path : path.parent_path()) / "boolean_variants";
      return true;
    }
  }
  REXGPU_ERROR("[gpu] no usable shader pack ({}); every draw is a pack miss", wanted);
  return false;
}

const PackAbi& ShaderPackAbi() { return g_source ? g_source->abi : g_default_abi; }

uint64_t ShaderPackIdentity() { return g_source ? g_source->identity : 0; }

std::filesystem::path ShaderPackSidecar(const char* name) {
  if (!g_source || g_source->sidecar_dir.empty()) return {};
  return g_source->sidecar_dir / name;
}

std::string ShaderPackName() { return g_source ? g_source->name : std::string("none"); }

const ShaderEntry* FindShader(ShaderStage stage, uint64_t runtime_hash, uint64_t fp2) {
  BurstTiming timing(15);
  if (!g_source) return nullptr;
  return g_source->Find(stage, g_source->uses_fp2 ? fp2 : runtime_hash);
}

bool ShaderPackUsesFp2() { return g_source && g_source->uses_fp2; }

// Boolean-variant bundle magic; no AE tool writes bundles yet, so every lookup keeps the original shader.
constexpr char kVariantMagic[8] = {'A', 'E', 'B', 'O', 'O', 'L', '0', '1'};

const ShaderEntry* SpecializePixelShader(const ShaderEntry* original, uint32_t boolean_word) {
  if (!original || original->stage != ShaderStage::kPixel || !REXCVAR_GET(gpu_boolean_variants)) return original;
  struct Bundle {
    uint32_t mask = 0;
    std::unordered_map<uint32_t, std::vector<uint8_t>> blobs;
    std::unordered_map<uint32_t, std::unique_ptr<ShaderEntry>> entries;
  };
  static std::unordered_map<const ShaderEntry*, Bundle> bundles;
  static uint32_t next_id = 0x80000000u;
  auto [it, inserted] = bundles.try_emplace(original);
  auto& bundle = it->second;
  if (inserted) {
    std::ostringstream filename;
    filename << std::uppercase << std::hex << std::setw(16) << std::setfill('0') << original->runtime_hash << ".bin";
    const auto data = ReadFile(g_variant_root / filename.str());
    if (data.size() < 20 || std::memcmp(data.data(), kVariantMagic, 8) != 0) return original;
    size_t offset = 8;
    auto word = [&]() {
      uint32_t value = 0;
      if (offset + 4 <= data.size()) { std::memcpy(&value, data.data() + offset, 4); offset += 4; }
      return value;
    };
    const uint32_t base_size = word(), mask = word(), count = word();
    if (base_size != original->dxil.size() || base_size > data.size() - offset ||
        !mask || (mask & 0xFFFFu) || std::popcount(mask) > 10 || count != (1u << std::popcount(mask)) ||
        std::memcmp(data.data() + offset, original->dxil.data(), base_size) != 0) {
      REXGPU_WARN("[gpu] boolean variants for {:016X} rejected: base shader or mask mismatch", original->runtime_hash);
      return original;
    }
    offset += base_size;
    for (uint32_t i = 0; i < count; ++i) {
      if (data.size() - offset < 8) break;
      const uint32_t value = word(), size = word();
      if ((value & ~mask) || size < 32 || size > data.size() - offset ||
          std::memcmp(data.data() + offset, "DXBC", 4) != 0) break;
      uint32_t container_size = 0;
      std::memcpy(&container_size, data.data() + offset + 24, 4);
      if (container_size != size) break;
      bundle.blobs.emplace(value, std::vector<uint8_t>(data.begin() + offset, data.begin() + offset + size));
      offset += size;
    }
    if (bundle.blobs.size() != count || offset != data.size()) {
      bundle.blobs.clear();
      REXGPU_WARN("[gpu] boolean variants for {:016X} rejected: incomplete bundle", original->runtime_hash);
      return original;
    }
    bundle.mask = mask;
    REXGPU_INFO("[gpu] loaded {} boolean variants for {:016X}, mask {:08X}", count, original->runtime_hash, mask);
  }
  if (!bundle.mask) return original;
  const uint32_t value = (boolean_word << 16) & bundle.mask;
  auto& entry = bundle.entries[value];
  if (!entry) {
    entry = std::make_unique<ShaderEntry>(*original);
    entry->id = next_id++;
    entry->dxil = bundle.blobs.at(value);
    REXGPU_INFO("[gpu] specialized PS {:016X} bool {:08X}: {} -> {} DXIL bytes", original->runtime_hash,
                value, original->dxil.size(), entry->dxil.size());
  }
  return entry.get();
}

namespace {

std::mutex g_fallback_mutex;
std::unordered_map<uint64_t, std::unique_ptr<ShaderEntry>> g_fallback;  // (stage, fp2) -> entry, null = failed
uint32_t g_fallback_id = 0x40000000;

fs::path CacheRoot() {
  const std::string configured = REXCVAR_GET(gpu_shader_cache_dir);
  if (configured == "none") return {};
  if (!configured.empty()) return fs::path(configured);
  return rex::filesystem::GetExecutableFolder() / "gpu_shader_cache";
}

}  // namespace

const ShaderEntry* TranslateMiss(ShaderStage stage, uint64_t fp2, std::span<const uint32_t> ucode,
                                 const char** source) {
  *source = "failed";
  if (!ShaderPackUsesFp2()) return nullptr;
  const uint64_t key = fp2 ^ (uint64_t(stage) << 62);
  std::lock_guard lock(g_fallback_mutex);
  if (auto it = g_fallback.find(key); it != g_fallback.end()) {
    *source = it->second ? "translated" : "failed";
    return it->second.get();
  }
  const auto xstage = stage == ShaderStage::kPixel ? xenosrecomp::Stage::kPixel : xenosrecomp::Stage::kVertex;
  const fs::path root = CacheRoot();
  const fs::path dir = root.empty() ? fs::path() : xenosrecomp::CacheDirectory(root);
  auto e = std::make_unique<ShaderEntry>();
  e->id = g_fallback_id++;
  e->stage = stage;
  e->runtime_hash = fp2;
  std::string error;
  xenosrecomp::Pack cached;
  if (!dir.empty() && xenosrecomp::CacheLoad(dir, xstage, fp2, cached, &error) && !cached.entries().empty()) {
    const auto& pe = cached.entries()[0];
    cached.Decompress(pe, xenosrecomp::BlobKind::kDxil, e->dxil, &error);
    cached.Decompress(pe, xenosrecomp::BlobKind::kDxilTrim, e->dxil_trim, &error);
    FillEntry(*e, cached.Reflection(pe));
    *source = "cache";
  } else {
    // Container-less: the translator derives the float file and export mask from the microcode itself.
    xenosrecomp::TranslateInput in;
    in.stage = xstage;
    in.ucode = ucode;
    const auto result = xenosrecomp::Translate(in);
    if (!result.ok) {
      REXGPU_ERROR("[gpu] runtime translation of {} fp2={:016X} failed: {}",
                   stage == ShaderStage::kPixel ? "ps" : "vs", fp2, result.error);
      g_fallback[key] = nullptr;
      return nullptr;
    }
    e->dxil = result.dxil, e->dxil_trim = result.dxil_trim;
    FillEntry(*e, result.reflection);
    if (!dir.empty()) {
      std::error_code ec;
      fs::create_directories(dir, ec);
      xenosrecomp::CacheStore(dir, xstage, result, &error);
      // The inputs a pack rebuild needs to cover this shader (contract 5.4).
      xenosrecomp::CacheWriteMiss(dir, xstage, result.runtime_ucode_hash, ucode, {}, &error);
    }
    *source = "translated";
  }
  auto* out = e.get();
  g_fallback[key] = std::move(e);
  return out;
}

const ShaderEntry* BuiltinShader(const char* name) { return g_source ? g_source->Builtin(name) : nullptr; }

}  // namespace ae::gpu
