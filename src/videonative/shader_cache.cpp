// GPL-3.0, see LICENSE in this directory.

#include "shader_cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <rex/logging.h>

namespace rex::videonative {

bool ShaderCache::Load(const std::string& pack_dir_str) {
  std::filesystem::path pack_dir(pack_dir_str);
  std::ifstream manifest(pack_dir / "manifest.csv");
  if (!manifest.is_open()) {
    REXGPU_WARN("videonative: shader pack manifest not found: {}",
                (pack_dir / "manifest.csv").string());
    return false;
  }

  // runtime_ucode_hash,type,fingerprint,dxil,bindings,float_bitmap,reason
  std::string line;
  std::getline(manifest, line);  // header
  while (std::getline(manifest, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::string columns[5];
    size_t column_start = 0;
    for (uint32_t i = 0; i < 5 && column_start <= line.size(); ++i) {
      size_t comma = line.find(',', column_start);
      columns[i] = line.substr(
          column_start,
          (comma == std::string::npos ? line.size() : comma) - column_start);
      column_start = (comma == std::string::npos) ? line.size() + 1 : comma + 1;
    }
    if (columns[3].empty()) continue;  // not generated (translator fallback)

    std::ifstream dxil_file(pack_dir / columns[3],
                            std::ios::binary | std::ios::ate);
    if (!dxil_file.is_open()) {
      REXGPU_WARN("videonative: shader pack missing {}", columns[3]);
      continue;
    }
    PackShader entry;
    entry.dxil.resize(size_t(dxil_file.tellg()));
    dxil_file.seekg(0);
    dxil_file.read(reinterpret_cast<char*>(entry.dxil.data()),
                   entry.dxil.size());

    // Trimmed VS variant ({hash}.vst.dxil): unwritten outputs removed from
    // the signature. Preferred at draw time except under the expansion GS
    // (which links against the full signature). Absent in older packs, where
    // the dxil above stays the only variant.
    if (columns[1] == "vs") {
      std::ifstream trim_file(pack_dir / (columns[0] + ".vst.dxil"),
                              std::ios::binary | std::ios::ate);
      if (trim_file.is_open()) {
        entry.dxil_trim.resize(size_t(trim_file.tellg()));
        trim_file.seekg(0);
        trim_file.read(reinterpret_cast<char*>(entry.dxil_trim.data()),
                       entry.dxil_trim.size());
      }
    }

    // "kind:fetch_constant:dimension:is_signed" per b4 slot, ';'-separated.
    const std::string& bindings = columns[4];
    size_t binding_start = 0;
    bool bindings_valid = true;
    while (binding_start < bindings.size()) {
      size_t separator = bindings.find(';', binding_start);
      size_t binding_end =
          (separator == std::string::npos) ? bindings.size() : separator;
      PackShader::Binding binding = {};
      uint32_t fields[3] = {};
      if (binding_end - binding_start >= 2 &&
          (bindings[binding_start] == 's' || bindings[binding_start] == 't') &&
          bindings[binding_start + 1] == ':' &&
          sscanf(bindings.c_str() + binding_start + 2, "%u:%u:%u", &fields[0],
                 &fields[1], &fields[2]) == 3) {
        binding.is_sampler = bindings[binding_start] == 's' ? 1 : 0;
        binding.fetch_constant = uint8_t(fields[0]);
        binding.dimension = uint8_t(fields[1]);
        binding.is_signed = uint8_t(fields[2]);
        entry.bindings.push_back(binding);
      } else {
        bindings_valid = false;
        break;
      }
      binding_start =
          (separator == std::string::npos) ? bindings.size() : separator + 1;
    }
    if (!bindings_valid) {
      REXGPU_WARN("videonative: shader pack bad bindings for {}, skipping",
                  columns[0]);
      continue;
    }

    uint64_t ucode_hash = strtoull(columns[0].c_str(), nullptr, 16);
    auto& map = (columns[1] == "ps") ? pixel_shaders_ : vertex_shaders_;
    map.emplace(ucode_hash, std::move(entry));
  }

  auto load_blob = [&](const char* name, std::vector<uint8_t>* out,
                       const char* missing_consequence) {
    std::ifstream file(pack_dir / name, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
      out->resize(size_t(file.tellg()));
      file.seekg(0);
      file.read(reinterpret_cast<char*>(out->data()), out->size());
    } else {
      REXGPU_WARN("videonative: pack has no {}, {} (regenerate the pack)",
                  name, missing_consequence);
    }
  };
  load_blob("rect_expand.gs.dxil", &rect_gs_dxil_,
            "RECTLIST draws will be skipped");
  load_blob("point_expand.gs.dxil", &point_gs_dxil_,
            "POINTLIST draws will render as 1px points");
  load_blob("blit.vs.dxil", &blit_vs_dxil_, "Swap composite will be skipped");
  load_blob("blit.ps.dxil", &blit_ps_dxil_, "Swap composite will be skipped");

  loaded_ = !vertex_shaders_.empty() || !pixel_shaders_.empty();
  REXGPU_INFO("videonative: shader pack loaded, {} vs + {} ps{} from {}",
              vertex_shaders_.size(), pixel_shaders_.size(),
              rect_gs_dxil_.empty() ? "" : " + rect GS", pack_dir_str);
  return loaded_;
}

}  // namespace rex::videonative
