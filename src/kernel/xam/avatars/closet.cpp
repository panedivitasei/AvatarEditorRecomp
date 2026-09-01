/**
 ******************************************************************************
 * ReXGlue avatar closet: imported Xbox LIVE marketplace avatar items.       *
 ******************************************************************************
 */

#include "closet.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include <rex/logging.h>

#include "xe_compat.h"

// PNG encoder for generated icons (EncodePng). The static build keeps the
// symbols TU-local; nothing else in the runtime compiles this implementation.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

namespace rex {
namespace avatars {

static const uint8_t kStockGuidTail[8] = {0xC1, 0xC8, 0xF1, 0x09,
                                          0xA1, 0x9C, 0xB2, 0xE0};

bool IsStockPackId(const AssetId& id) {
  return std::memcmp(id.d, kStockGuidTail, sizeof(kStockGuidTail)) == 0;
}

bool IsAwardId(const AssetId& id) {
  return ((id.c.get() >> 8) & 0xFu) == 1u;
}

uint32_t TitleIdOf(const AssetId& id) {
  return (uint32_t(id.d[4]) << 24) | (uint32_t(id.d[5]) << 16) | (uint32_t(id.d[6]) << 8) |
         uint32_t(id.d[7]);
}

// "aaaaaaaa-bbbb-cccc-dddd-dddddddddddd" -> AssetId. Returns false on any
// malformed field.
static bool ParseGuid(const std::string& s, AssetId* out) {
  if (s.size() != 36 || s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-') {
    return false;
  }
  auto hex = [](const std::string& part, uint64_t* value) {
    char* end = nullptr;
    *value = std::strtoull(part.c_str(), &end, 16);
    return end && *end == '\0';
  };
  uint64_t a, b, c, d0, d1;
  if (!hex(s.substr(0, 8), &a) || !hex(s.substr(9, 4), &b) || !hex(s.substr(14, 4), &c) ||
      !hex(s.substr(19, 4), &d0) || !hex(s.substr(24, 12), &d1)) {
    return false;
  }
  *out = AssetId{};
  out->a = static_cast<uint32_t>(a);
  out->b = static_cast<uint16_t>(b);
  out->c = static_cast<uint16_t>(c);
  out->d[0] = static_cast<uint8_t>(d0 >> 8);
  out->d[1] = static_cast<uint8_t>(d0 >> 0);
  for (int i = 0; i < 6; ++i) {
    out->d[2 + i] = static_cast<uint8_t>(d1 >> (8 * (5 - i)));
  }
  return true;
}

bool Closet::Load(const std::filesystem::path& dir) {
  if (is_loaded_) {
    return true;
  }
  dir_ = dir;
  const std::filesystem::path index_path = dir / "closet_index.tsv";
  FILE* f = std::fopen(index_path.string().c_str(), "rb");
  if (!f) {
    return false;  // no closet: fine
  }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string text((size_t)std::max(0L, sz), '\0');
  size_t got = std::fread(text.data(), 1, text.size(), f);
  std::fclose(f);
  text.resize(got);

  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    // guid \t categories-hex \t bodies \t name
    size_t t1 = line.find('\t');
    size_t t2 = t1 == std::string::npos ? std::string::npos : line.find('\t', t1 + 1);
    size_t t3 = t2 == std::string::npos ? std::string::npos : line.find('\t', t2 + 1);
    if (t3 == std::string::npos) continue;
    ClosetItem item{};
    const std::string guid = line.substr(0, t1);
    if (!ParseGuid(guid, &item.id)) continue;
    item.categories = (uint32_t)std::strtoul(line.substr(t1 + 1, t2 - t1 - 1).c_str(), nullptr, 16);
    item.bodies = (uint8_t)std::strtoul(line.substr(t2 + 1, t3 - t2 - 1).c_str(), nullptr, 10);
    item.name = line.substr(t3 + 1);
    items_.push_back(std::move(item));
  }
  // Name order for the selection grids (the index file is guid-ordered).
  std::sort(items_.begin(), items_.end(),
            [](const ClosetItem& a, const ClosetItem& b) { return a.name < b.name; });
  for (size_t i = 0; i < items_.size(); ++i) {
    by_guid_[items_[i].id.to_string()] = i;
  }
  is_loaded_ = true;
  LoadAwardDetails();
  return true;
}

// Reads a whole text file; empty on failure.
static std::string ReadTextFile(const std::filesystem::path& path) {
  FILE* f = std::fopen(path.string().c_str(), "rb");
  if (!f) return std::string();
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string text((size_t)std::max(0L, sz), '\0');
  size_t got = std::fread(text.data(), 1, text.size(), f);
  std::fclose(f);
  text.resize(got);
  return text;
}

// Splits TSV text into rows of fields; tolerates a UTF-8 BOM and CRLF.
static std::vector<std::vector<std::string>> ParseTsv(const std::string& text) {
  std::vector<std::vector<std::string>> rows;
  size_t pos = text.rfind("\xEF\xBB\xBF", 0) == 0 ? 3 : 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<std::string> fields;
    size_t start = 0;
    for (;;) {
      size_t tab = line.find('\t', start);
      fields.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
      if (tab == std::string::npos) break;
      start = tab + 1;
    }
    rows.push_back(std::move(fields));
  }
  return rows;
}

// Award items: classify by the id's provenance nibble, then attach the
// per-game details the editor's Awards page shows (closet_awards.tsv /
// closet_titles.tsv, see closet.h). Everything is optional: an award with no
// sidecar row still lists under its title id with the bin's mtime as the
// date and an empty "reason".
void Closet::LoadAwardDetails() {
  std::unordered_map<uint32_t, std::string> title_names;
  for (const auto& row : ParseTsv(ReadTextFile(dir_ / "closet_titles.tsv"))) {
    if (row.size() >= 2) {
      title_names[(uint32_t)std::strtoul(row[0].c_str(), nullptr, 16)] = row[1];
    }
  }
  struct AwardRow {
    std::string title_name, description;
    uint64_t unlock_time = 0;
  };
  std::unordered_map<std::string, AwardRow> award_rows;
  for (const auto& row : ParseTsv(ReadTextFile(dir_ / "closet_awards.tsv"))) {
    if (row.empty() || row[0].size() != 36) continue;
    AwardRow r;
    if (row.size() >= 3) r.title_name = row[2];
    if (row.size() >= 4) r.description = row[3];
    if (row.size() >= 5) r.unlock_time = std::strtoull(row[4].c_str(), nullptr, 10);
    std::string guid = row[0];
    for (auto& c : guid) c = (char)std::tolower((unsigned char)c);
    award_rows[guid] = std::move(r);
  }
  for (auto& item : items_) {
    if (!IsAwardId(item.id)) continue;
    item.is_award = true;
    item.title_id = TitleIdOf(item.id);
    auto it = award_rows.find(item.id.to_string());
    if (it != award_rows.end()) {
      item.title_name = it->second.title_name;
      item.description = it->second.description;
      item.unlock_time = it->second.unlock_time;
    }
    if (item.title_name.empty()) {
      auto tn = title_names.find(item.title_id);
      if (tn != title_names.end()) item.title_name = tn->second;
    }
    if (item.title_name.empty()) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%08X", item.title_id);
      item.title_name = buf;
    }
    if (!item.unlock_time) {
      // std::filesystem::file_time_type on MSVC counts FILETIME ticks.
      std::error_code ec;
      auto ft = std::filesystem::last_write_time(dir_ / (item.id.to_string() + ".bin"), ec);
      if (!ec) item.unlock_time = (uint64_t)ft.time_since_epoch().count();
    }
    ++award_count_;
  }
}

const ClosetItem* Closet::Find(const AssetId& id) const {
  if (items_.empty() || IsStockPackId(id)) {
    return nullptr;
  }
  auto it = by_guid_.find(id.to_string());
  return it == by_guid_.end() ? nullptr : &items_[it->second];
}

static bool ReadWholeFile(const std::filesystem::path& path, std::vector<uint8_t>& out) {
  FILE* f = std::fopen(path.string().c_str(), "rb");
  if (!f) {
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  out.resize((size_t)std::max(0L, sz));
  size_t got = std::fread(out.data(), 1, out.size(), f);
  std::fclose(f);
  return sz > 0 && got == out.size();
}

bool Closet::ReadItemBytes(const AssetId& id, std::vector<uint8_t>& out) const {
  const ClosetItem* item = Find(id);
  if (!item) {
    return false;
  }
  const std::filesystem::path path = dir_ / (id.to_string() + ".bin");
  if (!ReadWholeFile(path, out)) {
    REXKRNL_WARN("[avatar] closet item file missing: {}", path.string());
    return false;
  }
  return true;
}

bool Closet::ReadItemIcon(const AssetId& id, std::vector<uint8_t>& out) const {
  const ClosetItem* item = Find(id);
  if (!item) {
    return false;
  }
  // Missing icon art is the normal case until --closet-icons has run against
  // the raw marketplace archive; stay quiet about it.
  return ReadWholeFile(dir_ / "icons" / (id.to_string() + ".png"), out);
}

bool Closet::ReadTitleIcon(uint32_t title_id, std::vector<uint8_t>& out) const {
  if (!is_loaded_) {
    return false;
  }
  // The Dry Cleaner stores whatever the user picked (png/jpg/bmp/gif); the
  // tile server decodes by content and re-encodes to PNG.
  static const char* const kExts[] = {"png", "jpg", "jpeg", "bmp", "gif"};
  for (const char* ext : kExts) {
    char name[32];
    std::snprintf(name, sizeof(name), "%08X.%s", title_id, ext);
    if (ReadWholeFile(dir_ / "titles" / name, out)) {
      return true;
    }
  }
  // No art for this title: serve titles/_default.* when the user provides one.
  // The catalog popup's logo element only repaints when a tile loads, so a
  // FILE_NOT_FOUND here leaves the previous game's logo on screen; a neutral
  // default keeps a missing logo from standing in for another title. Without
  // the file, behavior is unchanged.
  for (const char* ext : kExts) {
    char name[32];
    std::snprintf(name, sizeof(name), "_default.%s", ext);
    if (ReadWholeFile(dir_ / "titles" / name, out)) {
      return true;
    }
  }
  return false;
}

bool EncodePngRgb(const uint8_t* rgba, uint32_t width, uint32_t height,
                  uint32_t stride, std::vector<uint8_t>& out) {
  if (!rgba || !width || !height || stride < width * 4) {
    return false;
  }
  std::vector<uint8_t> rgb(size_t(width) * height * 3);
  for (uint32_t y = 0; y < height; ++y) {
    const uint8_t* src = rgba + size_t(y) * stride;
    uint8_t* dst = &rgb[size_t(y) * width * 3];
    for (uint32_t x = 0; x < width; ++x, src += 4, dst += 3) {
      dst[0] = src[0];
      dst[1] = src[1];
      dst[2] = src[2];
    }
  }
  int len = 0;
  unsigned char* png = stbi_write_png_to_mem(rgb.data(), (int)(width * 3), (int)width,
                                             (int)height, 3, &len);
  if (!png || len <= 0) {
    return false;
  }
  out.assign(png, png + len);
  STBIW_FREE(png);
  return true;
}

bool EncodePng(const uint8_t* rgba, uint32_t width, uint32_t height,
               std::vector<uint8_t>& out) {
  int len = 0;
  unsigned char* png = stbi_write_png_to_mem(rgba, (int)(width * 4), (int)width,
                                             (int)height, 4, &len);
  if (!png || len <= 0) {
    return false;
  }
  out.assign(png, png + len);
  STBIW_FREE(png);
  return true;
}

Closet& GetCloset() {
  static Closet closet;
  return closet;
}

}  // namespace avatars
}  // namespace rex
