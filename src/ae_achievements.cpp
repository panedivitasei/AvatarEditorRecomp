#include "ae_achievements.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/system/achievements.h>
#include <rex/system/kernel_state.h>

#include "kernel/xam/avatars/guest_asset.h"

namespace ae::achievements {
namespace {

namespace cat = rex::avatars::ComponentCategory;

constexpr uint32_t kOutfitCategories = cat::kTop | cat::kBottom | cat::kShoes | cat::kHat |
                                       cat::kGloves | cat::kGlasses | cat::kWristwear |
                                       cat::kEarrings | cat::kRing;
constexpr size_t kClosetHoard = 25;
constexpr uint32_t kGamerscore = 100;
// textures[] slots 0..2 are mouth, eyes and brows (guest_load_asset.cpp).
constexpr size_t kFaceTextureSlots = 3;

struct Entry {
  Id id;
  const char* label;
  const char* description;
};

constexpr Entry kCatalog[] = {
    {kHello, "Hello!", "Open the Avatar Editor for the first time."},
    {kFreshFit, "Fresh Fit", "Change your avatar's outfit and save it."},
    {kWindowShopper, "Window Shopper", "Connect to the marketplace for the first time."},
    {kTreatYourself, "Treat Yourself", "Get an item from the marketplace."},
    {kGlowUp, "Glow Up", "Change your avatar's face."},
    {kClosetHoarder, "Closet Hoarder", "Own 25 items in your closet."},
    {kStrikeAPose, "Strike a Pose", "Try out a prop on your avatar."},
};

std::vector<std::string> OutfitOf(const rex::avatars::X_AVATAR_METADATA& meta) {
  std::vector<std::string> ids;
  for (const auto& comp : meta.components) {
    if (comp.matches(kOutfitCategories)) {
      ids.push_back(comp.asset_id.to_string());
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

bool FaceChanged(const rex::avatars::X_AVATAR_METADATA& before,
                 const rex::avatars::X_AVATAR_METADATA& after) {
  for (size_t i = 0; i < kFaceTextureSlots; ++i) {
    if (before.textures[i].asset_id != after.textures[i].asset_id) {
      return true;
    }
  }
  for (size_t i = 0; i < std::size(before.blend_shapes); ++i) {
    if (before.blend_shapes[i].asset_id != after.blend_shapes[i].asset_id) {
      return true;
    }
  }
  return false;
}

std::string TomlString(const char* text) {
  std::string out = "\"";
  for (const char* c = text; *c; ++c) {
    if (*c == '"' || *c == '\\') {
      out += '\\';
    }
    out += *c;
  }
  return out + "\"";
}

// JMprofile lists a game's achievements from <titleid>.catalog.toml beside the SDK's unlock file.
void WriteCatalog() {
  auto* kernel = REX_KERNEL_STATE();
  if (!kernel || kernel->emulator()->user_data_root().empty()) {
    return;
  }
  const auto dir = kernel->emulator()->user_data_root() / "achievements";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  std::ofstream out(dir / fmt::format("{:08X}.catalog.toml", kernel->title_id()), std::ios::binary);
  out << "title = \"Avatar Editor\"\n";
  for (const auto& entry : kCatalog) {
    out << "\n[[achievements]]\n"
        << "id = " << uint32_t(entry.id) << "\n"
        << "label = " << TomlString(entry.label) << "\n"
        << "description = " << TomlString(entry.description) << "\n"
        << "unachieved_description = " << TomlString(entry.description) << "\n"
        << "gamerscore = " << kGamerscore << "\n";
  }
  // The game tile beside it is the title icon from the XEX's XDBF.
  const auto db = kernel->title_xdbf();
  if (const auto icon = db.is_valid() ? db.icon() : rex::system::util::XdbfBlock{}; icon) {
    std::ofstream png(dir / fmt::format("{:08X}.png", kernel->title_id()), std::ios::binary);
    png.write(reinterpret_cast<const char*>(icon.buffer), icon.size);
  }
}

}  // namespace

void Register() {
  for (const auto& entry : kCatalog) {
    rex::system::AchievementInfo info;
    info.id = entry.id;
    info.label = entry.label;
    info.description = entry.description;
    info.unachieved_description = entry.description;
    info.gamerscore = kGamerscore;
    rex::system::RegisterAchievement(std::move(info));
  }
  WriteCatalog();
}

void Unlock(Id id) {
  if (rex::system::UnlockAchievement(id)) {
    REXLOG_INFO("[achievements] unlocked {:#x}", uint32_t(id));
  }
}

void OnManifestSaved(const rex::avatars::X_AVATAR_METADATA& before,
                     const rex::avatars::X_AVATAR_METADATA& after) {
  if (OutfitOf(before) != OutfitOf(after)) {
    Unlock(kFreshFit);
  }
  if (FaceChanged(before, after)) {
    Unlock(kGlowUp);
  }
}

void OnClosetCount(size_t count) {
  if (count >= kClosetHoard) {
    Unlock(kClosetHoarder);
  }
}

}  // namespace ae::achievements
