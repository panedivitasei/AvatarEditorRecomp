/**
 ******************************************************************************
 * ReXGlue avatar closet: imported Xbox LIVE marketplace avatar items.       *
 ******************************************************************************
 *
 * A closet directory holds one <product-guid>.bin (the marketplace item's raw
 * YTGR/STRB asset_v2.bin) per item plus closet_index.tsv
 * (guid TAB categories-hex TAB bodies TAB name), produced by
 * `avatarextract --closet-import`. Items are enumerated into the Avatar
 * Editor's selection grids alongside the stock asset pack and resolve by their
 * full product GUID; stock pack ids resolve by pack index and always carry the
 * C1C8F109A19CB2E0 tail, while marketplace ids end in the item's title id, so
 * the two populations cannot collide.
 *
 * Awards (items a game granted) are told apart from purchases by the asset
 * id's provenance nibble, ((id.c >> 8) & 0xF): 1 = award, 2 = marketplace.
 * The granting title is id.d[4..7].
 *
 * Optional art and award metadata sit alongside the .bin files:
 *   icons/<guid>.png   the marketplace package's ICON.PNG, handed to the
 *                      editor verbatim by XamAvatarGetAssetIcon (its icon
 *                      pipeline consumes image files, not pixels)
 *   closet_awards.tsv  guid TAB title-id-hex TAB title name TAB description
 *                      TAB unlock FILETIME (decimal; 0/missing = bin mtime)
 *   closet_titles.tsv  title-id-hex TAB title name   (fallback game names)
 *   titles/<TITLEID>.png|jpg|jpeg|bmp|gif  the granting game's tile icon, any
 *                      size; the tile server resamples to the largest square
 *                      the editor's 16 KB buffer fits and re-encodes to PNG
 */

#ifndef REX_AVATARS_CLOSET_H_
#define REX_AVATARS_CLOSET_H_

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "asset_pack.h"

namespace rex {
namespace avatars {

struct ClosetItem {
  AssetId id;
  uint32_t categories;
  uint8_t bodies;
  std::string name;  // UTF-8 display name for the selection grids
  // Avatar awards (provenance nibble 1): per-game details for the editor's
  // Awards page. title_name/description are UTF-8.
  bool is_award = false;
  uint32_t title_id = 0;      // granting title, from id.d[4..7]
  std::string title_name;     // closet_awards.tsv, else closet_titles.tsv, else ""
  std::string description;    // "how it was earned"; closet_awards.tsv, else ""
  uint64_t unlock_time = 0;   // FILETIME; closet_awards.tsv, else the bin's mtime
};

// Reads the canonical 8-4-4-4-12 GUID text an item id is written as, the
// inverse of AssetId::to_string(). False on any malformed field.
bool ParseAssetId(const std::string& text, AssetId* out);

// True when the asset id carries the award provenance nibble.
bool IsAwardId(const AssetId& id);
// The granting title id encoded in an award/marketplace id (d[4..7]).
uint32_t TitleIdOf(const AssetId& id);

class Closet {
 public:
  // Loads closet_index.tsv from the directory. Missing dir/index = empty
  // closet (not an error). Safe to call repeatedly; loads once.
  bool Load(const std::filesystem::path& dir);

  bool is_loaded() const { return is_loaded_; }
  const std::vector<ClosetItem>& items() const { return items_; }
  size_t award_count() const { return award_count_; }

  const ClosetItem* Find(const AssetId& id) const;

  // Reads the item's raw YTGR/STRB blob from disk, or from the custom
  // registry when the title handed the bytes over itself.
  bool ReadItemBytes(const AssetId& id, std::vector<uint8_t>& out) const;
  // Whether the bytes are at hand (custom map or file), without touching the
  // index tables the guest thread may be rebuilding.
  bool HasItemBytes(const AssetId& id) const;

  // Keeps a blob the title pushed through XamAvatarSetCustomAsset, or one the
  // store fetched, so the loader resolves its id like any closet item; lives
  // until exit. RegisterCustomIcon is the same for art.
  void RegisterCustomItem(const AssetId& id, const uint8_t* data, size_t size);
  void RegisterCustomIcon(const AssetId& id, const uint8_t* data, size_t size);

  // A purchase landing: writes <guid>.bin and icons/<guid>.png, appends the
  // closet_index.tsv row and lists the item, so it is owned and in the
  // wardrobe from this call on.
  bool InstallItem(const AssetId& id, const std::vector<uint8_t>& blob,
                   const std::vector<uint8_t>& icon, uint32_t categories, uint8_t bodies,
                   const std::string& name);

  // Reads the item's imported marketplace icon (icons/<guid>.png) as raw PNG
  // file bytes. False when the item exists but carries no icon art.
  bool ReadItemIcon(const AssetId& id, std::vector<uint8_t>& out) const;

  // Reads a game's tile icon (titles/<TITLEID>.png, upper-case hex) as raw
  // image file bytes. False when no art is stored for the title.
  bool ReadTitleIcon(uint32_t title_id, std::vector<uint8_t>& out) const;

 private:
  void LoadAwardDetails();

  bool is_loaded_ = false;
  size_t award_count_ = 0;
  std::filesystem::path dir_;
  std::vector<ClosetItem> items_;
  std::unordered_map<std::string, size_t> by_guid_;  // guid string -> index
  mutable std::mutex custom_mutex_;
  std::unordered_map<std::string, std::vector<uint8_t>> custom_items_;  // guid -> blob
  std::unordered_map<std::string, std::vector<uint8_t>> custom_icons_;  // guid -> png
};

// Global closet instance, loaded next to the avatar asset pack.
Closet& GetCloset();

// True when the id carries the stock asset pack's GUID tail (resolvable by
// pack index); false for marketplace/closet product ids.
bool IsStockPackId(const AssetId& id);

// Encode an RGBA8 image as a PNG file image. The Avatar Editor's icon pipeline
// only consumes image files (it magic-dispatches on PNG/JPEG/' RGB' headers),
// so any generated icon has to go through these.
// EncodePngRgb drops alpha from RGBA rows of `stride` bytes: the gamer picture
// the editor composites is opaque, and an RGB PNG keeps a 32x32/64x64 icon
// inside the title's 0x1000/0x4000 output buffers with stb's weak deflate.
bool EncodePngRgb(const uint8_t* rgba, uint32_t width, uint32_t height,
                  uint32_t stride, std::vector<uint8_t>& out);
bool EncodePng(const uint8_t* rgba, uint32_t width, uint32_t height,
               std::vector<uint8_t>& out);

}  // namespace avatars
}  // namespace rex

#endif  // REX_AVATARS_CLOSET_H_
