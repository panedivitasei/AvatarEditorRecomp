/**
 * @file        marketplace.h
 * @brief       The store's server address, parsed once from avatar_marketplace_server,
 *              and the title-side fetches that go to it: item art, item packages,
 *              and the install that a purchase turns into.
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 * @license     BSD 3-Clause License, see LICENSE in the project root.
 */

#ifndef REX_KERNEL_XAM_MARKETPLACE_H_
#define REX_KERNEL_XAM_MARKETPLACE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace rex {
namespace avatars {
struct AssetId;
}  // namespace avatars

namespace kernel {
namespace xam {

struct MarketplaceServer {
  std::string host;      // lowercase, no scheme or port
  uint16_t port = 80;
  bool secure = false;   // https://
  std::string base;      // "http://host:port", the form the title's own URL parsers accept
  bool valid = false;
};

// Parsed from avatar_marketplace_server on every call, so a config edit is
// picked up by the next request.
MarketplaceServer GetMarketplaceServer();

// Blocking GET of <server><path>. True on a 200 with the body filled in; the
// raw CRLF header block is handed back too when asked for.
bool MarketplaceGet(const std::string& path, std::string* body, std::string* headers = nullptr);

// Downloads an item the closet does not hold and installs it: blob, icon and
// index row. True when the item is in the closet afterwards.
bool MarketplaceInstallItem(const avatars::AssetId& id);

// Fetches the tile art of the given item guids in parallel, skipping what the
// closet already has. With wait set it returns once the art is in; a catalog
// page is handed to the title only after that, so its grid opens complete.
void MarketplacePrefetchIcons(const std::vector<std::string>& guids, bool wait);

// Fetches the item packages of the given guids in the background, for the
// hover try-on, skipping what the closet already has.
void MarketplacePrefetchItems(const std::vector<std::string>& guids);

}  // namespace xam
}  // namespace kernel
}  // namespace rex

#endif  // REX_KERNEL_XAM_MARKETPLACE_H_
