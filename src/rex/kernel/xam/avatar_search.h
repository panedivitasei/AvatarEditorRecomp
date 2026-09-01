/**
 * @file        rex/kernel/xam/avatar_search.h
 * @brief       Avatar catalog search filter (the Ctrl+F item search).
 *
 * A host-side, case-insensitive substring filter over item display names,
 * applied inside XamAvatarBeginEnumAssets: while a query is set, only matching
 * items in the scoped catalog channel are enumerated (other categories pass
 * through unfiltered), so the browsed grid shows exactly the matches. The guest
 * enumerates only during its catalog rebuild-all, so the title tick
 * (video_hooks.cpp AeXuiSearchTick) drives that rebuild on apply/clear and then
 * latches the registry's dirty flags so the open grid re-pushes natively.
 *
 * Set/cleared by the Ctrl+F overlay (ui side, wired through ReXApp);
 * consumed by xam_avatar.cpp. Thread-safe.
 *
 * @copyright   Copyright (c) 2026. BSD 3-Clause (see LICENSE).
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rex::kernel::xam {

// Set the active catalog filter (UTF-8; empty clears). Matching is a
// case-insensitive substring test against item display names.
void SetAvatarCatalogSearch(const std::string& query_utf8);
std::string GetAvatarCatalogSearch();

// Scope: the guest catalog channel (0..24, the grid's category bucket) the
// filter applies to; -1 = unscoped. Captured from the open grid screen by the
// title's per-frame tick (the same vtbl+52/+56 test the guest's own
// content-changed pump performs) and consumed by the enumeration filter and
// QueryAvatarCatalog, so a search only affects the catalog being browsed and
// items in every other category enumerate unfiltered.
void SetAvatarCatalogSearchScope(int channel);
int GetAvatarCatalogSearchScope();

// Matches from the most recent enumeration that ran with a non-empty filter
// (for the overlay's status line). Reset on each filtered enumeration.
uint32_t GetAvatarCatalogSearchMatches();

// Live catalog query for the search overlay's results list: case-
// insensitive substring over pack + closet item names. Returns up to
// max_results matches (name, category bits, closet origin) plus the total
// match count (which may exceed the returned rows).
struct AvatarCatalogMatch {
  std::string name;
  uint32_t categories = 0;
  bool from_closet = false;
};
uint32_t QueryAvatarCatalog(const std::string& query_utf8, size_t max_results,
                            std::vector<AvatarCatalogMatch>& out);

}  // namespace rex::kernel::xam
