// catalog_search.cpp: the Ctrl+F item finder's input side. See the header.
#include "catalog_search.h"

#include <atomic>
#include <cstdio>
#include <vector>

#include <rex/kernel/xam/avatar_search.h>
#include "kernel/xam/marketplace.h"
#include <rex/logging.h>
#include <rex/ui/virtual_key.h>

#include "videonative/renderer_fps.h"

namespace ae_search {

namespace {
// Above the ImGui drawer and the input drivers, which register low.
constexpr size_t kTopmost = ~size_t(0);
constexpr size_t kMaxQuery = 96;
std::atomic<bool> g_games_list_open{false};
std::atomic<bool> g_games_reload{false};
}  // namespace

void SetGamesListOpen(bool open) { g_games_list_open.store(open, std::memory_order_relaxed); }
bool GamesListOpen() { return g_games_list_open.load(std::memory_order_relaxed); }
bool ConsumeGamesReloadRequest() { return g_games_reload.exchange(false); }

Controller& Get() {
  static Controller controller;
  return controller;
}

void Controller::Attach(rex::ui::Window* window) {
  window_ = window;
  if (window_) {
    window_->AddInputListener(this, kTopmost);
  }
}

void Controller::Detach() {
  if (window_) {
    if (open_) {
      window_->SetTextInputActive(false);
    }
    window_->RemoveInputListener(this);
    window_ = nullptr;
  }
  open_ = false;
}

// The heading line: query and match count, then a few names for the strip.
void Controller::Push() {
  std::vector<std::string> lines;
  // The game list lives on the server, so there is no local match list.
  if (!query_.empty() && !games_mode_) {
    std::vector<rex::kernel::xam::AvatarCatalogMatch> matches;
    const uint32_t total = rex::kernel::xam::QueryAvatarCatalog(query_, 8, matches);
    char head[64];
    std::snprintf(head, sizeof(head), "%u match%s", total, total == 1 ? "" : "es");
    lines.emplace_back(head);
    for (auto& m : matches) {
      lines.emplace_back(m.from_closet ? m.name + "  (closet)" : m.name);
    }
    if (total > matches.size()) {
      lines.emplace_back("...");
    }
  }
  std::vector<const char*> line_ptrs;
  line_ptrs.reserve(lines.size());
  for (const auto& s : lines) line_ptrs.push_back(s.c_str());
  rex::videonative::fps::SetSearchOverlay(open_, query_.c_str(), line_ptrs.data(),
                                          int(line_ptrs.size()));
}

void Controller::Toggle() {
  // A catalog grid or the Game Styles list can be searched; elsewhere the
  // key does nothing.
  if (!open_) {
    const bool items = rex::kernel::xam::GetAvatarCatalogSearchScope() >= 0;
    if (!items && !GamesListOpen()) {
      return;
    }
    games_mode_ = !items;
  }
  open_ = !open_;
  // The title tick clears an applied filter when the user leaves the page it
  // was applied in; pick that up here.
  const bool live = games_mode_ ? !rex::kernel::xam::MarketplaceGamesFilter().empty()
                                : !rex::kernel::xam::GetAvatarCatalogSearch().empty();
  if (applied_ && !live) {
    applied_ = false;
  }
  // A fresh box, unless a filter is live: then it opens prefilled for editing.
  if (open_ && !applied_) {
    query_.clear();
  }
  if (window_) {
    window_->SetTextInputActive(open_);
  }
  Push();
}

// Enter arms the filter and asks the tick for the rebuild that re-populates
// the page. An empty query, or Esc, clears.
void Controller::Apply(bool clear) {
  if (clear) {
    query_.clear();
  }
  const bool arming = !query_.empty();
  const bool was_armed = applied_;
  applied_ = arming;
  rex::videonative::fps::SetSearchApplied(arming);
  if (games_mode_) {
    rex::kernel::xam::SetMarketplaceGamesFilter(query_);
    if (arming || was_armed) {
      g_games_reload.store(true);
    }
  } else {
    rex::kernel::xam::SetAvatarCatalogSearch(query_);
    if (arming || was_armed) {
      rex::videonative::fps::RequestCatalogRebuild();
    }
  }
  open_ = false;
  if (window_) {
    window_->SetTextInputActive(false);
  }
  Push();
}

void Controller::OnKeyDown(rex::ui::KeyEvent& e) {
  using rex::ui::VirtualKey;
  if (!open_) {
    if (e.is_ctrl_pressed() && e.virtual_key() == VirtualKey::kF) {
      Toggle();
      e.set_handled(true);
    }
    return;
  }
  switch (e.virtual_key()) {
    case VirtualKey::kReturn:
      Apply(false);
      break;
    case VirtualKey::kEscape:
      Apply(true);
      break;
    case VirtualKey::kBack:
      if (!query_.empty()) {
        query_.pop_back();
        Push();
      }
      break;
    case VirtualKey::kF:
      // Ctrl+F while typing cancels the box and keeps a live filter.
      if (e.is_ctrl_pressed()) {
        Toggle();
      }
      break;
    default:
      break;
  }
  // Whatever it was, it was typing: nothing below gets to treat it as a pad.
  e.set_handled(true);
}

void Controller::OnKeyUp(rex::ui::KeyEvent& e) {
  if (open_) {
    e.set_handled(true);
  }
}

void Controller::OnKeyChar(rex::ui::KeyEvent& e) {
  if (!open_) {
    return;
  }
  const uint32_t ch = uint32_t(e.virtual_key());  // the codepoint rides here
  if (ch >= 0x20 && ch < 0x7F && query_.size() < kMaxQuery) {
    query_.push_back(char(ch));
    Push();
  }
  e.set_handled(true);
}

}  // namespace ae_search
