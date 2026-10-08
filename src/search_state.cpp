// search_state.cpp: mutex-guarded Ctrl+F state, see search_state.h.
#include "search_state.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace ae_search {

namespace {

std::mutex g_srchMu;
bool g_srchOpen = false;
std::string g_srchQuery;
std::vector<std::string> g_srchLines;
bool g_srchPossessed = false;  // the in-game XUI label carries the bar line
bool g_srchApplied = false;    // filter armed (post-Enter)
std::atomic<bool> g_srchRebuild{false};

constexpr int kSrchMaxLines = 10;  // bar + up to 9 result rows

}  // namespace

void SetSearchOverlay(bool open, const char* query, const char* const* lines, int line_count) {
  std::lock_guard<std::mutex> lk(g_srchMu);
  g_srchOpen = open;
  g_srchQuery = query ? query : "";
  g_srchLines.clear();
  for (int i = 0; i < line_count && i < kSrchMaxLines - 1; i++) {
    g_srchLines.emplace_back(lines && lines[i] ? lines[i] : "");
  }
}

void SetSearchApplied(bool applied) {
  std::lock_guard<std::mutex> lk(g_srchMu);
  g_srchApplied = applied;
}

void RequestCatalogRebuild() {
  g_srchRebuild.store(true, std::memory_order_release);
}

bool ConsumeCatalogRebuildRequest() {
  return g_srchRebuild.exchange(false, std::memory_order_acq_rel);
}

void SetSearchLabelPossessed(bool possessed) {
  std::lock_guard<std::mutex> lk(g_srchMu);
  g_srchPossessed = possessed;
}

int GetSearchStatusLine(char* utf8, int cap) {
  std::lock_guard<std::mutex> lk(g_srchMu);
  if (!utf8 || cap < 8) return 0;
  // Applied but closed: the heading says the filter is live and how to edit it.
  if (!g_srchOpen && g_srchApplied && !g_srchQuery.empty()) {
    const int m = g_srchLines.empty()
                      ? std::snprintf(utf8, size_t(cap), "Filter: %s  (Ctrl+F to edit)",
                                      g_srchQuery.c_str())
                      : std::snprintf(utf8, size_t(cap), "Filter: %s  (%s, Ctrl+F to edit)",
                                      g_srchQuery.c_str(), g_srchLines[0].c_str());
    return (m > 0 && m < cap) ? m : (m > 0 ? cap - 1 : 0);
  }
  if (!g_srchOpen) return 0;
  // Query plus match count only, the matches show in the grid through the filter.
  int n = std::snprintf(utf8, size_t(cap), "Search: %s_", g_srchQuery.c_str());
  if (n < 0) return 0;
  if (!g_srchQuery.empty() && !g_srchLines.empty() && n < cap) {
    n += std::snprintf(utf8 + n, size_t(cap - n), "  (%s)", g_srchLines[0].c_str());
  }
  if (n < 0) return 0;
  return n < cap ? n : cap - 1;
}

}  // namespace ae_search
