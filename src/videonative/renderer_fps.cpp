// rexvideonative, FPS overlay rasterizer (see renderer_fps.h).
// GPL-3.0, see LICENSE in this directory.

#include "renderer_fps.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <rex/cvar.h>

REXCVAR_DEFINE_BOOL(native_video_fps, false, "GPU",
                    "Show an FPS counter in the top-left corner when the "
                    "native video layer is presenting.");

namespace rex::videonative::fps {
namespace {

// ---- builtin 5x7 font ------------------------------------------------------
// One byte per row, low 5 bits used (bit 4 = leftmost column). Coverage:
// digits, '.', ' ', and "FPS".
struct BuiltinGlyph {
  char ch;
  uint8_t rows[7];
};
constexpr BuiltinGlyph kBuiltinFont[] = {
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}},
    {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}},
    {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}},
    {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}},
    {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}},
    {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}},
    {' ', {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
    {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}},
    {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
    {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}},
};

const uint8_t* BuiltinRows(char ch) {
  for (const BuiltinGlyph& g : kBuiltinFont) {
    if (g.ch == ch) return g.rows;
  }
  return nullptr;  // unknown chars render as space
}

constexpr uint32_t kMaxChars = 9;  // "999.9 FPS"
constexpr uint32_t kGlyphW = 5, kGlyphH = 7;
constexpr uint32_t kAdvance = kGlyphW + 1;
constexpr uint32_t kPad = 2;
constexpr uint32_t kFbWidth = kMaxChars * kAdvance - 1 + 2 * kPad;  // 55
constexpr uint32_t kFbHeight = kGlyphH + 2 * kPad;                  // 11

// ---- shared strip state ----------------------------------------------------
bool g_fontTried = false;
uint32_t g_width = 0, g_height = 0;
uint32_t g_usedWidth = 0;
std::vector<uint8_t> g_pixels;
std::vector<float> g_cov;  // glyph coverage, composited by FinalizeOutline
std::string g_lastText;

// Turns the coverage buffer into the final RGBA strip: transparent
// background, black outline (coverage dilated by `radius`), white core.
// The overlay pipeline alpha-blends, so no backdrop box is needed for
// legibility, the outline carries it over bright scenes.
void FinalizeOutline(int radius) {
  for (int y = 0; y < int(g_height); y++) {
    for (int x = 0; x < int(g_width); x++) {
      const float core = g_cov[size_t(y) * g_width + x];
      float dilated = core;
      for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
          const int sx = x + dx, sy = y + dy;
          if (sx < 0 || sy < 0 || sx >= int(g_width) || sy >= int(g_height)) {
            continue;
          }
          dilated = std::max(dilated, g_cov[size_t(sy) * g_width + sx]);
        }
      }
      uint8_t* px = &g_pixels[(size_t(y) * g_width + x) * 4];
      const float alpha = std::max(core, dilated * 0.85f);
      if (alpha <= 0.004f) {
        px[0] = px[1] = px[2] = px[3] = 0;
        continue;
      }
      const uint8_t white = uint8_t(std::lround(255.0f * (core / alpha)));
      px[0] = px[1] = px[2] = white;
      px[3] = uint8_t(std::lround(255.0f * alpha));
    }
  }
}

// Left-aligned: the leading edge stays put as the digit count changes.
void RasterizeBuiltin(const std::string& text) {
  std::fill(g_cov.begin(), g_cov.end(), 0.0f);
  const uint32_t count = std::min<uint32_t>(uint32_t(text.size()), kMaxChars);
  uint32_t x = kPad;
  for (uint32_t c = 0; c < count; c++, x += kAdvance) {
    const uint8_t* rows = BuiltinRows(text[c]);
    if (!rows) continue;
    for (uint32_t gy = 0; gy < kGlyphH; gy++) {
      for (uint32_t gx = 0; gx < kGlyphW; gx++) {
        if (!(rows[gy] & (0x10 >> gx))) continue;
        g_cov[(kPad + gy) * g_width + x + gx] = 1.0f;
      }
    }
  }
  g_usedWidth = std::min(g_width, count * kAdvance - 1 + 2 * kPad);
  FinalizeOutline(1);
}

}  // namespace

Overlay Tick() {
  using Clock = std::chrono::steady_clock;
  static Clock::time_point last = Clock::now();
  static double ema_ms = 0.0;

  const Clock::time_point now = Clock::now();
  const double dt_ms =
      std::chrono::duration<double, std::milli>(now - last).count();
  last = now;
  // EMA over swap-to-swap time; clamp startup/pause outliers so the readout
  // recovers in a few frames instead of averaging a multi-second gap.
  if (dt_ms > 0.0 && dt_ms < 1000.0) {
    ema_ms = ema_ms == 0.0 ? dt_ms : ema_ms * 0.95 + dt_ms * 0.05;
  }

  Overlay out;
  if (!REXCVAR_GET(native_video_fps) || ema_ms <= 0.0) return out;

  // First visible tick latches the layout (the overlay texture is created
  // once at these dimensions).
  if (!g_fontTried) {
    g_fontTried = true;
    g_width = kFbWidth;
    g_height = kFbHeight;
    g_pixels.assign(size_t(g_width) * g_height * 4, 0);
    g_cov.assign(size_t(g_width) * g_height, 0.0f);
  }

  char text[16];
  const double fps_value = 1000.0 / ema_ms;
  if (fps_value >= 99.95) {
    std::snprintf(text, sizeof(text), "%.0f FPS", fps_value);
  } else {
    std::snprintf(text, sizeof(text), "%.1f FPS", fps_value);
  }
  out.visible = true;
  out.width = g_width;
  out.height = g_height;
  out.pixels = g_pixels.data();
  out.scale = 1.5f;
  out.dirty = g_lastText != text;
  if (out.dirty) {
    g_lastText = text;
    RasterizeBuiltin(g_lastText);
  }
  out.used_width = g_usedWidth ? g_usedWidth : g_width;
  return out;
}

// Catalog search state. The query and result lines are delivered
// through the game's own heading label.

namespace {

std::mutex g_srchMu;
bool g_srchOpen = false;
std::string g_srchQuery;
std::vector<std::string> g_srchLines;
bool g_srchDirty = false;

constexpr int kSrchMaxLines = 10;  // bar + up to 9 result rows

}  // namespace

void SetSearchOverlay(bool open, const char* query, const char* const* lines,
                      int line_count) {
  std::lock_guard<std::mutex> lk(g_srchMu);
  g_srchOpen = open;
  g_srchQuery = query ? query : "";
  g_srchLines.clear();
  for (int i = 0; i < line_count && i < kSrchMaxLines - 1; i++) {
    g_srchLines.emplace_back(lines && lines[i] ? lines[i] : "");
  }
  g_srchDirty = true;
}

namespace {
bool g_srchPossessed = false;  // the in-game XUI label carries the bar line
bool g_srchApplied = false;    // filter armed (post-Enter)
std::atomic<bool> g_srchRebuild{false};
}  // namespace

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
  if (g_srchPossessed != possessed) {
    g_srchPossessed = possessed;
    g_srchDirty = true;
  }
}

int GetSearchStatusLine(char* utf8, int cap) {
  std::lock_guard<std::mutex> lk(g_srchMu);
  if (!utf8 || cap < 8) return 0;
  // Applied-but-closed: the filter is in the grid, and the heading says
  // so. Ctrl+F reopens the box prefilled; leaving the catalog clears the
  // filter. Esc clears it as well. 
  if (!g_srchOpen && g_srchApplied && !g_srchQuery.empty()) {
    const int m = std::snprintf(utf8, size_t(cap),
                                "Filter: %s  (Ctrl+F to edit)",
                                g_srchQuery.c_str());
    return (m > 0 && m < cap) ? m : (m > 0 ? cap - 1 : 0);
  }
  if (!g_srchOpen) return 0;
  // The possessed heading line: query + match count only; the matches
  // themselves show in the navigator through the grid filter.
  int n = std::snprintf(utf8, size_t(cap), "Search: %s_", g_srchQuery.c_str());
  if (n < 0) return 0;
  if (!g_srchQuery.empty() && !g_srchLines.empty()) {
    n += std::snprintf(utf8 + n, size_t(cap - n), "  (%s)",
                       g_srchLines[0].c_str());
  }
  if (n < 0) return 0;
  return n < cap ? n : cap - 1;
}

Overlay SearchTick() {
  // The strip is not drawn: the in-game heading label carries the whole
  // search line (see GetSearchStatusLine).
  return Overlay{};
}

}  // namespace rex::videonative::fps
