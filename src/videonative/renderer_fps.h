// rexvideonative, FPS overlay for the native path.

#pragma once

#include <cstdint>

namespace rex::videonative::fps {

// Result of one presentation tick. `pixels` is an RGBA8 buffer owned by the
// module, valid until the next Tick(); re-upload to the overlay texture only
// when `dirty`.
struct Overlay {
  bool visible = false;
  uint32_t width = 0;
  uint32_t height = 0;
  const uint8_t* pixels = nullptr;
  bool dirty = false;
  // Draw scale for the strip (the 5x7 font renders at 1.5x).
  float scale = 1.0f;
  // Pixels of the strip covered by text + padding; the renderer
  // scissors the box to this so short readouts don't trail dead backdrop.
  uint32_t used_width = 0;
};

// Call once per SwapFrontbuffer: advances the swap-to-swap clock and, when the
// native_video_fps cvar is on, rasterizes "<fps> FPS" into the pixel buffer.
Overlay Tick();

// Search feature
// State pushed by the app-side controller; rendered on native path. 
// Replace the overlay state. lines = result rows under the search bar.
void SetSearchOverlay(bool open, const char* query, const char* const* lines,
                      int line_count);

// Per-present tick for the search overlay strip (same contract as Tick()).
Overlay SearchTick();

// One-line status for the possessed in-game XUI label ("Search: q_ (N)").
// Returns the byte length written (0 when the search is closed).
int GetSearchStatusLine(char* utf8, int cap);

// Applied-filter state: after Enter the typing closes but the filter is
// armed, the heading shows "Filter: q (Ctrl+F to clear)" until cleared.
void SetSearchApplied(bool applied);

// Catalog rebuild request bridge (app -> the title's Swap-frame tick,
// which invokes the guest's own rebuild-all worker entry).
void RequestCatalogRebuild();
bool ConsumeCatalogRebuildRequest();

// The in-game label is live: the fallback strip suppresses its own bar
// line 
void SetSearchLabelPossessed(bool possessed);

}  // namespace rex::videonative::fps
