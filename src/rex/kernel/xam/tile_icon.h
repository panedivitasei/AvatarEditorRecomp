/**
 * @file        rex/kernel/xam/tile_icon.h
 * @brief       Per-requester size hint for XamReadTile/XamReadTileEx game icons.
 *
 * The runtime serves a title's closet icon (titles/<TITLEID>.<ext>) as a square
 * PNG. Console title icons were 64x64, and the Avatar Editor's per-game Awards
 * "Highlight image" box draws the texture at native size, so a larger tile is
 * cropped there; the Awards grid scales its tiles instead and looks soft at 64.
 * Only the title knows which element is asking, so a title hook sets the
 * preferred size on the requesting thread right before the guest calls
 * XamReadTileEx, and the serve consumes it.
 *
 * @copyright   Copyright (c) 2026. BSD 3-Clause (see LICENSE).
 */
#pragma once

#include <cstdint>

namespace rex::kernel::xam {

// Preferred square size in px for the next game-icon serve on this thread
// (0 = default 64). 128 enables the 128 -> 96 -> 64 ladder (largest PNG that
// fits the caller's buffer; never upscaled). Consumed (reset) by that serve.
void SetTileSizeHint(uint32_t px);

}  // namespace rex::kernel::xam
