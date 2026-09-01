// Guest-side patches for running outside the console environment.

#include <atomic>
#include <cstdint>
#include <cstring>

#include "generated/ae/ae_init.h"

#include <rex/kernel/xam/tile_icon.h>
#include <rex/system/kernel_state.h>

// The save flow polls a GamerPicManager op that isn't available offline;
// r3 == 0 doubles as the idle result for both pollers.
bool AE_GuardNullOpState(PPCRegister& r3) {
  return r3.u32 == 0;
}

bool AE_GuardNullOpActive(PPCRegister& r3) {
  return r3.u32 == 0;
}

// Grid tiles need the 128px icon; the awards highlight box draws at native
// size and only has room for 64px.
void AE_TileSizeHint(PPCRegister& r3) {
  auto* mem = rex::system::kernel_memory();
  const auto* p = mem->TranslateVirtual<const uint8_t*>(r3.u32 - 36);
  if (!p) {
    return;
  }
  char name[8] = {};
  for (int i = 0; i < 7; i++) {
    const uint8_t hi = p[i * 2], lo = p[i * 2 + 1];
    if (hi || lo < 0x20 || lo >= 0x7F) {
      break;
    }
    name[i] = char(lo);
  }
  rex::kernel::xam::SetTileSizeHint(std::strncmp(name, "PREVIEW", 7) == 0 ? 128u : 64u);
}

// The 16KB icon buffer is too small for 128px PNGs; the type 15 path
// already uses 64KB.
void AE_TileBufferSize(PPCRegister& r11) {
  if (r11.u32 == 0x4000) {
    r11.u32 = 0x10000;
  }
}
