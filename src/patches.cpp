// Guest-side patches for running outside the console environment.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "ae_achievements.h"
#include "generated/ae/ae_init.h"
#include "kernel/xam/avatars/closet.h"
#include "kernel/xam/marketplace.h"

#include <rex/cvar.h>
#include <rex/kernel/xam/tile_icon.h>
#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/types.h>

// The save flow polls a GamerPicManager op that isn't available offline;
// r3 == 0 doubles as the idle result.
bool AE_GuardNullOpActive(PPCRegister& r3) {
  return r3.u32 == 0;
}

// Guest memory access for the marketplace hooks.
namespace {

std::string MktGuestAnsi(uint32_t addr, size_t cap = 96) {
  const auto* p = rex::system::kernel_memory()->TranslateVirtual<const char*>(addr);
  if (!p) return "?";
  std::string s;
  for (size_t i = 0; i < cap && p[i]; ++i) s.push_back(p[i]);
  return s;
}

uint32_t MktGuestU32(uint32_t addr) {
  const auto* p = rex::system::kernel_memory()->TranslateVirtual<const rex::be<uint32_t>*>(addr);
  return p ? p->get() : 0xDEADBEEFu;
}

// A guest address that is safe to follow. Anything outside the mapped ranges
// is junk the guest is about to fault on, and reading it here would take the
// host down first.
bool MktPlausible(uint32_t addr) {
  return (addr >= 0x82000000u && addr < 0xA0000000u) ||
         (addr >= 0x40000000u && addr < 0x60000000u);
}

void MktGuestWriteU32(uint32_t addr, uint32_t value) {
  auto* p = rex::system::kernel_memory()->TranslateVirtual<rex::be<uint32_t>*>(addr);
  if (p) *p = value;
}

}  // namespace

// sub_920D4970: the item-metadata fetch state machine. Its first argument is
// the fetcher object taken from the registry's own first field, and state 0
// calls straight through that object's vtable+8.
// The fetchers seen so far, by request name: a fetcher caches its last result
// (state 2 at +12) and answers a repeat of the same name and dims from it, so
// a re-entered page never asks the server again. Resetting the state makes it.
static std::mutex g_mkt_fetcher_mu;
static std::map<std::string, uint32_t> g_mkt_fetchers;

void MktResetFetchers(const char* prefix) {
  std::lock_guard<std::mutex> lock(g_mkt_fetcher_mu);
  for (const auto& [name, fetcher] : g_mkt_fetchers) {
    if (name.compare(0, std::strlen(prefix), prefix) != 0 || !MktPlausible(fetcher)) continue;
    if (MktGuestU32(fetcher + 12) == 1) continue;  // in flight, leave it
    MktGuestWriteU32(fetcher + 12, 0);
    REXKRNL_DEBUG("[xuisearch] fetcher '{}' reset", name);
  }
}

void AE_MktItemFetch(PPCRegister& r3, PPCRegister& r4) {
  const uint32_t fetcher = r3.u32;
  if (MktPlausible(fetcher) && MktPlausible(r4.u32)) {
    std::lock_guard<std::mutex> lock(g_mkt_fetcher_mu);
    g_mkt_fetchers[MktGuestAnsi(r4.u32, 96)] = fetcher;
  }
}

// The FindGameOffers formatter (sub_922042D8) is the store's only nine-
// argument printf: its sixth vararg, the RatingIds tail, goes in the first
// stack slot. The title stores the significant word at slot+0 (0x54), but
// the host CRT shim reads the slot as a be<u64> and truncates, taking the
// word at 0x58 - whatever the UI code last left there, usually an eased
// float that then gets chased as a string pointer. Mirror the word into the
// low half before the call so both conventions read the same pointer.
void AE_MktRatingArgFix(PPCRegister& r1) {
  MktGuestWriteU32(r1.u32 + 0x58, MktGuestU32(r1.u32 + 0x54));
}

// The purchase-history manager lives at unk_945065F0; the class is shared
// with other resource managers, so both hooks leave those alone.
constexpr uint32_t kMktOwnershipObject = 0x945065F0u;

bool AE_MktOwnStatus(PPCRegister& r3) {
  if (r3.u32 != kMktOwnershipObject) {
    return false;
  }
  r3.u32 = 0;  // S_OK: the history is complete as far as the store is concerned
  return true;
}

bool AE_MktOwnContains(PPCRegister& r3, PPCRegister& r4) {
  if (r3.u32 != kMktOwnershipObject) {
    return false;
  }
  // Owned means installed: the closet holds it, whether it came with the
  // closet or was bought below.
  const auto* id =
      rex::system::kernel_memory()->TranslateVirtual<const rex::avatars::AssetId*>(r4.u32);
  r3.u32 = id && rex::avatars::GetCloset().Find(*id) != nullptr ? 1u : 0u;
  return true;
}

// The Purchase button. The overlay it used to open was a system UI, so the
// sale completes here: the tile's item (its 216-byte record starts with the
// id at +704) is downloaded into the closet and the generation word is bumped
// so every tile re-asks whether it is owned.
void AE_MktPurchase(PPCRegister& r3) {
  const uint32_t tile = r3.u32;
  const auto* id =
      rex::system::kernel_memory()->TranslateVirtual<const rex::avatars::AssetId*>(tile + 704);
  if (!id || id->is_zero()) {
    return;
  }
  if (rex::avatars::GetCloset().Find(*id)) {
    REXKRNL_INFO("[mkt-buy] {} is already in the closet", id->to_string());
    return;
  }
  if (!rex::kernel::xam::MarketplaceInstallItem(*id)) {
    REXKRNL_WARN("[mkt-buy] {} could not be downloaded from the marketplace server",
                 id->to_string());
    return;
  }
  MktGuestWriteU32(0x9452E398u, MktGuestU32(0x9452E398u) + 1u);
  // The console raised XN_LIVE_CONTENT_INSTALLED when a download landed; the
  // editor re-enumerates its assets on it (sub_920B8C40), so the wardrobe updates.
  REX_KERNEL_STATE()->BroadcastNotification(0x02000007u, 0);
  REXKRNL_INFO("[mkt-buy] {} bought and installed (tile {:#x})", id->to_string(), tile);
  ae::achievements::Unlock(ae::achievements::kTreatYourself);
}

// Seeds the store's item generation word (App+0x2248508) so tiles leave
// their busy state and try the hovered item on through sub_920DEB08.
void AE_MktHoverTryOn(PPCRegister& r3) {
  static std::atomic<bool> seeded{false};
  if (seeded.exchange(true)) {
    return;
  }
  MktGuestWriteU32(0x9452E398u, 1u);
  REXKRNL_DEBUG("[mkt-wear] hover try-on enabled (generation seeded) tile={:#x}", r3.u32);
}

bool AE_MktNoParentalGate(PPCRegister& r3) {
  r3.u32 = 0;  // not restricted
  return true;
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
