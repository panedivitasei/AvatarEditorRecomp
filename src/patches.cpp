// Guest-side patches for running outside the console environment.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

#include <fmt/format.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

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
// r3 == 0 doubles as the idle result for both pollers.
bool AE_GuardNullOpState(PPCRegister& r3) {
  return r3.u32 == 0;
}

bool AE_GuardNullOpActive(PPCRegister& r3) {
  return r3.u32 == 0;
}

// Marketplace manifest parser probes. Field offsets are the ones the parser
// (sub_922077B8) and its validators (sub_92207680 slot, sub_922075B8 channel)
// read; nothing here is written back.
namespace {

std::string MktGuestAnsi(uint32_t addr, size_t cap = 96) {
  const auto* p = rex::system::kernel_memory()->TranslateVirtual<const char*>(addr);
  if (!p) return "?";
  std::string s;
  for (size_t i = 0; i < cap && p[i]; ++i) s.push_back(p[i]);
  return s;
}

// UTF-16BE, printed as ASCII with anything wider shown as '?'.
std::string MktGuestWide(uint32_t addr, size_t cap = 48) {
  const auto* p = rex::system::kernel_memory()->TranslateVirtual<const uint8_t*>(addr);
  if (!p) return "?";
  std::string s;
  for (size_t i = 0; i < cap; ++i) {
    const uint16_t c = (uint16_t(p[i * 2]) << 8) | p[i * 2 + 1];
    if (!c) break;
    s.push_back(c < 0x7F ? char(c) : '?');
  }
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

uint32_t MktGuestU32Safe(uint32_t addr) {
  return MktPlausible(addr) ? MktGuestU32(addr) : 0xBADF00Du;
}

void MktGuestWriteU32(uint32_t addr, uint32_t value) {
  auto* p = rex::system::kernel_memory()->TranslateVirtual<rex::be<uint32_t>*>(addr);
  if (p) *p = value;
}

// Marketplace faults kept landing on addresses that decode as small floats
// (stale FPU spills read back as pointers); show the decoded value next to
// anything in a plausible range.
std::string MktFloatTag(uint32_t v) {
  if (v < 0x38000000u || v > 0x41000000u) return "";
  float f;
  std::memcpy(&f, &v, sizeof(f));
  return fmt::format(" <float {:g}>", f);
}

}  // namespace

void AE_MktParserEvent(PPCRegister& r3) {
  const uint32_t ctx = r3.u32;
  // The link register names the tokenizer feeding this state machine.
  const uint32_t lr = rex::runtime::ThreadState::Get()
                          ? uint32_t(rex::runtime::current_ppc_context()->lr)
                          : 0;
  REXKRNL_INFO("[mkt-parse] state={} start={} end={} name='{}' value='{}' from lr={:#x}",
               MktGuestU32(ctx + 0x560), MktGuestU32(ctx + 0x548), MktGuestU32(ctx + 0x54C),
               MktGuestAnsi(ctx + 0x428, 32), MktGuestAnsi(ctx + 0x448, 96), lr);
}

void AE_MktSlotValidate(PPCRegister& r3) {
  const uint32_t ctx = r3.u32;
  REXKRNL_INFO(
      "[mkt-slot] action='{}' desc='{}' img='{}' spare147C={:#x} flag187C={} slots={}/{}",
      MktGuestAnsi(ctx + 0xEFC), MktGuestWide(ctx + 0xF7C), MktGuestAnsi(ctx + 0x107C),
      MktGuestAnsi(ctx + 0x147C, 1).empty() ? 0 : 1, MktGuestU32(ctx + 0x187C),
      MktGuestU32(ctx + 0xEF8), MktGuestU32(ctx + 0x570));
}

void AE_MktChannelValidate(PPCRegister& r3) {
  const uint32_t ctx = r3.u32;
  REXKRNL_INFO(
      "[mkt-channel] action='{}' desc='{}' bg='{}' spare6F4={} committed={} slots={}/{}",
      MktGuestAnsi(ctx + 0x574), MktGuestWide(ctx + 0x5F4), MktGuestAnsi(ctx + 0xAF4),
      MktGuestAnsi(ctx + 0x6F4, 1).empty() ? 0 : 1, MktGuestU32(ctx + 0xEF4),
      MktGuestU32(ctx + 0xEF8), MktGuestU32(ctx + 0x570));
}

// Net driver probes. The XHTTP pump struct (sub_9228EF10's int array) lives
// at driver+840: [0] state, [2] request handle, [4] read buffer, [6..7] content
// length, [8] pending result.
void AE_MktNetPoll(PPCRegister& r3, PPCRegister& r4) {
  const uint32_t obj = r3.u32, pump = obj + 840;
  REXKRNL_INFO("[mkt-net] poll obj={:#x} target={} abort={:#x} pump.state={} req={:#x} buf={:#x} "
               "len={} result={:#x}",
               obj, r4.u32, MktGuestU32(obj + 288), MktGuestU32(pump), MktGuestU32(pump + 8),
               MktGuestU32(pump + 16), MktGuestU32(pump + 28), MktGuestU32(pump + 32));
}

void AE_MktNetHeaders(PPCRegister& r3, PPCRegister& r4) {
  REXKRNL_INFO("[mkt-net] headers-stage obj={:#x} hr={:#x}", r3.u32, r4.u32);
}

void AE_MktNetTeardown(PPCRegister& r3) {
  const uint32_t obj = r3.u32, pump = obj + 840;
  // The link register names which stage decided to tear the request down.
  const uint32_t lr = rex::runtime::ThreadState::Get()
                          ? uint32_t(rex::runtime::current_ppc_context()->lr)
                          : 0;
  REXKRNL_INFO("[mkt-net] TEARDOWN obj={:#x} abort={:#x} pump.state={} result={:#x} from lr={:#x}",
               obj, MktGuestU32(obj + 288), MktGuestU32(pump), MktGuestU32(pump + 32), lr);
}

void AE_MktNetClose(PPCRegister& r3) {
  REXKRNL_INFO("[mkt-net] close conn={:#x}", r3.u32);
}

// Store navigation. sub_920D2EF0 looks a scene/slot up by its action string;
// sub_920E0620 then lazily resolves the selected object's own name (+572) and
// sub_920E04B0 copies the 0x984 record it found into it. The record's narrow
// string at +384 becomes a preview asset path.
void AE_MktSceneLookup(PPCRegister& r3, PPCRegister& r4, PPCRegister& r5) {
  // The item-download builder (sub_920B5C20) and the preview pipeline
  // (sub_920DFDF0 / sub_920E0680) refuse before doing anything when these
  // LIVE flags are clear, which looks exactly like "nothing happened".
  static std::atomic<bool> dumped{false};
  if (!dumped.exchange(true)) {
    const auto* p = rex::system::kernel_memory()->TranslateVirtual<const uint8_t*>(0x94249EA8);
    if (p) {
      REXKRNL_INFO("[mkt-gate] 94249EA8={} A9={} AA={} AB={} AC={}", p[0], p[1], p[2], p[3], p[4]);
    }
  }
  REXKRNL_INFO("[mkt-nav] lookup registry={:#x} name='{}' flags={}", r3.u32,
               MktGuestAnsi(r4.u32, 128), r5.u32);
}

void AE_MktPreviewLoad(PPCRegister& r3) {
  const uint32_t obj = r3.u32;
  REXKRNL_INFO("[mkt-nav] preview-load obj={:#x} ready={} name='{}' state={:#x}", obj,
               MktGuestAnsi(obj + 3008, 1).empty() ? 0 : 1, MktGuestAnsi(obj + 572, 128),
               MktGuestU32(obj + 3012));
}

// sub_920D4970: the item-metadata fetch state machine. Its first argument is
// the fetcher object taken from the registry's own first field, and state 0
// calls straight through that object's vtable+8.
void AE_MktItemFetch(PPCRegister& r3, PPCRegister& r4) {
  const uint32_t fetcher = r3.u32;
  const uint32_t vtable = MktGuestU32Safe(fetcher);
  REXKRNL_INFO("[mkt-fetch] fetcher={:#x} plausible={} vtable={:#x} vtable+8={:#x} state={} name='{}'",
               fetcher, MktPlausible(fetcher), vtable, MktGuestU32Safe(vtable + 8),
               MktPlausible(fetcher) ? MktGuestU32(fetcher + 12) : 0xBADF00Du,
               MktPlausible(r4.u32) ? MktGuestAnsi(r4.u32, 96) : "?");
}

// sub_920D4C70(fetcher, request): the request's scene kind lives at +136 and
// selects a query descriptor; the inner handle it dispatches through is at
// fetcher[2].
void AE_MktQueryIssue(PPCRegister& r3, PPCRegister& r4) {
  const uint32_t fetcher = r3.u32, request = r4.u32;
  const uint32_t handle = MktPlausible(fetcher) ? MktGuestU32(fetcher + 8) : 0xBADF00Du;
  REXKRNL_INFO("[mkt-query] fetcher={:#x} request={:#x} kind={} handle={:#x} handle.vtable={:#x}",
               fetcher, request, MktPlausible(request) ? MktGuestU32(request + 136) : 0xBADF00Du,
               handle, MktGuestU32Safe(handle));
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

// Kinds 13..23 map to real descriptors; 0..12 all fall through to 13, which is
// the out-of-range error value.
void AE_MktDispatchKind(PPCRegister& r4) {
  REXKRNL_INFO("[mkt-query] dispatch kind={}{}", r4.u32, r4.u32 > 0x17 || r4.u32 < 13
                                                             ? " (INVALID -> 13)" : "");
}

// sub_92203D60: the title:<id> query builder. It takes the scene name from
// request+16 and skips the 15 characters of "title:url:uuid:" to get the guid.
void AE_MktTitleQuery(PPCRegister& r3, PPCRegister& r4) {
  const uint32_t client = r3.u32, request = r4.u32;
  const uint32_t name_ptr = MktPlausible(request) ? MktGuestU32(request + 16) : 0;
  REXKRNL_INFO("[mkt-title] client={:#x} request={:#x} name={:#x} '{}'", client, request, name_ptr,
               MktPlausible(name_ptr) ? MktGuestAnsi(name_ptr, 96) : "?");
}

// sub_922031E0 validates the online XUID at handle+16 (top 16 bits must be
// 0x0009) and calls through the interface vptr embedded at +4; r5 is the
// mapped query id. Dump every slot the call touches.
void AE_MktMarshal(PPCRegister& r3, PPCRegister& r5) {
  const uint32_t handle = r3.u32;
  const bool ok = MktPlausible(handle);
  const uint32_t iface = ok ? MktGuestU32(handle + 4) : 0xBADF00Du;
  REXKRNL_INFO("[mkt-marshal] handle={:#x} vptr={:#x} iface={:#x}{} refs={:#x} xuid={:#x}:{:08x} "
               "inner={:#x} id={}",
               handle, ok ? MktGuestU32(handle) : 0xBADF00Du, iface, MktFloatTag(iface),
               ok ? MktGuestU32(handle + 12) : 0xBADF00Du,
               ok ? MktGuestU32(handle + 16) : 0xBADF00Du,
               ok ? MktGuestU32(handle + 20) : 0xBADF00Du,
               ok ? MktGuestU32(handle + 24) : 0xBADF00Du, r5.u32);
}

// The try-on pipeline: sub_920DFDF0 starts it and sub_920E0680 continues it,
// both refusing early on the LIVE flags and recording why in the error global
// at 0x94249EC0. Hovering a store item should drive this; silence here means
// the grid never asks for a preview at all.
void AE_MktWearStart(PPCRegister& r3) {
  static std::atomic<uint32_t> logged{0};
  if (logged.fetch_add(1) >= 40) {
    return;
  }
  REXKRNL_INFO("[mkt-wear] start obj={:#x} err={}", r3.u32, MktGuestU32Safe(0x94249EC0));
}

void AE_MktWearStep(PPCRegister& r3) {
  static std::atomic<uint32_t> logged{0};
  if (logged.fetch_add(1) >= 40) {
    return;
  }
  REXKRNL_INFO("[mkt-wear] step obj={:#x} err={}", r3.u32, MktGuestU32Safe(0x94249EC0));
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
}

// Seeds the store's item generation word (App+0x2248508) so tiles leave
// their busy state and try the hovered item on through sub_920DEB08.
void AE_MktHoverTryOn(PPCRegister& r3) {
  static std::atomic<bool> seeded{false};
  if (seeded.exchange(true)) {
    return;
  }
  MktGuestWriteU32(0x9452E398u, 1u);
  REXKRNL_INFO("[mkt-wear] hover try-on enabled (generation seeded) tile={:#x}", r3.u32);
}

bool AE_MktNoParentalGate(PPCRegister& r3) {
  r3.u32 = 0;  // not restricted
  return true;
}

void AE_MktPreviewApply(PPCRegister& r3, PPCRegister& r4) {
  const uint32_t obj = r3.u32, record = r4.u32;
  REXKRNL_INFO("[mkt-nav] preview-apply obj={:#x} record={:#x} rec.name='{}' rec+384='{}'", obj,
               record, MktGuestAnsi(record, 96), MktGuestAnsi(record + 384, 128));
}

// First-chance AV reporter. The runtime's own handler names the faulting
// address but not where the guest was, which has made the store crash a
// guessing game; the generated code keeps the full guest register file
// current in the thread's PPCContext, so dump it here. GPU write-watch
// faults hit committed pages and are filtered out by VirtualQuery; anything
// landing in free or reserved memory is a genuine wild access.
namespace {

LONG WINAPI MktFaultProbe(EXCEPTION_POINTERS* info) {
  static std::atomic<uint32_t> reports{0};
  if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  const uintptr_t target = uintptr_t(info->ExceptionRecord->ExceptionInformation[1]);
  MEMORY_BASIC_INFORMATION mbi{};
  if (VirtualQuery(reinterpret_cast<void*>(target), &mbi, sizeof(mbi)) &&
      mbi.State == MEM_COMMIT) {
    return EXCEPTION_CONTINUE_SEARCH;  // access-callback machinery, not ours
  }
  if (!rex::runtime::ThreadState::Get() || reports.fetch_add(1) >= 8) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  auto* ctx = rex::runtime::current_ppc_context();
  if (!ctx) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  REXKRNL_ERROR(
      "[mkt-fault] {} of host {:#x} at rip={:#x}; guest lr={:#x} ctr={:#x} r1={:#x}",
      info->ExceptionRecord->ExceptionInformation[0] ? "write" : "read", target,
      uintptr_t(info->ContextRecord->Rip), uint32_t(ctx->lr), ctx->ctr.u32,
      ctx->r1.u32);
  REXKRNL_ERROR(
      "[mkt-fault] r3={:#x} r4={:#x} r5={:#x} r6={:#x} r7={:#x} r8={:#x} r9={:#x} r10={:#x} "
      "r11={:#x} r12={:#x}",
      ctx->r3.u32, ctx->r4.u32, ctx->r5.u32, ctx->r6.u32, ctx->r7.u32, ctx->r8.u32, ctx->r9.u32,
      ctx->r10.u32, ctx->r11.u32, ctx->r12.u32);
  REXKRNL_ERROR(
      "[mkt-fault] r26={:#x} r27={:#x} r28={:#x} r29={:#x} r30={:#x} r31={:#x}",
      ctx->r26.u32, ctx->r27.u32, ctx->r28.u32, ctx->r29.u32, ctx->r30.u32, ctx->r31.u32);
  std::string stack;
  for (uint32_t i = 0; i < 16; ++i) {
    stack += fmt::format("{}{:08x}", i ? " " : "", MktGuestU32Safe(ctx->r1.u32 + i * 4));
  }
  REXKRNL_ERROR("[mkt-fault] stack @r1: {}", stack);
  return EXCEPTION_CONTINUE_SEARCH;
}

struct MktFaultProbeInstall {
  MktFaultProbeInstall() { AddVectoredExceptionHandler(1, MktFaultProbe); }
} g_mkt_fault_probe;

}  // namespace

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
