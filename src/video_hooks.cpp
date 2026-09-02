// Native renderer boundary. Draw entry points are replaced under
// ae_vn_replace; state setters are mirrored so their guest bodies still
// write the device shadows the renderer reads at draw time. Addresses are
// for the base image at 0x92000000 (no title update).

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "generated/ae/ae_init.h"

#include "catalog_search.h"
#include "kernel/xam/marketplace.h"

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/kernel/xam/avatar_search.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>
#include <rex/system/thread_state.h>
#include <video_native.h>

// A missed draw path means absent geometry, not a crash; ae_vn_strict
// turns unwired draw APIs fatal to look for gaps.
REXCVAR_DEFINE_BOOL(ae_vn_replace, true, "AE",
                    "Skip the guest body for draw entry points so the ring "
                    "GPU stops duplicating the work.");
REXCVAR_DEFINE_BOOL(ae_vn_strict, true, "AE",
                    "Fatal on any draw reaching an unwired native-renderer "
                    "boundary point.");

// Separate forwarding toggles for the tiling bracket.
REXCVAR_DEFINE_BOOL(ae_vn_begin_tiling, true, "AE",
                    "Forward D3DDevice_BeginTiling to the native renderer.");
REXCVAR_DEFINE_BOOL(ae_vn_end_tiling, true, "AE",
                    "Forward D3DDevice_EndTiling (per-tile resolve) natively.");

namespace vn = rex::videonative;
namespace rex::videonative::renderer {
// Probing memoized guest reader (renderer.h): safe on uncommitted pages.
const uint8_t* GuestDataPtrProbe(uint32_t addr, size_t bytes);
// Pending-writeback base query for the LockRect sync hook (renderer.h).
bool WritebackPendingForBase(uint32_t base);
// CPU-authority note for the LockRect hook (renderer.h).
void NoteGuestTextureCpuLock(uint32_t base);
}

// Catalog search feature: while the search box is open, the set-text sniff
// swaps the heading message's text pointer for the status line, so it
// renders as a real element.
namespace rex::videonative::fps {
int GetSearchStatusLine(char* utf8, int cap);          // renderer_fps.h
void SetSearchLabelPossessed(bool possessed);          // renderer_fps.h
bool ConsumeCatalogRebuildRequest();                   // renderer_fps.h
void SetSearchApplied(bool applied);                   // renderer_fps.h
}

std::atomic<uint32_t> g_aeFrame{0};  // bumped once per Swap by AeXuiSearchTick
static uint32_t g_srchGuestTextEa = 0;   // guest scratch for the status line
static bool g_srchPossessed = false;
static uint32_t g_srchDirtyScratch = 0;  // guest 25x 0x01 for the dirty latch
// Current catalog channel (0..24) when a grid screen is open, else -1.
static std::atomic<int> g_aeCurCatalogScope{-1};

// Guest memory base, fetched once; hook handlers receive named registers
// only.
static uint8_t* GuestBase() {
  static uint8_t* b = rex::system::kernel_state()->memory()->virtual_membase();
  return b;
}

// Live PPC context of the guest thread this hook runs on. ThreadState binds
// it at thread start, so it is the same object the recompiled code mutates.
static PPCContext& GuestCtx() {
  return *rex::runtime::current_ppc_context();
}

// Call a guest function by dynamic address (vtable slots). Bounds-checked
// so a garbage pointer degrades to a no-op instead of a fatal trap.
static uint32_t AeCallGuest(PPCContext& parent, uint8_t* base, uint32_t target,
                            uint32_t r3, uint32_t r4) {
  if ((target & 3u) || uint32_t(target - REX_CODE_BASE) >= REX_CODE_SIZE) {
    return 0xFFFFFFFFu;
  }
  PPCContext ctx = parent;
  ctx.r1.u32 -= 0x70;  // scratch above the hook frame (ImportFunction ABI)
  ctx.r3.u32 = r3;
  ctx.r4.u32 = r4;
  REX_CALL_INDIRECT_FUNC(target);
  return ctx.r3.u32;
}

// XUI handle -> element object, the guest's table walk from sub_92158A38
// (handle = generation<<16 | index, page table at 0x9457EAF8).
static uint32_t AeXuiHandleToElement(uint32_t handle) {
  auto* mem = rex::system::kernel_memory();
  const auto rd = [&](uint32_t ea, uint32_t* out) -> bool {
    if (!ea) return false;
    const uint32_t* p = mem->TranslateVirtual<const uint32_t*>(ea);
    if (!p) return false;
    *out = __builtin_bswap32(*p);
    return true;
  };
  const uint32_t idx = handle & 0xFFFFu;
  uint32_t count = 0, page = 0, gen = 0, node = 0, next = 0, elem = 0;
  if (!handle || !rd(0x9457EF18u, &count) || idx >= count) return 0;
  if (!rd(0x9457EAF8u + ((idx >> 6) & 0x3FFFFFCu), &page) || !page) return 0;
  const uint32_t entry = page + ((idx * 8u) & 0x7F8u);
  if (!rd(entry, &gen) || gen != (handle >> 16)) return 0;
  if (!rd(entry + 4u, &node) || !node) return 0;
  for (int guard = 0; guard < 64 && rd(node + 4u, &next) && next; ++guard) node = next;
  if (!rd(node + 32u, &elem)) return 0;
  return elem;
}

// Hide the Kinect "Identifying. Please face the sensor." widget.
// Clearing the visible bit and zeroing the opacity is how
// XUI's visibility walk tests it.
bool AE_HideIdentifyingGrp(PPCRegister& r31) {
  auto* mem = rex::system::kernel_memory();
  const uint32_t widget = r31.u32;
  if (!widget) return false;
  uint32_t hidden = 0, elems[2] = {0, 0};
  for (int i = 0; i < 2; ++i) {
    const uint32_t* hp = mem->TranslateVirtual<const uint32_t*>(widget + (i == 0 ? 12u : 8u));
    const uint32_t handle = hp ? __builtin_bswap32(*hp) : 0;
    const uint32_t elem = AeXuiHandleToElement(handle);
    elems[i] = elem;
    if (!elem) continue;
    uint32_t* opacity = mem->TranslateVirtual<uint32_t*>(elem + 36u);
    uint32_t* flags = mem->TranslateVirtual<uint32_t*>(elem + 180u);
    if (!opacity || !flags) continue;
    *opacity = 0;  // 0.0f
    *flags = __builtin_bswap32(__builtin_bswap32(*flags) & ~1u);
    hidden++;
  }
  return false;
}

void AE_SniffXuiSetText(PPCRegister& r3, PPCRegister& r4) {
  const uint32_t msg_ea = r4.u32;
  if (!msg_ea) return;
  // Params words are often not pointers; probe-read everything.
  const auto rd32 = [&](uint32_t ea) -> uint32_t {
    const uint8_t* q = rex::videonative::renderer::GuestDataPtrProbe(ea, 4);
    if (!q) return 0;
    return (uint32_t(q[0]) << 24) | (uint32_t(q[1]) << 16) |
           (uint32_t(q[2]) << 8) | q[3];
  };
  const uint32_t msg_id = rd32(msg_ea + 4);
  const uint32_t params_ea = rd32(msg_ea + 16);
  if (!params_ea) return;
  const uint32_t text_ea = rd32(params_ea + 12);
  if (!text_ea) return;
  const uint8_t* p =
      rex::videonative::renderer::GuestDataPtrProbe(text_ea, 130);
  if (!p) return;
  // Only treat plausible ASCII-leading UTF-16 strings as text payloads.
  if (p[0] != 0 || p[1] < 0x20 || p[1] >= 0x7F) return;
  static const char kPage[] = "Page ";
  bool is_page = true;
  for (int i = 0; i < 5; i++) {
    if (p[i * 2] != 0 || p[i * 2 + 1] != uint8_t(kPage[i])) {
      is_page = false;
      break;
    }
  }
  // ASCII-fold the original text; the guest always hands the clean string
  // here (the swapped buffer never comes back through).
  char orig[64];
  int orig_len = 0;
  for (; orig_len < 60; orig_len++) {
    const uint8_t hi = p[orig_len * 2], lo = p[orig_len * 2 + 1];
    if (!hi && !lo) break;
    orig[orig_len] = (!hi && lo >= 0x20 && lo < 0x7F) ? char(lo) : ' ';
  }
  orig[orig_len] = 0;
  // The heading is the id-2016 label the guest refreshes every frame,
  // plus any label carrying the same text (page flips move the refresh to
  // a new handle). Other id-2016 labels are left alone.
  bool is_heading = false;
  if (!is_page && msg_id == 2016) {
    // Track per-handle frame streaks; the heading is the longest-running
    // per-frame setter, and heading_text only follows a unique longest so
    // a tie cannot steal it.
    struct Setter {
      uint32_t handle, last_frame, streak;
    };
    static Setter setters[8] = {};
    static char heading_text[64] = "";
    extern std::atomic<uint32_t> g_aeFrame;
    const uint32_t frame = g_aeFrame.load(std::memory_order_relaxed);
    Setter* me = nullptr;
    Setter* victim = &setters[0];
    for (auto& st : setters) {
      if (st.handle == r3.u32) {
        me = &st;
        break;
      }
      if (st.last_frame < victim->last_frame) victim = &st;
    }
    if (!me) {
      me = victim;
      *me = Setter{r3.u32, frame, 0};
    } else if (me->last_frame + 1 == frame) {
      me->streak++;
      me->last_frame = frame;
    } else if (me->last_frame != frame) {
      me->streak = 0;
      me->last_frame = frame;
    }
    uint32_t best = 0, best_count = 0;
    for (const auto& st : setters) {
      if (!st.handle || st.last_frame + 1 < frame) continue;  // not live
      if (st.streak > best) {
        best = st.streak;
        best_count = 1;
      } else if (st.streak == best) {
        ++best_count;
      }
    }
    const bool longest = me->streak >= 2 && me->streak >= best;
    if (longest && best_count == 1 && orig_len > 0) {
      std::memcpy(heading_text, orig, size_t(orig_len) + 1);
    }
    is_heading = (longest && best_count == 1) ||
                 (heading_text[0] && std::strcmp(orig, heading_text) == 0);
  }
  // Swap the text pointer inside the guest's own set-text message; the
  // heading restores itself once the rewriting stops.
  if (is_heading) {
    char line[96];
    int len =
        rex::videonative::fps::GetSearchStatusLine(line, sizeof(line));
    // Nothing open or armed: append the search hint to the
    // catalog's own heading.
    if (len <= 0 && (g_aeCurCatalogScope.load(std::memory_order_relaxed) >= 0 ||
                     ae_search::GamesListOpen())) {
      int i = 0;
      for (; i < 60; i++) {
        const uint8_t hi = p[i * 2], lo = p[i * 2 + 1];
        if (!hi && !lo) break;
        line[i] = (!hi && lo >= 0x20 && lo < 0x7F) ? char(lo) : ' ';
      }
      if (i > 0) {
        len = i + std::snprintf(line + i, sizeof(line) - size_t(i),
                                " (Ctrl+F to search)");
      }
    }
    if (len > 0) {
      auto* mem = rex::system::kernel_state()->memory();
      if (!g_srchGuestTextEa) {
        g_srchGuestTextEa = mem->SystemHeapAlloc(512);
      }
      uint8_t* buf = g_srchGuestTextEa
                         ? mem->TranslateVirtual<uint8_t*>(g_srchGuestTextEa)
                         : nullptr;
      uint8_t* params =
          mem->TranslateVirtual<uint8_t*>(params_ea + 12);
      if (buf && params) {
        int i = 0;
        for (; i < len && i < 126; i++) {
          buf[i * 2] = 0;
          buf[i * 2 + 1] = uint8_t(line[i]);
        }
        buf[i * 2] = 0;
        buf[i * 2 + 1] = 0;
        params[0] = uint8_t(g_srchGuestTextEa >> 24);
        params[1] = uint8_t(g_srchGuestTextEa >> 16);
        params[2] = uint8_t(g_srchGuestTextEa >> 8);
        params[3] = uint8_t(g_srchGuestTextEa);
        if (!g_srchPossessed) {
          g_srchPossessed = true;
          rex::videonative::fps::SetSearchLabelPossessed(true);
        }
      }
    } else if (g_srchPossessed) {
      g_srchPossessed = false;
      rex::videonative::fps::SetSearchLabelPossessed(false);
    }
  }
}

// Per-frame search bookkeeping, run from the Swap hook.
static void AeXuiSearchTick(PPCContext& ctx, uint8_t* base) {
  g_aeFrame.fetch_add(1, std::memory_order_relaxed);
  char line[96];
  const int n =
      rex::videonative::fps::GetSearchStatusLine(line, sizeof(line));
  const bool open = n > 0;
  const bool rebuild = rex::videonative::fps::ConsumeCatalogRebuildRequest();

  auto* mem = rex::system::kernel_memory();
  const auto rd32 = [&](uint32_t ea) {
    return __builtin_bswap32(*mem->TranslateVirtual<const uint32_t*>(ea));
  };

  // Catalog scope: the navigator global caches the active screen handler
  // at +37092; grid screens answer kind==1 via vtbl+52 and their channel
  // via vtbl+56. A filter applies to the open grid's channel only.
  int scope = -1;
  // The awards screens answer the grid probe below too, so gate them out
  // by navigation page kind (ring of 16 x 2188-byte entries).
  bool in_awards = false;
  uint32_t top_kind = 0;
  {
    const uint32_t nav = 0x922E7604u;
    const uint32_t forced = rd32(nav + 35012u);
    const int top = int(rd32(nav));
    // Awards pages (kind 16) and the gamer picture booth (52/53/75/77).
    const auto unsearchable = [](uint32_t kind) {
      return kind == 16u || kind == 52u || kind == 53u || kind == 75u || kind == 77u;
    };
    in_awards = unsearchable(forced);
    for (int i = 0; !in_awards && i <= top && i < 16; ++i) {
      in_awards = unsearchable(rd32(nav + 4u + 2188u * uint32_t(i)));
    }
    top_kind = (top >= 0 && top < 16) ? rd32(nav + 4u + 2188u * uint32_t(top)) : 0u;
  }
  {
    const uint32_t screen = in_awards ? 0u : rd32(0x922E7604u + 37092u);
    if (screen) {
      const uint32_t vtbl = rd32(screen);
      if (vtbl && AeCallGuest(ctx, base, rd32(vtbl + 52), screen, 0) == 1) {
        const uint32_t chan =
            AeCallGuest(ctx, base, rd32(vtbl + 56), screen, 0);
        if (chan < 25) scope = int(chan);
      }
    }
    g_aeCurCatalogScope.store(scope, std::memory_order_relaxed);
    static int last_scope = -2;
    if (scope != last_scope) {
      last_scope = scope;
      rex::kernel::xam::SetAvatarCatalogSearchScope(scope);
    }
  }

  // Latch the registry's dirty flags (sub_920B8AD8); the title's own pump
  // then re-pushes the open grid screen with its selection restored.
  const auto latch_dirty = [&]() {
    if (!g_srchDirtyScratch) {
      g_srchDirtyScratch =
          rex::system::kernel_state()->memory()->SystemHeapAlloc(32);
      if (g_srchDirtyScratch) {
        std::memset(mem->TranslateVirtual<uint8_t*>(g_srchDirtyScratch), 1,
                    25);
      }
    }
    if (g_srchDirtyScratch) {
      PPCContext c2 = ctx;
      c2.r3.u32 = 0x922F088Cu;  // the catalog registry singleton
      c2.r4.u32 = g_srchDirtyScratch;
      sub_920B8AD8(c2, base);
    }
  };

  // A synchronous catalog rebuild would freeze a frame; spawn the guest's
  // own async rebuild (sub_920B8BC8, the boot and content-install path)
  // and watch for its ready flag below.
  static bool s_rbWatch = false;
  static uint32_t s_rbFrames = 0;
  const auto spawn_rebuild = [&]() {
    PPCContext c = ctx;
    c.r3.u32 = 0x922F088Cu;
    sub_920B8BC8(c, base);
    if (c.r3.u32) {
      s_rbWatch = true;
      s_rbFrames = 0;
    } else {
      // Worker spawn failed: fall back to the synchronous rebuild.
      REXKRNL_WARN("[xuisearch] async spawn failed, rebuilding inline");
      PPCContext c2 = ctx;
      c2.r3.u32 = 0;
      sub_920B8B50(c2, base);
      latch_dirty();
    }
  };

  // A filter belongs to the catalog it was applied in: clear it when the
  // scope changes. Filters applied outside any grid keep the Esc-to-clear
  // lifetime.
  static bool s_srchArmed = false;
  static int s_srchArmedScope = -1;
  const bool auto_clear =
      open && s_srchArmed && !rebuild && scope != s_srchArmedScope;
  if (auto_clear) {
    rex::kernel::xam::SetAvatarCatalogSearch("");
    rex::videonative::fps::SetSearchApplied(false);
    s_srchArmed = false;
    spawn_rebuild();
  }

  // Filter apply/clear from the Ctrl+F box (Enter/Esc).
  if (rebuild) {
    spawn_rebuild();
    s_srchArmed =
        !rex::kernel::xam::GetAvatarCatalogSearch().empty() && scope >= 0;
    s_srchArmedScope = scope;
  }

  // The store's Game Styles list is page kind 70; 67 is the loading page a
  // tile press pushes and 73 a game opened from the list.
  constexpr uint32_t kNavGameStyles = 70u, kNavGameOpening = 67u, kNavGameItems = 73u;
  struct Repush {
    bool pending;
    int frames;
    uint32_t slot_ea, title_ea;
  };
  static Repush s_repush = {};
  const auto wr32 = [&](uint32_t ea, uint32_t v) {
    *mem->TranslateVirtual<uint32_t*>(ea) = __builtin_bswap32(v);
  };
  // Fetched lists live in the store's ring of eight records (sub_920D3668),
  // and a page with the same slot name reuses one that looks live instead of
  // querying. Marking the list's records empty makes the next entry ask.
  const auto drop_game_list_cache = [&]() {
    constexpr uint32_t kStoreRing = 0x9426CF08u + 4u, kRecordStride = 158372u;
    for (uint32_t n = 0; n < 8; ++n) {
      const uint32_t rec = kStoreRing + n * kRecordStride;
      const char* name = reinterpret_cast<const char*>(mem->TranslateVirtual<const uint8_t*>(rec));
      if (name && std::strncmp(name, "storelist:alltitles", 20) == 0) {
        wr32(rec + 2432u, 0);
        wr32(rec + 158368u, 0xFFFFFFFFu);
        wr32(rec + 80396u, 0xFFFFFFFFu);
        wr32(rec + 80396u + 77964u, 0xFFFFFFFFu);
      }
    }
  };
  ae_search::SetGamesListOpen(top_kind == kNavGameStyles && !s_repush.pending);
  if (ae_search::ConsumeGamesReloadRequest() && top_kind == kNavGameStyles) {
    // Re-enter the page the way a tile press does: pop to the storefront,
    // then push the loading page with the slot name, whose router fetches the
    // list and swaps the real page in. The slot and title strings come from
    // the record being popped, so they go through a scratch copy.
    const auto nav_get = [&](void (*fn)(PPCContext&, uint8_t*)) {
      PPCContext c = ctx;
      c.r3.u32 = 0x922E7604u;
      fn(c, base);
      return c.r3.u32;
    };
    const uint32_t prev = nav_get(sub_920EBD50);
    static uint32_t s_navScratch = 0;
    if (!s_navScratch) {
      s_navScratch = mem->SystemHeapAlloc(0x200);
    }
    uint32_t slot_ea = 0, title_ea = 0;
    const uint32_t nav = 0x922E7604u;
    const int top = int(rd32(nav));
    if (s_navScratch && top >= 0 && top < 16) {
      const uint32_t rec = nav + 2188u * uint32_t(top);
      const uint8_t* slot = mem->TranslateVirtual<const uint8_t*>(rec + 8u);
      const uint8_t* title = mem->TranslateVirtual<const uint8_t*>(rec + 136u);
      uint8_t* out = mem->TranslateVirtual<uint8_t*>(s_navScratch);
      std::memset(out, 0, 0x200);
      for (int i = 0; i < 0x7F && slot[i]; ++i) out[i] = slot[i];
      for (int i = 0; i < 0xFF; ++i) {
        const uint8_t hi = title[i * 2], lo = title[i * 2 + 1];
        if (!hi && !lo) break;
        out[0x80 + i] = (!hi && lo >= 0x20 && lo < 0x7F) ? lo : ' ';
      }
      slot_ea = s_navScratch;
      title_ea = out[0x80] ? s_navScratch + 0x80u : 0u;
    }
    drop_game_list_cache();
    PPCContext c = ctx;
    c.r3.u32 = 0x922E7604u;
    c.r4.u32 = prev;
    c.r5.u32 = 2;
    c.r6.u32 = 1;
    sub_920EDA10(c, base);
    s_repush = Repush{true, 0, slot_ea, title_ea};
    REXKRNL_INFO("[xuisearch] game list reloading with filter '{}'",
                 rex::kernel::xam::MarketplaceGamesFilter());
  }
  // The push waits a frame for the pop to settle.
  bool just_pushed = false;
  if (s_repush.pending && ++s_repush.frames >= 2) {
    s_repush.pending = false;
    just_pushed = true;
    PPCContext c = ctx;
    c.r3.u32 = 0x922E7604u;
    c.r4.u32 = kNavGameOpening;
    c.r5.u32 = s_repush.slot_ea;
    c.r6.u32 = s_repush.title_ea;
    c.r7.u32 = 0xFFFFFFFFu;
    c.r8.u32 = 0xFFFFFFFFu;
    sub_920ED0E0(c, base);
  }
  // Leaving the list for anything but one of its games drops the filter and
  // the filtered list it cached. top_kind predates this tick's push, so the
  // push frame still reads as the storefront and is skipped.
  if (!rex::kernel::xam::MarketplaceGamesFilter().empty() && !s_repush.pending && !just_pushed &&
      top_kind != kNavGameStyles && top_kind != kNavGameOpening && top_kind != kNavGameItems) {
    rex::kernel::xam::SetMarketplaceGamesFilter("");
    drop_game_list_cache();
    if (rex::kernel::xam::GetAvatarCatalogSearch().empty()) {
      rex::videonative::fps::SetSearchApplied(false);
    }
  }

  // Rebuild completion watch: on ready, latch so the open grid re-pushes.
  if (s_rbWatch) {
    ++s_rbFrames;
    PPCContext c = ctx;
    c.r3.u32 = 0x922F088Cu;
    sub_9220F720(c, base);
    if (c.r3.u32) {
      s_rbWatch = false;
      latch_dirty();
    } else if (s_rbFrames > 1800) {  // ~30s: worker died/aborted, stop
      s_rbWatch = false;
      REXKRNL_WARN("[xuisearch] async rebuild never signalled ready");
    }
  }
  // Drop the possessed flag if the search closed while off-screen.
  if (!open && scope < 0 && g_srchPossessed) {
    g_srchPossessed = false;
    rex::videonative::fps::SetSearchLabelPossessed(false);
  }
}

// D3DTexture LockRect (0x9211B6A0): the gate before any CPU texture
// read. Deliver pending writeback now so the compose reads resolved art.
void AE_HookTextureLock(PPCRegister& r3) {
  const uint32_t tex = r3.u32;
  if (!tex) return;
  auto* mem = rex::system::kernel_memory();
  // D3DBaseTexture: 7 header dwords, then the 6-dword GPUTEXTURE_FETCH
  // constant at +0x1C; dw1 (the base address dword) lives at +0x20.
  const uint32_t* hdr = mem->TranslateVirtual<const uint32_t*>(tex + 0x20);
  if (!hdr) return;
  const uint32_t base = __builtin_bswap32(*hdr) & 0xFFFFF000u;
  if (!base) return;
  // The base is CPU-owned from this lock on; the redirect must not serve
  // the raw resolve product over it.
  rex::videonative::renderer::NoteGuestTextureCpuLock(base);
  // Only fires when a pending dest is about to be CPU-read (menu
  // transitions, not scroll frames).
  if (rex::videonative::renderer::WritebackPendingForBase(base)) {
    vn::FlushResolveWritebacksSync();
  }
}

// True once the native renderer is authoritative for draws.
static bool ReplaceDraws() { return REXCVAR_GET(ae_vn_replace); }

namespace {
void FlushPendingUp();  // deferred BeginVertices draw (defined below)
void FlushPendingDraw();  // deferred boundary draws (defined below)
extern uint32_t g_aeDevice;  // real guest device (defined below)
extern bool g_tilingActive;  // tiling bracket pairing (defined below)
}

// Host init only; the guest body still creates the real device, and the
// native layer attaches to it at first Swap via NoteRawDevice.
bool VideoNative_CreateDevice() {
  if (!vn::Enabled()) return false;
  if (!vn::Init()) {
    REXKRNL_WARN("[ae-vn] host device init failed, staying on ring");
  }
  return false;
}

// The D3D Swap contract resets surface bindings to the defaults and the
// scene pass relies on that; replicate it through the recompiled setters.
REX_IMPORT(sub_92119668, g_aeSetRenderTargetImport, u32(u32, u32, u32));
REX_IMPORT(sub_921199B8, g_aeSetDepthSurfaceImport, u32(u32, u32));

namespace {
uint32_t g_defaultRt0 = 0, g_defaultDs = 0;
}

// D3DDevice_Swap_Impl(device, front_buffer?, D3DVIDEO_SCALER_PARAMETERS*).
bool VideoNative_Swap(PPCRegister& r3, PPCRegister& r4, PPCRegister& r5) {
  if (!vn::IsActive()) return false;
  // Search bookkeeping, once per frame before the present. The tick calls
  // guest helpers, so it needs the live context and base.
  AeXuiSearchTick(GuestCtx(), GuestBase());
  FlushPendingUp();
  FlushPendingDraw();
  static std::atomic<bool> s_deviceNoted{false};
  if (!s_deviceNoted.exchange(true, std::memory_order_relaxed)) {
    vn::NoteRawDevice(r3.u32);
    g_aeDevice = r3.u32;
  }
  vn::Swap(r3.u32, r4.u32, r5.u32);
  // Remember the last real bindings; restore them when the frame ended
  // with them unbound (the RTT save/restore null case).
  {
    auto* mem = rex::system::kernel_memory();
    const uint32_t dev = r3.u32;
    const auto rd = [&](uint32_t off) {
      return __builtin_bswap32(
          *mem->TranslateVirtual<const uint32_t*>(dev + off));
    };
    const uint32_t cur_rt0 = rd(12616);  // SetRenderTarget slot 0
    const uint32_t cur_ds = rd(12632);   // bound depth surface (the gate)
    if (cur_rt0) g_defaultRt0 = cur_rt0;
    if (cur_ds) g_defaultDs = cur_ds;
    if (!cur_rt0 && g_defaultRt0) {
      g_aeSetRenderTargetImport(dev, 0, g_defaultRt0);
    }
    if (!cur_ds && g_defaultDs) {
      g_aeSetDepthSurfaceImport(dev, g_defaultDs);
    }
  }
  // The XDK Swap body drains the deferred Release queue; the ring present
  // is a natural no-op since vn owns the window.
  return false;
}

// D3DDevice_Clear(device, count, rects, flags, color, z f1, stencil r9);
// the float z consumes r8's GPR slot, so the stencil rides r9.
bool VideoNative_Clear(PPCRegister& r3, PPCRegister& r4, PPCRegister& r5,
                       PPCRegister& r6, PPCRegister& r7, PPCRegister& f1,
                       PPCRegister& r9) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::Clear(r3.u32, r4.u32, r5.u32, r6.u32, r7.u32, f1.f64, r9.u32);
  return false;
}

// Int SetViewport entry; unhooked (the shared tail below sees every
// caller), kept in the new named-register form.
bool VideoNative_SetViewport(PPCRegister& r3, PPCRegister& r4) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetViewport(r3.u32, r4.u32);
  return false;
}

// Shared tail of SetViewport and SetViewportF, viewport in f1..f6;
// hooking the tail catches both entries.
bool VideoNative_SetViewportTail(PPCRegister& r3, PPCRegister& f1,
                                 PPCRegister& f2, PPCRegister& f3,
                                 PPCRegister& f4, PPCRegister& f5,
                                 PPCRegister& f6) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetViewportValues(r3.u32, float(f1.f64), float(f2.f64), float(f3.f64),
                        float(f4.f64), float(f5.f64), float(f6.f64));
  return false;
}


// BeginVertices returns ring space the app fills afterwards, with no End
// call; the draw is deferred to the next D3D boundary call.
namespace {
struct PendingUp {
  bool active = false;
  uint32_t device = 0, prim = 0, count = 0, stride = 0, scratch = 0;
};
PendingUp g_pendingUp;

// The guest patches VS vfetch microcode inside its draw-commit body,
// after the hook fires, so draws are deferred to the next boundary call.
// UP data is copied to a scratch ring at hook time.
//   kIndexed   - DrawApi_D: indexed from bound VB/IB.   a=startIdx b=idxCount
//   kUp        - DrawPrimitiveUP wrapper: data copied.  a=count b=data c=stride
//   kAuto      - auto-index commit: stream 0 comes from the device fetch
//                shadow, no data in the args.           a=startVtx b=count
//   kIndexedUp - DrawIndexedPrimitiveUP wrapper: both blocks copied.
//                a=numVerts b=idxCount c=scratchIb e=scratchVb f=stride
struct PendingDraw {
  enum Kind { kNone, kIndexed, kUp, kAuto, kIndexedUp } kind = kNone;
  uint32_t device = 0, prim = 0, a = 0, b = 0, c = 0, e = 0, f = 0;
  uint32_t snap = 0;  // hook-time device snapshot (kAuto)
};
PendingDraw g_pendingDraw;

// BeginVertices capture pair: the entry hook latches the draw shape, the
// return-site hook (0x921B2744) reads the ring pointer DrawApi_A returned.
struct BeginVerticesLatch {
  bool armed = false;
  uint32_t device = 0, prim = 0, count = 0, stride = 0;
};
BeginVerticesLatch g_bvLatch;

uint32_t g_aeDevice = 0;  // real guest device, captured at first Swap
bool g_tilingActive = false;  // a forwarded BeginTiling awaits its EndTiling
uint32_t g_upScratch = 0, g_upScratchOff = 0;
constexpr uint32_t kUpScratchSize = 1u << 20;  // 1MB wrap ring

// The engine writes bones and stream pairs inline (no callable
// function), so a deferred draw could otherwise flush after the next
// piece's state landed. Snapshot the device block at hook time instead.
constexpr uint32_t kDeviceBlockSize = 0x5F00;
uint32_t g_devSnapshot = 0;  // guest EA of the snapshot block

uint32_t SnapshotDevice(uint32_t device) {
  if (!device) return device;
  auto* mem = rex::system::kernel_memory();
  if (!g_devSnapshot) {
    g_devSnapshot = mem->SystemHeapAlloc(kDeviceBlockSize, 0x100);
    if (!g_devSnapshot) return device;  // no scratch: fall back to live reads
  }
  std::memcpy(mem->TranslateVirtual<void*>(g_devSnapshot),
              mem->TranslateVirtual<const void*>(device), kDeviceBlockSize);
  return g_devSnapshot;
}

// Second snapshot slot for the BeginVertices latch: pendingUp and
// pendingDraw coexist, so each needs its own block or a later snapshot
// overwrites state an unflushed draw still references.
uint32_t g_devSnapshotUp = 0;

uint32_t SnapshotDeviceUp(uint32_t device) {
  if (!device) return device;
  auto* mem = rex::system::kernel_memory();
  if (!g_devSnapshotUp) {
    g_devSnapshotUp = mem->SystemHeapAlloc(kDeviceBlockSize, 0x100);
    if (!g_devSnapshotUp) return device;
  }
  std::memcpy(mem->TranslateVirtual<void*>(g_devSnapshotUp),
              mem->TranslateVirtual<const void*>(device), kDeviceBlockSize);
  return g_devSnapshotUp;
}

// Copy the windows the guest draw-commit body writes after the hook (the
// fetch-constant shadow and the stream stride table) from the live device
// into a hook-time snapshot, so a flush reads call-time render state plus
// the body's vfetch patch.
void OverlayLiveWindows(uint32_t snap, uint32_t device) {
  if (!snap || !device || snap == device) return;
  auto* mem = rex::system::kernel_memory();
  std::memcpy(mem->TranslateVirtual<void*>(snap + 1152),
              mem->TranslateVirtual<const void*>(device + 1152), 32 * 24);
  std::memcpy(mem->TranslateVirtual<void*>(snap + 12704),
              mem->TranslateVirtual<const void*>(device + 12704), 64);
}

void FlushPendingUp() {
  if (!g_pendingUp.active) return;
  g_pendingUp.active = false;
  vn::DrawVerticesUP(g_pendingUp.device, g_pendingUp.prim, g_pendingUp.count,
                     g_pendingUp.scratch, g_pendingUp.stride);
}

void FlushPendingDraw() {
  if (g_pendingDraw.kind == PendingDraw::kNone) return;
  const PendingDraw d = g_pendingDraw;
  g_pendingDraw.kind = PendingDraw::kNone;
  switch (d.kind) {
    case PendingDraw::kIndexed:
      // a=startIndex, b=indexCount, c=baseVertex (disasm-decoded).
      vn::DrawIndexedVertices(d.device, d.prim, d.c, d.a, d.b);
      break;
    case PendingDraw::kUp:
      vn::DrawVerticesUP(d.device, d.prim, d.a, d.b, d.c);
      break;
    case PendingDraw::kAuto:
      // Call-time render state, post-body vfetch: the avatar lib brackets
      // states around the commit (the shadow blob draws with z-write off
      // and restores it right after), so flush-time state is already
      // restored. The body only writes the fetch shadow and stride table
      // after the hook; overlay those onto the hook snapshot.
      if (d.snap) {
        OverlayLiveWindows(d.snap, d.device);
        vn::DrawVertices(d.snap, d.prim, d.a, d.b);
      } else {
        vn::DrawVertices(SnapshotDevice(d.device), d.prim, d.a, d.b);
      }
      break;
    case PendingDraw::kIndexedUp:
      vn::DrawIndexedVerticesUP(d.device, d.prim, 0, d.a, d.b, d.c,
                                /*index_format=*/1, d.e, d.f);
      break;
    default:
      break;
  }
}

// DrawApi returns a ring pointer callers write through; hand back
// scratch instead of null.
uint32_t AllocUpScratch(uint32_t bytes) {
  auto* mem = rex::system::kernel_memory();
  if (!g_upScratch) {
    g_upScratch = mem->SystemHeapAlloc(kUpScratchSize, 0x100);
    if (!g_upScratch) return 0;
  }
  if (!bytes || bytes > kUpScratchSize) bytes = 0x10000;
  if (g_upScratchOff + bytes > kUpScratchSize) g_upScratchOff = 0;
  const uint32_t ea = g_upScratch + g_upScratchOff;
  g_upScratchOff += (bytes + 0xFFu) & ~0xFFu;
  return ea;
}
}  // namespace

// BeginVertices entry: r4=prim, r5=count, r6=stride. DrawApi_A embeds
// the vfetch pair in the PM4 packet, so the ring pointer it returns is
// the only source for the vertex data.
bool VideoNative_BeginVerticesUP(PPCRegister& r3, PPCRegister& r4,
                                 PPCRegister& r5, PPCRegister& r6) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  g_bvLatch.armed = false;
  // The same address doubles as a raw callback with a different contract;
  // only handle calls whose holder carries the known device.
  auto* mem = rex::system::kernel_memory();
  const uint32_t dev =
      __builtin_bswap32(*mem->TranslateVirtual<const uint32_t*>(r3.u32 + 12u));
  if (!g_aeDevice || dev != g_aeDevice) {
    return false;
  }
  const uint32_t prim = r4.u32;
  const uint32_t count = r5.u32;
  const uint32_t stride = r6.u32;
  if (!count || count > 0x4000 || !stride || stride > 0x1000) return false;
  g_bvLatch.armed = true;
  g_bvLatch.device = dev;
  g_bvLatch.prim = prim;
  g_bvLatch.count = count;
  g_bvLatch.stride = stride;
  // Never replace an allocator: skipping the body would leave the caller
  // writing into garbage.
  return false;
}

// BeginVertices return site: r3 = the ring pointer the guest will fill.
bool VideoNative_BeginVerticesRet(PPCRegister& r3) {
  if (!g_bvLatch.armed) return false;
  g_bvLatch.armed = false;
  if (!vn::IsActive()) return false;
  const uint32_t ring_ptr = r3.u32;
  if (!ring_ptr) return false;
  // Older pendings still reference the shared snapshot block; flush them
  // before this snapshot lands.
  FlushPendingUp();
  FlushPendingDraw();
  g_pendingUp.active = true;
  // Vertices are read live at flush; state (atlas bind, constants) is
  // snapshotted now.
  g_pendingUp.device = SnapshotDeviceUp(g_bvLatch.device);
  g_pendingUp.prim = g_bvLatch.prim;
  g_pendingUp.count = g_bvLatch.count;
  g_pendingUp.stride = g_bvLatch.stride;
  g_pendingUp.scratch = ring_ptr;
  return false;
}

// Auto-index draw commit (0x92121380): stream 0 comes from the device
// fetch shadow. Args: (device, prim, startVertex, count).
bool VideoNative_DrawVerticesUP(PPCRegister& r3, PPCRegister& r4,
                                PPCRegister& r5, PPCRegister& r6) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  const uint32_t prim = r4.u32;
  const uint32_t start = r5.u32;
  const uint32_t count = r6.u32;
  if (count == 0 || count > 0x10000 || start > 0x100000) {
    static std::atomic<uint32_t> clamped{0};
    if ((clamped.fetch_add(1) & 0x3FF) == 0) {
      REXKRNL_WARN(
          "[ae-vn] auto-draw skipped, implausible start={:#x} count={:#x}",
          start, count);
    }
    return false;
  }
  // The guest body patches the vfetch pair into the fetch shadow; the
  // flush overlays that window onto a hook-time snapshot so bracketed
  // render states (the shadow blob's z-write) keep call-time values.
  g_pendingDraw.kind = PendingDraw::kAuto;
  g_pendingDraw.device = r3.u32;
  g_pendingDraw.snap = SnapshotDevice(r3.u32);
  g_pendingDraw.prim = prim;
  g_pendingDraw.a = start;
  g_pendingDraw.b = count;
  return false;
}

// DrawIndexedPrimitiveUP wrapper (0x921212E0). Args: (dev, prim,
// minVertex, numVerts, idxCount, idxData, flags [bit2 = 32-bit indices],
// vertData, stride on the stack). Uploading from the unoffset base keeps
// the indices' original numbering.
bool VideoNative_DrawIndexedUP(PPCRegister& r1, PPCRegister& r3,
                               PPCRegister& r4, PPCRegister& r5,
                               PPCRegister& r6, PPCRegister& r7,
                               PPCRegister& r8, PPCRegister& r9,
                               PPCRegister& r10) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  auto* mem = rex::system::kernel_memory();
  const uint32_t prim = r4.u32;
  const uint32_t min_vertex = r5.u32;
  const uint32_t num_verts = r6.u32;
  const uint32_t index_count = r7.u32;
  const uint32_t index_data = r8.u32;
  const uint32_t flags = r9.u32;
  const uint32_t vertex_data = r10.u32;
  // Stride is the first stack argument, at r1+0x54.
  const uint32_t stride = __builtin_bswap32(
      *mem->TranslateVirtual<const uint32_t*>(r1.u32 + 0x54));
  const uint32_t total_verts = min_vertex + num_verts;
  if (!num_verts || !index_count || !stride || stride > 0x1000 ||
      total_verts > 0x10000 || index_count > 0x20000 || !vertex_data ||
      !index_data) {
    return false;
  }
  if (flags & 4) {
    static std::atomic<uint32_t> warned{0};
    if (warned.fetch_add(1) == 0) {
      REXKRNL_WARN("[ae-vn] idxUP with 32-bit indices, guest path only");
    }
    return false;
  }
  const uint32_t vbytes = total_verts * stride;
  const uint32_t ibytes = (index_count * 2 + 3) & ~3u;
  if (vbytes + ibytes > kUpScratchSize / 2) return false;
  const uint32_t scratch_vb = AllocUpScratch(vbytes);
  const uint32_t scratch_ib = AllocUpScratch(ibytes);
  if (!scratch_vb || !scratch_ib) return false;
  std::memcpy(mem->TranslateVirtual<void*>(scratch_vb),
              mem->TranslateVirtual<const void*>(vertex_data), vbytes);
  std::memcpy(mem->TranslateVirtual<void*>(scratch_ib),
              mem->TranslateVirtual<const void*>(index_data), ibytes);
  g_pendingDraw.kind = PendingDraw::kIndexedUp;
  g_pendingDraw.device = SnapshotDevice(r3.u32);
  g_pendingDraw.prim = prim;
  g_pendingDraw.a = total_verts;
  g_pendingDraw.b = index_count;
  g_pendingDraw.c = scratch_ib;
  g_pendingDraw.e = scratch_vb;
  g_pendingDraw.f = stride;
  return ReplaceDraws();  // native owns this draw
}

// SetStreamSource (0x92118C00): writes the stream vfetch pairs into the
// device fetch array.
bool VideoNative_SetStreamSource(PPCRegister& r3, PPCRegister& r4,
                                 PPCRegister& r5, PPCRegister& r6,
                                 PPCRegister& r7, PPCRegister& r8) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetStreamSource(r3.u32, r4.u32, r5.u32, r6.u32, r8.u32, r7.u32);
  return false;
}

// SetTexture (0x9211BA20): writes the texture fetch block at
// dev+1152+24*stage; the guest body performs the same shadow write.
bool VideoNative_SetTexture(PPCRegister& r3, PPCRegister& r4,
                            PPCRegister& r5) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetTexture(r3.u32, r4.u32, r5.u32);
  return false;
}

// vn patches VS microcode the same way the XDK does, by calling the
// recompiled patcher (0x92136018) on a scratch copy.
REX_IMPORT(sub_92136018, g_aeVfetchPatcherImport,
           u32(u32, u32, u32, u32, u32));

static void AeVfetchPatchBridge(uint32_t vs_obj, uint32_t code_copy,
                                uint32_t decl, uint32_t stride_table,
                                uint32_t variant) {
  g_aeVfetchPatcherImport(vs_obj, code_copy, decl, stride_table, variant);
}

namespace {
struct AeVfetchPatcherRegistrar {
  AeVfetchPatcherRegistrar() { vn::SetVfetchPatcher(AeVfetchPatchBridge); }
} g_aeVfetchPatcherRegistrar;
}  // namespace

// Shader binds:
//   0x9211C938 = SetVertexShader: stores current VS at dev+12872 and
//     applies the container default-constant stream.
//   0x9211C730 = SetPixelShader: stores current PS at dev+12868.
//   0x9211CB50 = SetVertexDeclaration (stores dev+11992 + dirty bit).
//   0x92118DA8 = SetIndices (takes a D3DIndexBuffer).
bool VideoNative_SetVertexShader(PPCRegister& r3, PPCRegister& r4) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetVertexShader(r3.u32, r4.u32);
  return false;
}

bool VideoNative_SetPixelShader(PPCRegister& r3, PPCRegister& r4) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetPixelShader(r3.u32, r4.u32);
  return false;
}

// Core texture lock (0x9211B090): XUI rasterizes glyphs into atlas
// textures via CPU lock+write after creation. Observe-only: invalidate
// the cache entry so the next bind re-uploads the post-write texels.
bool VideoNative_TextureLock(PPCRegister& r3) {
  if (!vn::IsActive()) return false;
  vn::InvalidateTextureByHeader(r3.u32);
  // Header matching misses the atlas entries, and XUI renders label
  // strips the same frame it rasterizes new glyphs; retiring every small
  // R8 entry on a lock is cheap and closes the race.
  vn::InvalidateSmallAlphaTextures();
  return false;
}
// Resource unlock (0x92119DD0): the CPU write is complete, so retire the
// range; the next bind uploads the finished texels.
bool VideoNative_TextureUnlock(PPCRegister& r3) {
  if (!vn::IsActive()) return false;
  vn::InvalidateTextureAfterUnlock(r3.u32);
  return false;
}

bool VideoNative_SetVertexDecl(PPCRegister& r3, PPCRegister& r4) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetVertexDeclaration(r3.u32, r4.u32);
  return false;
}

// A deferred draw must flush before the guest mutates state the renderer
// reads at flush time (the avatar loop sets bones between draws).
bool VideoNative_FlushPoint() {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  return false;
}

// SetRenderTarget (0x92119668, stores at dev+12616+4*index) and
// SetDepthStencilSurface (0x921199B8).
bool VideoNative_SetRenderTarget(PPCRegister& r3, PPCRegister& r4,
                                 PPCRegister& r5) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetRenderTarget(r3.u32, r4.u32, r5.u32);
  return false;
}

bool VideoNative_SetDepthStencilSurface(PPCRegister& r3, PPCRegister& r4) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetDepthStencilSurface(r3.u32, r4.u32, 0);
  return false;
}

// EDRAM tiling bracket. AE renders the whole frame in predicated tiles
// and reaches the front buffer only through EndTiling's per-tile resolve.
// Frame loop (sub_920B01C0):
//   BeginTiling(dev, 0, tileCount=app+276, rects=app+212, 0, 1.0)
//   ... XUI scene renders ...
//   EndTiling(dev, 0, 0, frontbuffer[app+192], 0, 1.0)  <- resolves tiles
//   Swap(dev, frontbuffer[app+192]);  app+192 ^= 1
// BeginTiling 0x9211F940(dev, flags, count, pRects, pClearColor, clearZ(f1))
bool VideoNative_BeginTiling(PPCRegister& r3, PPCRegister& r4,
                             PPCRegister& r5, PPCRegister& r6,
                             PPCRegister& r7, PPCRegister& f1) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  // The Kinect camera-preview path brackets with a null rect array; guard
  // here since vn walks the rects.
  const uint32_t count = r5.u32;
  const uint32_t rects = r6.u32;
  if (!rects || count == 0 || count > 16) {
    return false;
  }
  if (!REXCVAR_GET(ae_vn_begin_tiling)) return false;
  g_tilingActive = true;
  vn::BeginTiling(r3.u32, r4.u32, count, rects, r7.u32, float(f1.f64), 0);
  return false;
}

// EndTiling (0x9211FCB0). vn's third argument is the rect array; AE
// passes a dest point here, so let vn reuse the BeginTiling rects.
bool VideoNative_EndTiling(PPCRegister& r3, PPCRegister& r4,
                           PPCRegister& r6) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  // Brackets must be paired: the camera-preview path's BeginTiling is
  // skipped (null rects), so its EndTiling must not resolve either.
  if (!g_tilingActive) return false;
  g_tilingActive = false;
  if (!REXCVAR_GET(ae_vn_end_tiling)) return false;
  vn::EndTiling(r3.u32, r4.u32, 0, r6.u32);
  return false;
}

// D3DDevice_Resolve (0x92123790): dev, flags, srcRect, destTexture,
// destPoint in r3..r7.
bool VideoNative_Resolve(PPCRegister& r3, PPCRegister& r4, PPCRegister& r5,
                         PPCRegister& r6, PPCRegister& r7) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::Resolve(r3.u32, r4.u32, r5.u32, r6.u32, r7.u32);
  return false;
}

// DrawApi_D (0x92121798): indexed draw from the bound VB/IB.
bool VideoNative_DrawIndexedVertices(PPCRegister& r3, PPCRegister& r4,
                                     PPCRegister& r5, PPCRegister& r6,
                                     PPCRegister& r7) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  // Arg decode from the call site at 0x9214c830:
  //   li r5,0 / srawi r6,(idxPtr-ibBase),1 / mulli r7,tris,3
  // => DrawApi_D(dev, prim, baseVertex r5, startIndex r6, indexCount r7).
  g_pendingDraw = {PendingDraw::kIndexed, SnapshotDevice(r3.u32),
                   r4.u32, r6.u32, r7.u32, r5.u32};
  return ReplaceDraws();  // native owns this draw
}

bool VideoNative_SetIndices(PPCRegister& r3, PPCRegister& r4) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  vn::SetIndices(r3.u32, r4.u32);
  return false;
}

// DrawPrimitiveUP wrapper (0x92120D58): dev, prim, vertexCount,
// pVertexData, stride in r3..r7.
bool VideoNative_DrawVerticesUP_Packed(PPCRegister& r3, PPCRegister& r4,
                                       PPCRegister& r5, PPCRegister& r6,
                                       PPCRegister& r7) {
  if (!vn::IsActive()) return false;
  FlushPendingUp();
  FlushPendingDraw();
  const uint32_t count = r5.u32;
  const uint32_t data = r6.u32;
  const uint32_t stride = r7.u32;
  if (count == 0 || count > 0x10000 || stride == 0 || stride > 0x1000) {
    static std::atomic<uint32_t> clamped{0};
    if ((clamped.fetch_add(1) & 0x3FF) == 0) {
      REXKRNL_WARN("[ae-vn] PackedUP skipped, implausible count={:#x} "
                   "stride={:#x}",
                   count, stride);
    }
    return false;
  }
  // Copy the caller's transient vertex data to scratch now; draw at the
  // next boundary.
  const uint32_t bytes = count * stride;
  const uint32_t scratch = AllocUpScratch(bytes);
  if (!scratch) return false;
  auto* mem = rex::system::kernel_memory();
  std::memcpy(mem->TranslateVirtual<void*>(scratch),
              mem->TranslateVirtual<const void*>(data), bytes);
  g_pendingDraw = {PendingDraw::kUp, SnapshotDevice(r3.u32), r4.u32,
                   count, scratch, stride};
  return ReplaceDraws();  // native owns this draw
}

// Unwired mapped draw APIs (DrawApi_A 0x92120898 / B 0x92120DA0 /
// D 0x92121798): fatal under ae_vn_strict so the crash names the next
// function to wire. With strict off they fall through to the guest ring.
bool AE_VnUnwiredDraw() {
  if (!vn::IsActive()) return false;
  // Known double-counts, not gaps: both are wired entries whose guest
  // bodies call down into DrawApi_A, so the draw is already shadowed.
  //   0x92120D58..0x92120DA0  DrawPrimitiveUP wrapper
  //   0x921B2744              return site inside BeginVertices (mirrored)
  // lr is not a hookable register in v0.10; read it from the live context.
  const uint32_t wrapper_lr = uint32_t(GuestCtx().lr);
  if (wrapper_lr >= 0x92120D58u && wrapper_lr < 0x92120DA0u) return false;
  if (wrapper_lr == 0x921B2744u) return false;
  static std::atomic<uint32_t> n{0};
  const uint32_t c = n.fetch_add(1, std::memory_order_relaxed);
  if (REXCVAR_GET(ae_vn_strict)) {
    REXKRNL_ERROR("[ae-vn] STRICT: unwired draw API hit (lr={:#x}), wire "
                  "this DrawApi before proceeding",
                  wrapper_lr);
    abort();
  }
  if ((c & 0xFFu) == 0) {
    REXKRNL_ERROR("[ae-vn] unwired draw API (lr={:#x}, total {})",
                  wrapper_lr, c + 1);
  }
  return false;
}
