// xui_hooks.cpp: XUI-side title hooks (Ctrl+F heading, per-frame search tick, Kinect hint, gamma ramp capture).

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "generated/ae/ae_init.h"

#include "catalog_search.h"
#include "gamma_ramp.h"
#include "kernel/xam/marketplace.h"
#include "search_state.h"

#include <rex/hook.h>
#include <rex/kernel/xam/avatar_search.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>
#include <rex/system/thread_state.h>
#include <rex/system/xmemory.h>

void MktResetFetchers(const char* prefix);  // patches.cpp: forget cached store queries

// Guest reader for words that may not be pointers, returns nullptr unless the whole span is committed.
static const uint8_t* GuestDataPtrProbe(uint32_t addr, size_t bytes) {
  // The zero page is committed but fenced off, so anything in the first 64 KB reads as absent.
  if (addr < 0x10000 || !bytes) return nullptr;
  auto* mem = rex::system::kernel_memory();
  if (!mem) return nullptr;
  const auto readable = [&](uint32_t ea) {
    auto* heap = mem->LookupHeap(ea);
    rex::memory::HeapAllocationInfo info{};
    uint32_t size = 0, protect = 0;
    // A free page is rejected by its zero region size first: QueryRegionInfo would walk the whole free run.
    return heap && heap->QuerySize(ea, &size) && size && heap->QueryRegionInfo(ea, &info) &&
           (info.state & rex::memory::kMemoryAllocationCommit) && heap->QueryProtect(ea, &protect) &&
           (protect & rex::memory::kMemoryProtectRead);
  };
  const uint32_t last = addr + uint32_t(bytes - 1);
  if (last < addr || !readable(addr) || !readable(last)) return nullptr;
  return mem->TranslateVirtual<const uint8_t*>(addr);
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

// Hides the Kinect "Identifying, please face the sensor" widget by clearing the visible bit and zeroing the
// opacity, which is what XUI's visibility walk tests.
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
    const uint8_t* q = GuestDataPtrProbe(ea, 4);
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
      GuestDataPtrProbe(text_ea, 130);
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
        ae_search::GetSearchStatusLine(line, sizeof(line));
    // Nothing open or armed: append the search hint to the
    // catalog's own heading.
    if (len <= 0 && (g_aeCurCatalogScope.load(std::memory_order_relaxed) >= 0 ||
                     ae_search::CurrentStorePage() != ae_search::StorePage::kNone)) {
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
          ae_search::SetSearchLabelPossessed(true);
        }
      }
    } else if (g_srchPossessed) {
      g_srchPossessed = false;
      ae_search::SetSearchLabelPossessed(false);
    }
  }
}

// Per-frame search bookkeeping, run from the Swap entry hook.
static void AeXuiSearchTick(PPCContext& ctx, uint8_t* base) {
  g_aeFrame.fetch_add(1, std::memory_order_relaxed);
  char line[96];
  const int n =
      ae_search::GetSearchStatusLine(line, sizeof(line));
  const bool open = n > 0;
  const bool rebuild = ae_search::ConsumeCatalogRebuildRequest();

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
    ae_search::SetSearchApplied(false);
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
    uint32_t slot_ea, title_ea, kind;
  };
  static Repush s_repush = {};
  const auto wr32 = [&](uint32_t ea, uint32_t v) {
    *mem->TranslateVirtual<uint32_t*>(ea) = __builtin_bswap32(v);
  };
  // Fetched lists live in the store's ring of eight records (sub_920D3668),
  // and a page with the same slot name reuses one that looks live instead of
  // querying. Marking a record empty makes the next entry ask again.
  const auto drop_store_cache = [&](const char* prefix) {
    constexpr uint32_t kStoreRing = 0x9426CF08u + 4u, kRecordStride = 158372u;
    const size_t plen = std::strlen(prefix);
    for (uint32_t n = 0; n < 8; ++n) {
      const uint32_t rec = kStoreRing + n * kRecordStride;
      const char* name = reinterpret_cast<const char*>(mem->TranslateVirtual<const uint8_t*>(rec));
      if (name && std::strncmp(name, prefix, plen) == 0) {
        wr32(rec + 2432u, 0);
        wr32(rec + 158368u, 0xFFFFFFFFu);
        wr32(rec + 80396u, 0xFFFFFFFFu);
        wr32(rec + 80396u + 77964u, 0xFFFFFFFFu);
      }
    }
  };
  const auto drop_game_list_cache = [&]() { drop_store_cache("storelist:alltitles"); };
  // Item pages keep their results in the registry's category records (8 of
  // them at +1266984, stride 166496, named): per subcategory a 13864-byte
  // page record holding two cached 32-item ranges (count at +6920 of each
  // 6928-byte range) and the stamped total at +13860. A page whose total is
  // stamped never asks the server again, so a filter change unstamps them.
  const auto drop_item_pages = [&](const char* prefix) {
    const size_t plen = std::strlen(prefix);
    for (uint32_t n = 0; n < 8; ++n) {
      const uint32_t base = 0x9426CF08u + 1266984u + 166496u * n;
      const char* name = reinterpret_cast<const char*>(mem->TranslateVirtual<const uint8_t*>(base));
      if (!name || std::strncmp(name, prefix, plen) != 0) continue;
      for (uint32_t slot = 0; slot < 11; ++slot) {
        const uint32_t page = base + 128u + 13864u * slot;
        wr32(page + 13860u, 0xFFFFFFFFu);
        wr32(page + 6920u, 0xFFFFFFFFu);
        wr32(page + 6928u + 6920u, 0xFFFFFFFFu);
      }
    }
  };
  // The open store page: the Game Styles list, or an item page, whose record
  // is named by its slot ("items:..." sections, "title:urn:uuid:..." games).
  const char* top_slot = "";
  {
    const int top = int(rd32(0x922E7604u));
    if (top >= 0 && top < 16) {
      const uint8_t* s = mem->TranslateVirtual<const uint8_t*>(0x922E7604u + 2188u * uint32_t(top) + 8u);
      if (s) top_slot = reinterpret_cast<const char*>(s);
    }
  }
  const bool item_page =
      !s_repush.pending && top_kind != kNavGameStyles && top_kind != kNavGameOpening &&
      top_kind != 68u && top_kind != 71u &&
      (std::strncmp(top_slot, "items:", 6) == 0 || std::strncmp(top_slot, "title:urn:uuid:", 15) == 0);
  ae_search::SetStorePage(top_kind == kNavGameStyles && !s_repush.pending
                              ? ae_search::StorePage::kGamesList
                              : item_page ? ae_search::StorePage::kItems : ae_search::StorePage::kNone);
  if (ae_search::ConsumeGamesReloadRequest() && (top_kind == kNavGameStyles || item_page)) {
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
    // Every store page enters through the loading page with its scene name,
    // the way a tile press does. A game page's scene is its slot; a section
    // page (kind 30 + n) has its scene in the registry's items table, where
    // slot n holds the name ("items:shoes"), the slot itself only naming the
    // shared "items:all" query.
    const bool game_page = std::strncmp(top_slot, "title:urn:uuid:", 15) == 0;
    // A section page (the All categories) enters through the loading page
    // with its scene name, which its nav record does not carry: the slot only
    // names the shared "items:all" query. The router maps the scene to a
    // category id (sub_920E1CA8) and that to the page kind (dword_92011040),
    // which is this table read backwards.
    if (!game_page && top_kind != kNavGameStyles && slot_ea) {
      static const struct { uint32_t kind; const char* scene; } kSections[] = {
          {37, "items:shirt"},    {38, "items:trouser"}, {39, "items:shoes"},
          {40, "items:hat"},      {41, "items:carryable"}, {42, "items:costume"},
          {43, "items:glasses"},  {44, "items:earrings"}, {45, "items:wristwear"},
          {46, "items:rings"},    {47, "items:gloves"},
      };
      const char* scene_name = nullptr;
      for (const auto& s : kSections) {
        if (s.kind == top_kind) scene_name = s.scene;
      }
      uint8_t* out = mem->TranslateVirtual<uint8_t*>(slot_ea);
      if (scene_name && out) {
        std::memset(out, 0, 0x7F);
        for (int i = 0; scene_name[i]; ++i) out[i] = uint8_t(scene_name[i]);
      }
    }
    const uint32_t push_kind = kNavGameOpening;
    if (top_kind == kNavGameStyles) {
      drop_game_list_cache();
    } else {
      // Both caches would otherwise answer the repeat without the server.
      drop_item_pages(game_page ? "title:urn:uuid:" : "items:");
      MktResetFetchers(game_page ? "title:urn:uuid:" : "items:");
    }
    PPCContext c = ctx;
    c.r3.u32 = 0x922E7604u;
    c.r4.u32 = prev;
    c.r5.u32 = 2;
    c.r6.u32 = 1;
    sub_920EDA10(c, base);
    s_repush = Repush{true, 0, slot_ea, title_ea, push_kind};
    REXKRNL_INFO("[xuisearch] store page reloading with filter '{}'",
                 top_kind == kNavGameStyles ? rex::kernel::xam::MarketplaceGamesFilter()
                                            : rex::kernel::xam::MarketplaceItemFilter());
  }
  // The push waits a frame for the pop to settle.
  bool just_pushed = false;
  if (s_repush.pending && ++s_repush.frames >= 2) {
    s_repush.pending = false;
    just_pushed = true;
    PPCContext c = ctx;
    c.r3.u32 = 0x922E7604u;
    c.r4.u32 = s_repush.kind;
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
      ae_search::SetSearchApplied(false);
    }
  }
  // An item filter belongs to the page it was applied in: leaving it drops
  // the filter and the filtered lists it cached.
  if (!rex::kernel::xam::MarketplaceItemFilter().empty() && !s_repush.pending && !just_pushed &&
      !item_page && top_kind != kNavGameOpening) {
    rex::kernel::xam::SetMarketplaceItemFilter("");
    drop_item_pages("items:");
    drop_item_pages("title:urn:uuid:");
    MktResetFetchers("items:");
    MktResetFetchers("title:urn:uuid:");
    if (rex::kernel::xam::GetAvatarCatalogSearch().empty()) {
      ae_search::SetSearchApplied(false);
    }
  }

  // A store match count arrives with the filtered response; re-post the box.
  {
    static int s_lastMatches = -2;
    const int matches = rex::kernel::xam::MarketplaceFilterMatches();
    if (matches != s_lastMatches) {
      s_lastMatches = matches;
      ae_search::Get().Refresh();
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
    ae_search::SetSearchLabelPossessed(false);
  }
}

// D3D::SetRawGammaRamp hands over the display-corrected ramp the XDK uploads to the DC_LUT, so keep a copy for the renderer.
void AE_HookRawGammaRamp(PPCRegister& r4) {
  if (!r4.u32) return;
  auto* mem = rex::system::kernel_memory();
  const uint8_t* p = mem->TranslateVirtual<const uint8_t*>(r4.u32);
  if (!p) return;
  for (uint32_t i = 0; i < 768; i++) {
    ae_gpu_gamma_ramp[i] = uint16_t((p[i * 2] << 8) | p[i * 2 + 1]);
  }
  ae_gpu_gamma_ramp_gen.fetch_add(1, std::memory_order_release);
}

// D3DDevice_Swap entry, used only as the once-per-frame point for the search tick.
void AE_XuiFrameTick() {
  AeXuiSearchTick(GuestCtx(), GuestBase());
}
