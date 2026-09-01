// rexvideonative renderer, draws the guest's D3D calls through plume using
// the XenosRecomp native pack (bindless DxbcShaderTranslator contract, see
// docs/native_shaders.md). GPL-3.0, see LICENSE.
//
// Guest resource creates stay recompiled. At draw time the referenced
// vertex/index ranges are uploaded into per-frame rings and the vertex-fetch 
// constants point at ring offsets (the pack shaders fetch vertices from a 
// ByteAddressBuffer at fetch-constant addresses, so this layer chooses
// the placement). Textures are cached host-side from guest headers.
// Render state is read from the shadow registers the recompiled XDK setters
// maintain inside the guest device block.

#pragma once

#include <cstdint>

namespace rex::videonative {

class ShaderCache;
struct ResolvedShader;

namespace renderer {

// Called from Init() after the plume device/queue/swapchain exist.
bool Init();
void Shutdown();

// Frame flow: EnsureFrameOpen() lazily acquires the backbuffer and opens the
// frame's command list; EndFrameAndPresent() closes, executes and presents.
void EnsureFrameOpen();
void EndFrameAndPresent();
// Resolve write-back (guest-visible resolve bytes). The engine samples
// resolve products as textures (see the texture-cache redirect); these
// bytes are a fallback path, not the mechanism. Pending count is
// cross-thread readable; the flush must run on the renderer-execution
// thread (worker under rq).
uint32_t ResolveWritebacksPending();
bool ResolveWritebackEnabled();
// Deliver write-back bytes synchronously at each resolve (legacy) instead of
// at frame end. Default off: the sync path stalls the guest thread on a fence
// per resolve.
bool WritebackSyncEnabled();
void FlushResolveWritebacksInline();
// The raw D3D device EA (the real SetRenderTarget's r3, a different
// object than the draw device); recorded by the game hook, consumed by
// the RB-shadow RT-key fold.
void NoteRawDevice(uint32_t device_ea);
// True when an undelivered writeback record targets this texture base,
// the guest LockRect hook flushes inline before the CPU read.
bool WritebackPendingForBase(uint32_t base);
// Guest LockRect observed on this texture base: its pages are
// CPU-authoritative, the redirect must not serve the raw resolve product
// over them (now or on future re-bakes of the same dest slot).
void NoteGuestTextureCpuLock(uint32_t base);

// State capture from hooks.
void SetViewport(uint32_t x, uint32_t y, uint32_t w, uint32_t h, float min_z,
                 float max_z);
void SetScissor(int32_t left, int32_t top, int32_t right, int32_t bottom);
void SetTexture(uint32_t sampler, uint32_t guest_texture_header);
// Worker-side variant: applies a bind whose 6 fetch dwords were captured at
// call time by the render queue (render_queue.h).
void SetTextureWithSnapshot(uint32_t sampler, uint32_t header,
                            const uint32_t* dw6);
void SetStream(uint32_t stream, uint32_t vb_object, uint32_t offset_bytes,
               uint32_t stride_bytes);
void SetIndices(uint32_t ib_object);
void SetShaderConstantsF(bool pixel, uint32_t start_reg, uint32_t guest_data,
                         uint32_t vec4_count);
// Host-order data variant for command-buffer replay (record-time capture).
void SetShaderConstantsFHost(bool pixel, uint32_t start_reg,
                             const uint32_t* host_dwords, uint32_t vec4_count);
// Dword-granular float patch/merge into the replay mirror, the XDK shader
// definition table's group-2 (raw byte copies) and group-3 (masked merges)
// operate on arbitrary 4-byte offsets of the ALU float block; replayed binds
// must reproduce them (the recompiled Set*Shader body that does it live never
// runs during replay). dword_index counts 4-byte units within the stage's
// 256-register float block.
void PatchShaderFloatDwordsHost(bool pixel, uint32_t dword_index,
                                const uint32_t* host_dwords, uint32_t count);
void MergeShaderFloatDwordHost(bool pixel, uint32_t dword_index, uint32_t mask,
                               uint32_t orv);
// rects: optional host-order {l,t,r,b} array (guest D3DRECTs pre-swapped by
// the caller), clears are scoped to them like the XDK's pRects.
// stencil: D3DCLEAR_STENCIL (0x20) clear value (low 8 bits).
void Clear(uint32_t device, uint32_t flags, uint32_t color, float z,
           const int32_t* rects = nullptr, uint32_t rect_count = 0,
           uint32_t stencil = 0);

// Adapter vendor, set once at device init (vendor-gated features: the
// native_video_vs_trim Intel-only mode).
void SetAdapterIsIntel(bool is_intel);

// Render targets / resolve / present composite.
void RegisterSurface(uint32_t surface_obj, uint32_t width, uint32_t height,
                     uint32_t guest_format);
// Bind a guest color surface to MRT slot index (0-3). Slot 0 = the primary
// surface (0 = swapchain); slots 1-3 contribute their per-slot formats to the
// RT set (LightingPass: 10-bit diffuse RT0 + 8888 specular RT1).
void SetRenderTargetSurface(uint32_t index, uint32_t surface_obj);
void SetDepthSurface(uint32_t surface_obj);
// Predicated tiling: while an extent is set (BeginTiling..EndTiling), host RTs
// for the bound tile-height EDRAM surfaces are expanded to cover the full
// frame (max tile right/bottom), so full-frame draws land at true positions
// and per-tile resolves can cut their bands back out. (0,0) = tiling off.
void SetTilingExtent(uint32_t width, uint32_t height);
// Guest texture destroyed, drop its cached upload (address-reuse safety).
void InvalidateTextureByHeader(uint32_t header_addr);
void InvalidateSmallAlphaTextures();
// Guest texture CPU write complete (D3D Unlock, texture object r3): retire
// the guest-uploaded entries over its range so the next bind re-uploads the
// finished texels. Any thread; vertex-buffer unlocks are ignored.
void InvalidateTextureAfterUnlock(uint32_t header_addr);
// Engine-authoritative RTT pass target (see native_video_sem_rtt):
// the title reports the TEXTUREBASEDX9 the engine is about to render into,
// true logical dims, its own EDRAM surface objects, and the resolve-dest
// texture. While the note's surfaces are the ones bound, host RT identity
// and dimensions come from it instead of EDRAM inference.
void NoteRttBegin(uint32_t texbase, uint32_t width, uint32_t height,
                  uint32_t color_surf, uint32_t depth_surf, uint32_t dest_tex,
                  uint32_t format, uint32_t msaa, uint32_t tiling);
// src rect (frame coordinates) and dest point honored per the XDK Resolve
// semantics; has_rect=false = full-surface copy to (0,0). source = color MRT
// index 0-3 (Resolve flags bits 0-2).
void ResolveToTexture(uint32_t dest_texture_header, uint32_t source,
                      bool has_rect, int32_t src_left, int32_t src_top,
                      int32_t src_right, int32_t src_bottom, int32_t dest_x,
                      int32_t dest_y);
// Depth resolve (Resolve source 4): copies the active RT's host depth into
// the destination texture (k_24_8-family header -> R32_FLOAT host texture).
void ResolveDepthToTexture(uint32_t dest_texture_header, bool has_rect,
                           int32_t src_left, int32_t src_top,
                           int32_t src_right, int32_t src_bottom,
                           int32_t dest_x, int32_t dest_y);
void SwapFrontbuffer(uint32_t frontbuffer_texture_header);
// Upload the game's display gamma ramp (host-order R[256],G[256],B[256]
// 16-bit words) into the present blit's LUT; call when the guest
// D3DGAMMARAMP block changes. Applied only while cvar native_video_gamma_ramp
// is set.
void UpdateGammaRamp(const uint16_t* rgb768);

// Guest-state snapshot around command-buffer replay.
void PushState();
void PopState();

// Seed the replay float-constant mirror from the live device float shadow
// (m_Constants.Alu: VS at dev+1920, PS at dev+6016). XDK Run contract: a
// command buffer inherits the device's current constants; recorded in-CB
// constant sets then overlay as the replay encounters them.
// recording_device (optional): registers the run device never set (whole vec4
// zero) are taken from the recording device's shadow instead. Constants
// written once on the recording device through inlined paths before any
// recording are still baked into the CB by the XDK (its dirty bits persist
// from device creation), which no record-time diff can see.
void SeedReplayFloatConstants(uint32_t device, uint32_t recording_device);

// Per-draw render-state override for command-buffer replay: the game sets
// depth/cull/blend/alpha state on the recording device's shadow block while
// recording, so replayed draws must use the state captured at record time,
// not the live (main-device) shadows. 10 dwords: RB_DEPTHCONTROL,
// RB_BLENDCONTROL0, RB_COLORCONTROL, PA_SU_SC_MODE_CNTL, RB_COLOR_MASK,
// RB_ALPHA_REF, composed ZEnable, RB_BLENDCONTROL1-3 (raw register values).
// nullptr clears the override. dirty_mask bit i = slot i was programmed on
// the recording device (authoritative); clear bits inherit the run device's
// live shadow at draw time instead, since XDK CBs bake only dirty registers
// and the lighting pass sets its blend/mask on the main device before Run
// (see CbCall::st_dirty).
void SetReplayDrawState(const uint32_t* state10, uint32_t dirty_mask);

// Guest-range texture invalidation (kernel bridge target): queue a rewritten
// guest range from any thread; overlapping cache entries retire on the render
// thread before the next texture lookup. Reached through the public
// rex::videonative::InvalidateGuestTextureRange / FreezeGuestTextureRange
// wrappers (video_native.h), which the title's kernel code calls directly.
void QueueGuestTextureInvalidate(uint32_t guest_address, uint32_t size);
void QueueGuestTextureFreeze(uint32_t guest_address, uint32_t size);
// Probing+memoized guest span resolver (nullptr = unreadable). For the rq
// enqueue capture: junk fetch-shadow pairs must never fault the copy.
const uint8_t* GuestDataPtrProbe(uint32_t addr, size_t bytes);
// Bool/loop constants (40 host-order dwords) captured from the recording
// device at record time; nullptr clears (live draws read the draw device's
// shadow block instead).
void SetReplayBoolConstants(const uint32_t* dwords40);
// Recording-device fetch-constant shadow records for the current replayed
// draw ({slot, dw0..dw5} x count): texture truth captured at record time,
// including the game's direct fetch-shadow pokes. nullptr clears.
void SetReplayFetchConstants(const uint32_t* records, uint32_t count);
// Apply a record-time state capture's authoritative blend/mask slots to the
// persistent register file (XDK RunCommandBuffer-during-recording semantics:
// pending state is flushed into the outer CB before a nested-run token).
void ApplyReplayStatePersistent(const uint32_t* state10, uint32_t dirty_mask);
// Direct RB_BLENDCONTROL[rt] register-file write (XDK no-shadow PM4 emit,
// the avatar CB's predicated blend spans). Bypasses shadow/dirty entirely.
void ApplyBlendControlDirect(uint32_t rt, uint32_t value);
// Fold the run device's live blend/mask dirty bits into the register file
// and drain them, the XDK Run pre-flush (live writes reach hardware before
// the CB's PM4). Call at RunCommandBuffer start; SetupDraw runs it for live
// draws only, so it cannot reorder against a CB's recorded direct blend
// writes.
void ConsumeLiveStateDirty(uint32_t device);
// Worker-side apply of the values ConsumeLiveStateDirty captured.
void ApplyLiveStateDirtyHost(uint32_t consumed_mask, const uint32_t vals[5]);

// Draw entry points (already routed through shader resolution by the caller).
void DrawVertices(uint32_t device, uint32_t prim_type, uint32_t start_vertex,
                  uint32_t vertex_count);
void DrawIndexedVertices(uint32_t device, uint32_t prim_type,
                         uint32_t base_vertex, uint32_t start_index,
                         uint32_t index_count);
// UP-draw source override: replayed CBs draw from the vertex/index bytes
// captured at record time (the XDK copies UP data into the CB, and the guest
// scratch has been rewritten by replay time). Set before a replayed UP draw,
// cleared after.
void SetUPDataOverride(const void* vertex_data, const void* index_data);

void DrawVerticesUP(uint32_t device, uint32_t prim_type, uint32_t vertex_count,
                    uint32_t vertex_data, uint32_t stride_bytes);
void DrawIndexedVerticesUP(uint32_t device, uint32_t prim_type,
                           uint32_t min_vertex, uint32_t num_vertices,
                           uint32_t index_count, uint32_t index_data,
                           uint32_t index_format, uint32_t vertex_data,
                           uint32_t stride_bytes);

}  // namespace renderer
}  // namespace rex::videonative
