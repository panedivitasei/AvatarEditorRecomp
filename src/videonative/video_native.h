// rexvideonative, native implementation of the guest-facing XDK D3D API.
//
// Ported from UnleashedRecomp (https://github.com/hedge-dev/UnleashedRecomp,
// GPL-3.0) onto the plume RHI. See LICENSE in this directory: this module is
// GPL-3.0, unlike the rest of the SDK.
//
// Architecture (see the xdk_d3d_mapping RE notes):
//   - This module is title-agnostic. It exposes the XDK D3D surface with
//     plain host types; the app provides the per-title glue:
//     REX_HOOK(sub_82XXXXXX, rex::videonative::XXX) for each mapped address.
//   - No ring buffer, no EDRAM, no resolve/tiling emulation by construction:
//     BeginTiling/EndTiling/SetPredication become no-ops, Resolve becomes a
//     copy/alias, render targets are host textures.
//   - Shaders come from the XenosRecomp native pack (see shader_cache.h).

#pragma once

#include <cstdint>

namespace rex::videonative {

using GuestAddr = uint32_t;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create the plume device/swapchain and internal caches. Called lazily by
// CreateDevice, or explicitly at app startup. Safe to call more than once.
bool Init();
void Shutdown();

// True when the cvar `native_video` is set. Gate for the CreateDevice hook.
bool Enabled();

// True when enabled and the host device initialized. App hooks should fall
// through to the recompiled guest code when this is false, so the ring-buffer
// path remains the fallback.
bool IsActive();

// The XDK patches VS microcode per binding (vfetch format/stride/fetch-reg
// rewritten from the vertex declaration + stream strides) before the CP sees
// it; the native pack is keyed on the patched bytes. The app registers the
// recompiled patcher (base sub_82536700 / TU1 sub_82545DD0) so resolution can
// produce bit-exact runtime ucode: (vs_obj, code_copy, decl, stride_table,
// variant). stride_table = 16 bytes, stride>>2 per stream.
using VfetchPatchFn = void (*)(uint32_t vs_obj, uint32_t code_copy,
                               uint32_t decl, uint32_t stride_table,
                               uint32_t variant);
void SetVfetchPatcher(VfetchPatchFn fn);

// Record the raw D3D device EA (real-SetRenderTarget r3) for the RB-shadow
// RT-key fold.
void NoteRawDevice(uint32_t device_ea);

// The guest engine destroyed a texture: drop the native cache entry keyed on
// that header. Call from the title's texture-release hook (TU1 0x829F3CE0,
// header at [r3+36]).
// Without it, a guest address reused for a different texture is served the
// previous texture's pixels. Guest-uploaded entries only; host-rendered
// targets are kept.
void InvalidateTextureByHeader(uint32_t header_addr);
void InvalidateSmallAlphaTextures();
// Texture CPU write complete (D3D Unlock hook, texture object r3).
void InvalidateTextureAfterUnlock(uint32_t header_addr);

// Guest-range texture invalidate/freeze, the kernel bridge. Kernel-side
// guest-memory writers (XamAvatarGetAssets buffer rewrites) call these
// directly at the write site; overlapping cache entries retire on the
// render thread before the next texture lookup. Thread-safe, callable
// before Init (they only queue).
void InvalidateGuestTextureRange(uint32_t guest_address, uint32_t size);
void FreezeGuestTextureRange(uint32_t guest_address, uint32_t size);

// Engine-authoritative RTT pass target: called from the
// title's StartTextureRendering hook with the built TEXTUREBASEDX9's fields:
// guest EA of the texture object, logical width/height (+8/+12), its
// EDRAM color/depth surface objects (+44/+72), the 2D resolve-dest
// D3DTexture (+28), D3DFORMAT (+96), MSAA type (+84) and tiling config
// index (+88, 999 = untiled). Values are read title-side so TU struct
// drift stays the title's concern. Cvar native_video_sem_rtt gates use.
void NoteRttBegin(uint32_t texbase, uint32_t width, uint32_t height,
                  uint32_t color_surf, uint32_t depth_surf, uint32_t dest_tex,
                  uint32_t format, uint32_t msaa, uint32_t tiling);

// Deliver pending resolve write-backs into guest memory, synchronously
// (returns after the bytes are readable). Call from the guest thread at
// the game's resolve fence wait, the point real hardware guarantees
// memory visibility. No-op when native_video_resolve_writeback is off or
// nothing is pending.
void FlushResolveWritebacksSync();

// True when write-back bytes must be delivered synchronously (legacy
// native_video_writeback_sync). Default false: callers on the guest thread,
// notably the title's fence hook, must not stall on delivery. Resolve
// products are sampled as textures, so the bytes are a fallback, not a
// read-after-write contract.
bool WritebackSyncEnabled();

// GPU fence writeback emulation: queue an EVENT_WRITE_SHD
// fence's writeback address; it receives an incrementing counter at the
// Next Swap (one-frame GPU completion lag). Called by the hook on the XDK
// fence-packet writer (base 0x8253FDD8 / TU1 0x8254F4A8).
void QueueFenceWriteback(GuestAddr addr);

// ---------------------------------------------------------------------------
// Guest D3D surface (another title's XDK vintage). Signatures mirror the
// mapped XDK functions, not PC D3D9; see xdk_d3d_mapping.md per entry.
// All functions take/return guest addresses; marshalling from PPCContext is
// done by the REX_HOOK auto-marshaller in the app-side hook table.
// ---------------------------------------------------------------------------

// --- device / swap ---
// Direct3D_CreateDevice (0x82533E10): the game creates a main and a secondary
// device; both map to the same host device here.
uint32_t CreateDevice(GuestAddr d3d, uint32_t type, uint32_t unk, uint32_t flags,
                      GuestAddr present_params, GuestAddr out_device);
// D3DDevice_Swap (another title 0x82532B50), takes the front buffer *texture*.
void Swap(GuestAddr device, GuestAddr front_buffer_tex, GuestAddr params);
void AcquireThreadOwnership(GuestAddr device);
void ReleaseThreadOwnership(GuestAddr device);

// --- render targets / tiling ---
void SetRenderTarget(GuestAddr device, uint32_t index, GuestAddr surface);
void SetDepthStencilSurface(GuestAddr device, GuestAddr surface, uint32_t flags);
// D3DDevice_SetSurfaces(device, D3DSURFACES*{DS,RT0..3}, flags); the
// unbind idiom is SetSurfaces(all-NULL, 1).
void SetSurfaces(GuestAddr device, GuestAddr surfaces, uint32_t flags);
// D3DDevice_CreateSurface (0x8252BB30): NO device arg; returns the surface
// object pointer in r3 (0 on failure).
uint32_t CreateSurface(uint32_t width, uint32_t height, uint32_t format,
                       uint32_t msaa, GuestAddr params);
// Predicated tiling: no-ops on host (rendering is full size, no EDRAM).
void BeginTiling(GuestAddr device, uint32_t flags, uint32_t count,
                 GuestAddr rects, GuestAddr clear_color, float clear_z,
                 uint32_t clear_stencil);
void EndTiling(GuestAddr device, uint32_t flags = 0,
               GuestAddr rects = 0, GuestAddr dest_texture = 0);
void SetPredication(GuestAddr device, uint32_t mask);
// Resolve (another title 0x82531058): the EDRAM->texture copy becomes an
// RT->texture alias/copy.
void Resolve(GuestAddr device, uint32_t flags, GuestAddr rect,
             GuestAddr dest_texture, GuestAddr dest_point);

// --- command buffers ---
// The game records public D3DDevice_* calls on the secondary device between
// Begin/End and replays them with Run (RE notes dec_0049.c). The hooked call
// stream is captured per CB object and re-invoked at Run time; run-varying
// state (constants/textures/viewport) is inherited from live state, matching
// the XDK inherit-tags contract (inherit-all in practice).
uint32_t BeginCommandBuffer(GuestAddr device, GuestAddr cb, uint32_t flags);
uint32_t EndCommandBuffer(GuestAddr device);
void RunCommandBuffer(GuestAddr device, GuestAddr cb,
                      uint32_t predication_select);
void SetCommandBufferPredication(GuestAddr device, uint32_t tile_pred,
                                 uint32_t run_pred);
// XDK internal direct RB_BLENDCONTROL[rt] PM4 emitter (base 0x82537E40):
// writes the register straight into the current ring/command buffer,
// bypassing the device shadow + dirty bits. The avatar per-entry material
// callback (0x829DDDB8) records predication-bracketed pairs of these,
// [(0,1) replace][(0,2) One/One], which is how one recorded CB draws the
// base pass opaque under Run(sel=1) and the light passes additive under
// Run(sel=2).
void SetBlendControlDirect(GuestAddr device, uint32_t rt, uint32_t value);

// --- clear / viewport / scissor ---
void Clear(GuestAddr device, uint32_t count, GuestAddr rects, uint32_t flags,
           uint32_t color, double z, uint32_t stencil);
void SetViewport(GuestAddr device, GuestAddr viewport);  // {X,Y,W,H,minZ,maxZ}
// Float-value variant for the shared SetViewport tail hook (0x921192D0,
// f1..f6), the only path that also sees D3DDevice_SetViewportF callers.
void SetViewportValues(GuestAddr device, float x, float y, float w, float h,
                       float min_z, float max_z);
void SetScissorRect(GuestAddr device, GuestAddr rect);

// --- resources ---
uint32_t CreateTexture(GuestAddr device, uint32_t width, uint32_t height,
                       uint32_t levels, uint32_t unk, uint32_t usage,
                       uint32_t format, uint32_t pool, uint32_t type);
uint32_t CreateVertexBuffer(uint32_t byte_size, uint32_t flags);
uint32_t CreateIndexBuffer(uint32_t byte_size, uint32_t flags, uint32_t format);
void ReleaseResource(GuestAddr resource);
void ReleaseDevice(GuestAddr device);

// --- binding ---
void SetTexture(GuestAddr device, uint32_t sampler, GuestAddr texture);
void SetStreamSource(GuestAddr device, uint32_t stream, GuestAddr vb,
                     uint32_t offset, uint32_t unk, uint32_t stride);
void SetIndices(GuestAddr device, GuestAddr ib);
void SetVertexDeclaration(GuestAddr device, GuestAddr decl);

// --- shaders ---
// Set*Shader receive the XDK D3D shader object (created by the recompiled
// CreateVertexShader/CreatePixelShader, which stay unhooked): dword0 low
// nibble = resource type (6=VS, 7=PS); VS microcode ptr at +0x20, function
// header (code byte size at header+8) at +0x368; PS: ptr +0x18, header +0x28.
uint32_t CreateVertexShader(GuestAddr microcode);
uint32_t CreatePixelShader(GuestAddr microcode);
void SetVertexShader(GuestAddr device, GuestAddr shader);
void SetPixelShader(GuestAddr device, GuestAddr shader);
void SetVertexShaderConstantF(GuestAddr device, uint32_t start_reg,
                              GuestAddr data, uint32_t count);
void SetPixelShaderConstantF(GuestAddr device, uint32_t start_reg,
                             GuestAddr data, uint32_t count);
void SetShaderGPRAllocation(GuestAddr device, uint32_t flags,
                            uint32_t vs_count, uint32_t ps_count);  // no-op

// --- render states (XDK per-state setters) ---
void SetRenderState_ZEnable(GuestAddr device, uint32_t enable);
void SetRenderState_ZWriteEnable(GuestAddr device, uint32_t enable);
void SetRenderState_ZFunc(GuestAddr device, uint32_t func);
void SetRenderState_CullMode(GuestAddr device, uint32_t mode);  // GPU enc 0/2/6
void SetRenderState_AlphaTestEnable(GuestAddr device, uint32_t enable);
void SetRenderState_AlphaRef(GuestAddr device, uint32_t ref);
void SetRenderState_AlphaFunc(GuestAddr device, uint32_t func);
void SetBlendControl(GuestAddr device, uint32_t rt_index, uint32_t blend_bits);

// --- draws ---
void DrawVertices(GuestAddr device, uint32_t prim_type, uint32_t start,
                  uint32_t count);
void DrawIndexedVertices(GuestAddr device, uint32_t prim_type,
                         uint32_t base_vertex, uint32_t start, uint32_t count);
void DrawVerticesUP(GuestAddr device, uint32_t prim_type, uint32_t count,
                    GuestAddr vertex_data, uint32_t stride);
void DrawIndexedVerticesUP(GuestAddr device, uint32_t prim_type,
                           uint32_t min_index, uint32_t start,
                           uint32_t num_vertices, GuestAddr index_data,
                           uint32_t index_format, GuestAddr vertex_data,
                           uint32_t stride);

}  // namespace rex::videonative
