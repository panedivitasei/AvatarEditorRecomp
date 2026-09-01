// rexvideonative, render-thread offload queue (GPL-3.0, see LICENSE).
//
// The ring path pipelines translation across the guest and CP threads, so
// running it all inline on the guest thread costs throughput on weak CPUs.
// This module restores the two-core split: guest-side guards at the head of
// every mutating renderer::* entry point enqueue compact records
// (snapshotting only call-time-volatile guest data), and a worker thread
// re-enters the same renderer functions with a thread-local pass-through
// flag.
//
// Snapshot rules (mirrors the ring/PM4 contract):
//  - constants / state / texture-header dwords: captured at enqueue (the
//    XDK captures these into PM4 at call time).
//  - UP draw vertex/index bytes: copied at enqueue (the XDK copies UP data
//    into the ring at call time).
//  - device-block reads at draw time (fetch shadow, float files, bools,
//    point regs): kWinSize snapshot at enqueue, served to the unchanged
//    renderer code through a thread-local guest-read override window in
//    LoadGuestU32/GuestPtr.
//  - bulk VB/IB/texel data: read on the worker (ring-equivalent deferred
//    reads; the heal/graveyard machinery already tolerates those races).

#pragma once

#include <atomic>
#include <cstdint>

namespace rex::videonative {
struct ResolvedShader;
}

namespace rex::videonative::rq {

// Thread-local guest-read override windows (installed by the worker around
// records that captured device-block snapshots). LoadGuestU32/GuestPtr in
// renderer.cpp consult these before touching real guest memory.
// Covers device+0 .. +16KB. Must span every draw-path device read; the draw
// path reads up to dev+12308 (the raw z-enable request SetupDraw recomposes).
// A read of device+N with N >= kWinSize falls through to live guest memory on
// the worker, up to a frame stale, so any new one needs this raised or a
// capture of its own.
constexpr uint32_t kWinSize = 16384;
struct Win {
  const uint8_t* host = nullptr;
  uint32_t base = 0;
  uint32_t size = 0;
};
constexpr int kMaxWins = 24;
extern thread_local Win t_win[kMaxWins];

// True when the calling thread should enqueue instead of executing (queue
// running, cvar on, and not the worker itself).
bool Active();
// True on the guest render thread, the ring's only legal producer (false
// until its first present). Off-producer guest threads must not enqueue.
bool OnProducerThread();
// True when the worker is running at all (either thread).
bool Running();

// Spawn the worker (call once after renderer::Init succeeds). No-op when
// the native_video_render_thread cvar is off.
void Start();

// Flush the queue, stop and join the worker (Shutdown guard). Safe to call
// when never started.
void StopAndJoin();

// --- enqueue API (called from the renderer guards; one per entry point) ---
void EnqSetViewport(uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                    float min_z, float max_z);
void EnqSetScissor(int32_t l, int32_t t, int32_t r, int32_t b);
void EnqSetTexture(uint32_t sampler, uint32_t header, const uint32_t dw[6]);
void EnqSetStream(uint32_t stream, uint32_t vb, uint32_t off, uint32_t stride);
void EnqSetIndices(uint32_t ib);
void EnqSetShaderConstantsF(bool pixel, uint32_t start_reg,
                            uint32_t guest_data, uint32_t vec4_count);
void EnqSetShaderConstantsFHost(bool pixel, uint32_t start_reg,
                                const uint32_t* host_dwords,
                                uint32_t vec4_count);
void EnqPatchShaderFloatDwordsHost(bool pixel, uint32_t dword_index,
                                   const uint32_t* host_dwords,
                                   uint32_t count);
void EnqMergeShaderFloatDwordHost(bool pixel, uint32_t dword_index,
                                  uint32_t mask, uint32_t orv);
void EnqClear(uint32_t flags, uint32_t color, float z, const int32_t* rects,
              uint32_t rect_count, uint32_t stencil = 0);
void EnqRegisterSurface(uint32_t obj, uint32_t w, uint32_t h, uint32_t fmt);
void EnqSetRenderTargetSurface(uint32_t index, uint32_t obj);
void EnqSetDepthSurface(uint32_t obj);
void EnqSetTilingExtent(uint32_t w, uint32_t h);
void EnqInvalidateTexture(uint32_t header_addr);
void EnqNoteRtt(uint32_t texbase, uint32_t w, uint32_t h, uint32_t color_surf,
                uint32_t depth_surf, uint32_t dest_tex, uint32_t format,
                uint32_t msaa, uint32_t tiling);
void EnqResolveToTexture(uint32_t dest, uint32_t source, bool has_rect,
                         int32_t l, int32_t t, int32_t r, int32_t b,
                         int32_t dx, int32_t dy);
void EnqResolveDepthToTexture(uint32_t dest, bool has_rect, int32_t l,
                              int32_t t, int32_t r, int32_t b, int32_t dx,
                              int32_t dy);
void EnqSwapFrontbuffer(uint32_t header);
void EnqEndFrameAndPresent();  // includes the frame back-pressure wait
void EnqUpdateGammaRamp(const uint16_t* rgb768);
// Executes FlushResolveWritebacksInline on the worker. Returns a ticket;
// the flush has completed (guest memory holds the bytes) once
// WbFlushDone() >= ticket. Lifetime-free by design: a timed-out waiter's
// stack flag would dangle under a busy worker.
uint64_t EnqFlushResolveWriteback();
uint64_t WbFlushDone();
void EnqPushState();
void EnqPopState();
void EnqSeedReplayFloatConstants(uint32_t device, uint32_t recording_device);
void EnqSetReplayDrawState(const uint32_t* state10, uint32_t dirty_mask);
void EnqSetReplayBoolConstants(const uint32_t* dwords40);
void EnqSetReplayFetchConstants(const uint32_t* records, uint32_t count);
void EnqApplyReplayStatePersistent(const uint32_t* state10,
                                   uint32_t dirty_mask);
void EnqApplyBlendControlDirect(uint32_t rt, uint32_t value);
void EnqApplyLiveStateDirty(uint32_t consumed_mask, const uint32_t vals[5]);
void EnqDrawVertices(uint32_t device, uint32_t prim, uint32_t start_vertex,
                     uint32_t count);
void EnqDrawIndexedVertices(uint32_t device, uint32_t prim,
                            uint32_t base_vertex, uint32_t start_index,
                            uint32_t index_count);
void EnqDrawVerticesUP(uint32_t device, uint32_t prim, uint32_t vertex_count,
                       uint32_t vertex_data, uint32_t stride);
void EnqDrawIndexedVerticesUP(uint32_t device, uint32_t prim, uint32_t min_v,
                              uint32_t num_v, uint32_t index_count,
                              uint32_t index_data, uint32_t index_format,
                              uint32_t vertex_data, uint32_t stride);
// Guest-side pending UP override (replay path): consumed by the next UP
// draw enqueue instead of reading guest memory.
void SetPendingUPOverride(const void* vertex_data, const void* index_data);

// Cycle time of the guest (enqueue) thread for guest-vs-worker
// attribution; 0 before the first frame or inline.
uint64_t GuestThreadCycles();

// Write barrier for guest-memory rewrites (the freeze bracket): blocks the
// calling thread until the worker has executed every record committed so far.
// Big VBs (> kSnapVbMax) stay worker-read, so a rewrite that starts while
// their draws are still queued tears the read. No-op when the queue is off or
// on the worker itself.
void DrainForGuestRewrite();

}  // namespace rex::videonative::rq
