// Native renderer entry points used by the app shell and the capture hooks.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu/gpu_records.h"

namespace rex::ui {
class Window;
struct RawImage;
}  // namespace rex::ui

namespace ae::gpu {

// Creates the plume device and swap chain on the window and starts the render thread; false leaves the renderer off.
bool Initialize(rex::ui::Window* window);

// Stops the render thread, idles the device and destroys everything in reverse order; safe to call twice.
void Shutdown();

// The window's pixel size changed; the render thread rebuilds the swap chain before its next present.
void RequestResize();

// Guest thread: appends a record to the frame being built.
void Submit(Record record);

// Guest thread: hands the frame to the render thread, waiting while two frames are already queued.
void EndFrame();

// Frames handed to the render thread so far; the capture keys per-frame upload coalescing on it.
uint64_t FrameIndex();

// Guest thread: adds capture time and captured upload bytes to the frame statistics.
void AddCaptureStats(uint64_t nanoseconds, uint64_t bytes);

// Guest thread: time spent in one part of building a draw (0 shaders and state, 1 constants, 2 streams and
// textures, 3 geometry), reported per frame in the statistics window.
void AddCaptureBreakdown(int section, uint64_t nanoseconds);

// Frame timings: 0 translation, 1 decode, 2 indices, 3 streams, 4 texture description, 5 texture key,
// 6 resource lookup, 7 shader lookup, 8 preparation finish, 9 texture binding, 10 registration,
// 11 stream copy, 12 stream query, 13 unowned checks; kind 14 accumulates copied bytes instead of nanoseconds.
void AddGuestTiming(int kind, uint64_t nanoseconds);
// Range keys the host dropped from shared memory since the last call; the guest thread forgets it sent them. A key
// carrying kEvictedTextureTag names a destroyed host texture id instead of a range.
constexpr uint64_t kEvictedTextureTag = 1ull << 63;
void TakeEvicted(std::vector<uint64_t>& keys);
bool AnyEvicted();

// First-frame hold until the start-up pipeline warm-up has compiled (bounded by gpu_pipeline_warmup_wait).
void WaitForWarmup();

// Any thread: replaces the window title on the UI thread; empty restores the one the window was created with.
void SetWindowTitle(std::string title);

// Vertical sync on the swap chain, applied by the host thread at its next present.
void SetVsync(bool enabled);

// Frame pacing on the guest thread: kind 0 = time slept for gpu_fps_cap, 1 = time waiting for queue room.
void AddFrameTiming(int kind, uint64_t nanoseconds);

// Guest thread: copies the last presented front buffer as RGBX8, waiting for the render thread.
bool ReadbackFrontBuffer(rex::ui::RawImage& image);

// Any thread: the guest wrote [base, base + size) behind the D3D runtime's back, so cached host copies are retired.
void MarkRangeDirty(uint32_t base, uint32_t size);

namespace draw {
// Guest thread: submits a Begin*Vertices draw whose End the game inlined; every commit point calls it first.
void FinishPending();
}  // namespace draw

}  // namespace ae::gpu
