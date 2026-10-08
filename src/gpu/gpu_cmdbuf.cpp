// Command-buffer record and replay. A recorded entry keeps the recording device's whole struct, the vertex shader
// bytes as patched for that draw, and the union of the D3DTAG bits the runtime had pending at each flush since
// BeginCommandBuffer; at RunCommandBuffer those tagged registers, the shaders and the draw's own buffers come from
// the recording and every other register from the live device, which is what the runtime's register inheritance
// does for a buffer whose inherit mask is all ones.

#include "gpu/gpu_cmdbuf.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <memory>
#include <unordered_map>

#include <rex/cvar.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>

#include "gpu/gpu.h"
#include "gpu/gpu_memory.h"
#include "gpu/guest/d3d_device.h"

REXCVAR_DECLARE(int32_t, gpu_trace_frame);
REXCVAR_DEFINE_BOOL(gpu_cmdbuf_watch, false, "GPU",
                    "Log when the live texture fetch constants or palette stream a command-buffer replay inherits change");

namespace ae::gpu::cmdbuf {

namespace {

using guest::D3DDevice;

constexpr size_t kDeviceBytes = sizeof(D3DDevice);

template <typename T>
const T* Guest(uint32_t address) {
  return REX_KERNEL_MEMORY()->TranslateVirtual<const T*>(address);
}

bool Tracing() {
  const int32_t frame = REXCVAR_GET(gpu_trace_frame);
  return frame >= 0 && ae::gpu::FrameIndex() == uint64_t(frame);
}

struct Entry {
  bool clear = false;
  draw::DrawArgs draw;
  capture::ClearArgs clear_args;
  std::vector<uint8_t> device;   // the recording device when the entry was flushed
  draw::VsSnapshot vs;           // the vertex shader as patched for this draw
  uint64_t tags[5] = {};         // D3DTAG bits the recording wrote by then, MSB-first per mask
};

struct Buffer {
  uint32_t device = 0;
  uint32_t flags = 0;
  std::shared_ptr<const std::vector<Entry>> entries;
};

struct RecordingState {
  uint32_t buffer = 0;
  uint32_t flags = 0;
  uint64_t begin_pending[5] = {};  // tags pending before Begin: inherited, not recorded
  uint64_t tags[5] = {};
  uint32_t run = 0;
  std::vector<Entry> entries;
};

std::mutex g_mutex;
std::unordered_map<uint32_t, RecordingState> g_recording;  // by device
std::unordered_map<uint32_t, Buffer> g_buffers;            // by buffer address
std::atomic<bool> g_warn_inline{false};
std::atomic<bool> g_warn_unknown{false};
std::atomic<int> g_log_budget{80};   // the first recordings and runs are logged without the frame trace

bool Loud() { return Tracing() || g_log_budget.fetch_sub(1) > 0; }

void ReadPending(uint32_t device, uint64_t (&out)[5]) {
  const auto* d = Guest<D3DDevice>(device);
  for (int i = 0; i < 5; ++i) out[i] = d->m_Pending.m_Mask[i];
}

inline bool Bit(const uint64_t (&tags)[5], int mask, int bit) { return (tags[mask] >> (63 - bit)) & 1; }

// Copies the register range one D3DTAG bit names from the recording into the merged struct.
void CopyTag(uint8_t* merged, const uint8_t* snap, int mask, int bit) {
  auto copy = [&](size_t offset, size_t size) { std::memcpy(merged + offset, snap + offset, size); };
  switch (mask) {
    case 0: copy(offsetof(D3DDevice, m_Constants.VertexShaderF) + 64 * bit, 64); break;
    case 1: copy(offsetof(D3DDevice, m_Constants.PixelShaderF) + 64 * bit, 64); break;
    case 2:
      if (bit >= 6 && bit <= 21) {
        copy(offsetof(D3DDevice, m_DestinationPacket) + 4 * (bit - 6), 4);
        // The surface objects behind the colour and depth info registers travel with them.
        if (bit >= 7 && bit <= 11 && bit != 8) {
          const int n = bit == 7 ? 0 : bit - 8;
          copy(offsetof(D3DDevice, render_targets) + 4 * n, 4);
        }
        if (bit == 8) copy(offsetof(D3DDevice, depth_stencil), 4);
      } else if (bit >= 22 && bit <= 42) {
        copy(offsetof(D3DDevice, m_ValuesPacket) + 4 * (bit - 22), 4);
      } else if (bit == 43) {
        copy(offsetof(D3DDevice, pixel_shader), 4);
      } else if (bit == 44) {
        copy(offsetof(D3DDevice, vertex_shader), 4);
      } else if (bit >= 47 && bit <= 51) {
        copy(offsetof(D3DDevice, m_ProgramPacket) + 4 * (bit - 47), 4);
      } else if (bit >= 52) {
        copy(offsetof(D3DDevice, m_ControlPacket) + 4 * (bit - 52), 4);
      }
      break;
    case 3:
      if (bit >= 9 && bit <= 29) copy(offsetof(D3DDevice, m_TessellatorPacket) + 4 * (bit - 9), 4);
      else if (bit >= 32) copy(offsetof(D3DDevice, m_Constants.Fetch) + sizeof(guest::FetchConstant) * (bit - 32), sizeof(guest::FetchConstant));
      break;
    case 4:
      if (bit == 7) {
        // The flow constants: bool and loop words of both shaders, one tag.
        copy(offsetof(D3DDevice, m_Constants.VertexShaderB),
             offsetof(D3DDevice, m_ClipPlanes) - offsetof(D3DDevice, m_Constants.VertexShaderB));
      } else if (bit >= 15 && bit <= 17) {
        copy(offsetof(D3DDevice, m_WindowPacket) + 4 * (bit - 15), 4);
      } else if (bit >= 18 && bit <= 25) {
        copy(offsetof(D3DDevice, m_PointPacket) + 4 * (bit - 18), 4);
      } else if (bit >= 26) {
        copy(offsetof(D3DDevice, m_MiscPacket) + 4 * (bit - 26), 4);
      }
      break;
    default: break;
  }
}

// Live device plus the recorded registers and the recorded draw's own index buffer, declaration and stream strides.
std::vector<uint8_t> Merge(uint32_t live_device, const Entry& e, uint32_t select) {
  std::vector<uint8_t> merged(Guest<uint8_t>(live_device), Guest<uint8_t>(live_device) + kDeviceBytes);
  const uint8_t* snap = e.device.data();
  for (int mask = 0; mask < 5; ++mask) {
    if (!e.tags[mask]) continue;
    for (int bit = 0; bit < 64; ++bit) {
      if (Bit(e.tags, mask, bit)) CopyTag(merged.data(), snap, mask, bit);
    }
  }
  auto copy = [&](size_t offset, size_t size) { std::memcpy(merged.data() + offset, snap + offset, size); };
  copy(offsetof(D3DDevice, index_buffer), 4);
  copy(offsetof(D3DDevice, vertex_declaration), 4);
  copy(offsetof(D3DDevice, stream_stride_dwords), guest::kStreams);
  // Shader binds are tags 2:43 and 2:44 and every avatar recording carries them (the variant masks are zeroed
  // before CreateDrawCommands); copying the pointers outright gives the same result and keeps the replay whole
  // if a recording ever inherits a shader.
  copy(offsetof(D3DDevice, vertex_shader), 4);
  copy(offsetof(D3DDevice, pixel_shader), 4);
  (void)select;
  return merged;
}

}  // namespace

bool Recording(uint32_t device) {
  std::lock_guard lock(g_mutex);
  return g_recording.contains(device);
}

void Begin(uint32_t device, uint32_t buffer, uint32_t flags) {
  RecordingState s;
  s.buffer = buffer;
  s.flags = flags;
  ReadPending(device, s.begin_pending);
  std::lock_guard lock(g_mutex);
  g_recording[device] = std::move(s);
  if (Loud()) REXGPU_INFO("[gpu] command buffer {:#010x} recording on device {:#010x} flags {:#x}", buffer, device, flags);
}

void End(uint32_t device) {
  std::lock_guard lock(g_mutex);
  auto it = g_recording.find(device);
  if (it == g_recording.end()) return;
  Buffer& b = g_buffers[it->second.buffer];
  b.device = device;
  b.flags = it->second.flags;
  b.entries = std::make_shared<const std::vector<Entry>>(std::move(it->second.entries));
  // Buffers with draws (the avatar's) are logged every time, with the tags they carry, so an intermittent replay
  // fault can be read back from the log; the tile buffers only count against the budget.
  if (!b.entries->empty()) {
    const Entry& last = b.entries->back();
    REXGPU_INFO("[gpu] command buffer {:#010x} sealed with {} entries, flags {:#x}, tags {:#018x} {:#018x} {:#018x} "
                "{:#018x} {:#018x}, vs {:#010x} variant {} ({} bytes), fetch30 {:#010x}",
                it->second.buffer, b.entries->size(), b.flags, last.tags[0], last.tags[1], last.tags[2], last.tags[3],
                last.tags[4], last.vs.object, last.vs.variant1 ? 1 : 0, last.vs.code.size(),
                uint32_t(reinterpret_cast<const D3DDevice*>(last.device.data())->m_Constants.Fetch[30].dword[4]));
  } else if (Loud()) {
    REXGPU_INFO("[gpu] command buffer {:#010x} sealed with 0 entries", it->second.buffer);
  }
  g_recording.erase(it);
}

void SetPredication(uint32_t device, uint32_t run) {
  std::lock_guard lock(g_mutex);
  if (auto it = g_recording.find(device); it != g_recording.end()) it->second.run = run;
}

void NotePending(uint32_t device) {
  std::lock_guard lock(g_mutex);
  auto it = g_recording.find(device);
  if (it == g_recording.end()) return;
  // BeginCommandBuffer 0x92128978 overwrites the pending masks with ~inherit, so what is
  // pending at a flush is exactly what the recording set since Begin plus the non-inherited registers.
  uint64_t pending[5];
  ReadPending(device, pending);
  for (int i = 0; i < 5; ++i) it->second.tags[i] |= pending[i];
}

void RecordDraw(const draw::DrawArgs& a) {
  std::lock_guard lock(g_mutex);
  auto it = g_recording.find(a.device);
  if (it == g_recording.end()) return;
  if (a.inline_data) {
    if (!g_warn_inline.exchange(true)) REXGPU_WARN("[gpu] a Begin/UP draw inside a command buffer is not replayed");
    return;
  }
  Entry e;
  e.draw = a;
  e.device.assign(Guest<uint8_t>(a.device), Guest<uint8_t>(a.device) + kDeviceBytes);
  e.vs = draw::SnapshotVs(Guest<D3DDevice>(a.device));
  std::memcpy(e.tags, it->second.tags, sizeof(e.tags));
  if (Loud()) {
    const auto* d = Guest<D3DDevice>(a.device);
    const uint32_t vs = d->vertex_shader;
    const uint32_t header = vs ? uint32_t(*Guest<rex::be<uint32_t>>(vs + guest::kVsHeader)) : 0;
    REXGPU_INFO("[gpu] recorded draw: vs {:#010x} header {:#x} strides {} {} {} {} fetch30 {:#010x} {:#010x}", vs, header,
                d->stream_stride_dwords[0], d->stream_stride_dwords[1], d->stream_stride_dwords[2],
                d->stream_stride_dwords[3], uint32_t(d->m_Constants.Fetch[30].dword[4]),
                uint32_t(d->m_Constants.Fetch[30].dword[5]));
  }
  it->second.entries.push_back(std::move(e));
}

void RecordClear(uint32_t device, const capture::ClearArgs& a) {
  std::lock_guard lock(g_mutex);
  auto it = g_recording.find(device);
  if (it == g_recording.end()) return;
  Entry e;
  e.clear = true;
  e.clear_args = a;
  e.device.assign(Guest<uint8_t>(device), Guest<uint8_t>(device) + kDeviceBytes);
  std::memcpy(e.tags, it->second.tags, sizeof(e.tags));
  it->second.entries.push_back(std::move(e));
}

void Run(PPCContext& ctx, uint8_t* base, uint32_t device, uint32_t buffer, uint32_t select) {
  std::shared_ptr<const std::vector<Entry>> entries;
  uint32_t recorded_on = 0;
  {
    std::lock_guard lock(g_mutex);
    auto it = g_buffers.find(buffer);
    if (it == g_buffers.end()) {
      if (!g_warn_unknown.exchange(true)) REXGPU_WARN("[gpu] RunCommandBuffer on a buffer with no recording");
      return;
    }
    entries = it->second.entries;
    recorded_on = it->second.device;
  }
  if (!entries) return;
  if (Loud()) {
    REXGPU_INFO("[gpu] run command buffer {:#010x} select {} on device {:#010x}: {} entries", buffer, select, device,
                entries->size());
  }
  if (!entries->empty() && REXCVAR_GET(gpu_cmdbuf_watch)) {
    // Diagnostic: report when the live texture fetch constants a replay inherits change between runs, and when
    // the palette stream does, so a face or pose that stays still can be told from a binding that never moves.
    const auto* live = Guest<D3DDevice>(device);
    const uint64_t fetch_hash = XXH3_64bits(&live->m_Constants.Fetch[0], sizeof(guest::FetchConstant) * 26);
    const uint32_t palette = live->m_Constants.Fetch[30].dword[4];
    static std::unordered_map<uint32_t, std::pair<uint64_t, uint32_t>> last;
    static std::atomic<int> budget{200};
    std::lock_guard lock(g_mutex);
    auto& l = last[buffer];
    if ((l.first != fetch_hash || l.second != palette) && budget.fetch_sub(1) > 0) {
      REXGPU_INFO("[gpu] run {:#010x} frame {}: live texture fetches {:#018x}{}, palette {:#010x}{}", buffer,
                  ae::gpu::FrameIndex(), fetch_hash, l.first != fetch_hash ? " (changed)" : "", palette,
                  l.second != palette ? " (changed)" : "");
    }
    l = {fetch_hash, palette};
  }
  // The avatar's bone palette (stream 3, fetch constant 92) is CPU-written every frame, so the
  // half the live device points at is dirtied before the replay reads it.
  {
    const auto* live = Guest<D3DDevice>(device);
    const uint32_t palette = uint32_t(live->m_Constants.Fetch[30].dword[4]) & ~3u;
    const uint32_t bytes = uint32_t(live->m_Constants.Fetch[30].dword[5]) & 0x3FFFFFC;
    if (palette && bytes) memory::MarkDirty(palette & 0x1FFFFFFF, bytes);
  }
  for (const Entry& e : *entries) {
    if (Tracing()) {
      const auto* live = Guest<D3DDevice>(device);
      REXGPU_INFO("[gpu]   entry {} tags {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} live fetch30 {:#010x} {:#010x}",
                  e.clear ? "clear" : "draw", e.tags[0], e.tags[1], e.tags[2], e.tags[3], e.tags[4],
                  uint32_t(live->m_Constants.Fetch[30].dword[4]), uint32_t(live->m_Constants.Fetch[30].dword[5]));
    }
    const std::vector<uint8_t> merged = Merge(device, e, select);
    const auto* merged_device = reinterpret_cast<const D3DDevice*>(merged.data());
    if (e.clear) {
      ae::gpu::Submit(capture::MakeClear(merged_device, e.clear_args));
    } else {
      draw::DrawArgs a = e.draw;
      a.device = recorded_on;
      draw::Replay(ctx, base, a, merged_device, e.vs);
    }
  }
}

}  // namespace ae::gpu::cmdbuf
