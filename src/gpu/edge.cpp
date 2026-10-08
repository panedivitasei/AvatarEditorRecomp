// Edge layer: the GPU-facing side of the XDK D3D runtime with no GPU behind it.
// Hook points and evidence are in config/ae_hooks.toml and docs/research/gpu_edge.md.

#include <atomic>
#include <cstdint>

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc/context.h>
#include <rex/ppc/func.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xthread.h>
#include <rex/types.h>

#include "gpu/guest/d3d_device.h"

namespace {

using ae::gpu::guest::D3DDevice;

template <typename T>
T* Guest(uint32_t address) {
  return REX_KERNEL_MEMORY()->TranslateVirtual<T*>(address);
}

}  // namespace

// CDevice::SetFence 0x92125D90, at the store of the advanced counter (0x92125EC4): r6 is the fence being issued, r5
// the tagged ring position it marks, r11 the device. Writes the same two words as SetFence's own CPU-completion branch
// at 0x92125EB0, so fence and ring-space waits never block.
void AeEdge_FenceIssued(PPCRegister& r5, PPCRegister& r6, PPCRegister& r11) {
  const auto* device = Guest<const D3DDevice>(r11.u32);
  const uint32_t writeback = device->writeback_page;
  if (writeback) {
    auto* words = Guest<rex::be<uint32_t>>(writeback);
    words[0] = r6.u32;
    words[1] = r5.u32;
  }
}

// CDevice::AddCallsToPrimaryBuffer 0x92125808, at the store of the new primary write index (0x92125B70): r11 is the
// index, r24 the device. The ring has no reader, so its read-pointer writeback (+0x3C) follows the write index and
// BlockOnPrimaryRange 0x92124C18 never waits.
void AeEdge_PrimaryKick(PPCRegister& r11, PPCRegister& r24) {
  const auto* device = Guest<const D3DDevice>(r24.u32);
  const uint32_t writeback = device->writeback_page;
  if (writeback) {
    Guest<rex::be<uint32_t>>(writeback)[0x3C / 4] = r11.u32;
  }
}

// After each add to queued_chunks in QueueIndirectBuffer 0x92125ED0, r31 the device. The GPU subtracts as it retires
// buffers, so the count drops straight back to idle and BlockUntilIdle 0x921267B8 returns.
void AeEdge_ChunkQueued(PPCRegister& r31) {
  Guest<D3DDevice>(r31.u32)->queued_chunks = 0;
}

// D3DDevice_InsertCallback 0x92126830 (device, type, callback, context): on hardware the callback address travels in
// the ring and a GPU interrupt runs it. Nothing reads the ring, so it runs here.
REX_HOOK_RAW(sub_92126830) {
  const uint32_t callback = ctx.r5.u32;
  if (!callback) return;
  ctx.r3.u64 = ctx.r6.u32;
  rex::runtime::ResolveIndirectFunction(callback)(ctx, base);
}

// KeInsertQueueDpc: the SDK queues DPCs and never runs them. Its only D3D caller is KickOffOnWorkerQueue 0x92135060,
// whose routine hands a chunk to the XPS worker, so the routine runs on insert.
REX_HOOK_RAW(__imp__KeInsertQueueDpc) {
  const uint32_t dpc_address = ctx.r3.u32;
  auto* dpc = Guest<rex::system::XDPC>(dpc_address);
  const uint32_t routine = dpc->routine;
  dpc->arg1 = ctx.r4.u32;
  dpc->arg2 = ctx.r5.u32;
  if (routine) {
    ctx.r6.u64 = dpc->arg2;
    ctx.r5.u64 = dpc->arg1;
    ctx.r4.u64 = dpc->context;
    ctx.r3.u64 = dpc_address;
    rex::runtime::ResolveIndirectFunction(routine)(ctx, base);
  }
  ctx.r3.u64 = 1;
}

// Vd* ring and interrupt setup, which the SDK only warns about without a GPU plugin. The edge layer owns them; the
// ring and writeback stay in the guest memory D3D allocated.
namespace ae::gpu::edge {

std::atomic<uint32_t> g_interrupt_callback{0};
std::atomic<uint32_t> g_interrupt_user_data{0};

void VdSetGraphicsInterruptCallback_entry(rex::u32 callback, rex::u32 user_data) {
  g_interrupt_callback.store(callback);
  g_interrupt_user_data.store(user_data);
}

void VdInitializeRingBuffer_entry(rex::u32 ring, rex::i32 size_log2) {
  (void)ring;
  (void)size_log2;
}

void VdEnableRingBufferRPtrWriteBack_entry(rex::u32 writeback, rex::i32 block_size_log2) {
  (void)writeback;
  (void)block_size_log2;
}

}  // namespace ae::gpu::edge

REX_EXPORT(__imp__VdSetGraphicsInterruptCallback, ae::gpu::edge::VdSetGraphicsInterruptCallback_entry)
REX_EXPORT(__imp__VdInitializeRingBuffer, ae::gpu::edge::VdInitializeRingBuffer_entry)
REX_EXPORT(__imp__VdEnableRingBufferRPtrWriteBack, ae::gpu::edge::VdEnableRingBufferRPtrWriteBack_entry)
