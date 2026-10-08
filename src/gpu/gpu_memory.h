// Guest-thread tracking of GPU-visible guest memory: vertex, index and texture ranges registered by the D3D and XG
// lifetime functions, dirtied by Lock, Unlock and header changes; gpu_paranoid hashes referenced ranges to catch
// CPU writers the hooks miss.
#pragma once

#include <cstdint>
#include <memory>

namespace ae::gpu::memory {

struct RangeState {
  bool registered = false;   // inside a range a lifetime hook registered
  uint32_t generation = 0;   // changes on every dirty event touching the range
};

struct Range;
struct QueryCache {
  uint64_t key = 0;
  uint64_t revision = 0;
  std::shared_ptr<Range> range;
};

void Register(uint32_t base, uint32_t size);
void MarkDirty(uint32_t base, uint32_t size);
RangeState Query(uint32_t base, uint32_t size);
RangeState Query(uint32_t base, uint32_t size, QueryCache& cached);

// Registers or dirties the data range of a D3D resource header (vertex buffer, index buffer or texture).
void RegisterResource(uint32_t resource_address);
void DirtyResource(uint32_t resource_address);

bool Paranoid();

// Paranoid mode: remembers the contents a static range had when it was uploaded, and reports once per range when
// a later draw finds them changed with no tracked write. Returns true when the range must be uploaded again.
void NoteUploaded(uint64_t key, uint32_t base, uint32_t size, const void* bytes);
bool CheckUnchanged(uint64_t key, uint32_t base, uint32_t size, const void* bytes);

// Logs once per range that it is re-uploaded every draw because no lifetime hook owns it.
void NoteUnregistered(uint32_t base, uint32_t size, const char* what);

}  // namespace ae::gpu::memory
