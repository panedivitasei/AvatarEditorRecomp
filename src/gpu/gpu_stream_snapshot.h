#pragma once

#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ae::gpu {

// Workers allocate and touch destination pages; guest bytes are copied at the draw boundary.
// A slot returns through a mutex after its last reader releases the snapshot.
class StreamSnapshotPool {
  struct Slot {
    std::vector<uint8_t> bytes;
    bool free = true;
  };
  struct State {
    std::mutex mutex;
    std::vector<Slot> slots;
    bool ready = false;
  };
 public:
  StreamSnapshotPool() : state_(std::make_shared<State>()), warm_([state = state_] {
      std::vector<Slot> slots;
      slots.reserve(168);
      for (auto [size, count] : {std::pair<size_t, size_t>{256u << 10, 128},
                                {1u << 20, 32}, {4u << 20, 8}}) {
        for (size_t i = 0; i < count; ++i) slots.push_back({std::vector<uint8_t>(size), true});
      }
      std::lock_guard lock(state->mutex);
      state->slots = std::move(slots);
      state->ready = true;
    }) {}

  void WaitUntilWarm() { if (warm_.joinable()) warm_.join(); }

  std::shared_ptr<const std::vector<uint8_t>> Copy(const uint8_t* source, size_t size) {
    size_t index = SIZE_MAX;
    {
      std::lock_guard lock(state_->mutex);
      if (state_->ready) {
        for (size_t i = 0; i < state_->slots.size(); ++i) {
          auto& slot = state_->slots[i];
          if (slot.free && slot.bytes.size() >= size) {
            slot.free = false;
            index = i;
            break;
          }
        }
      }
    }
    if (index == SIZE_MAX) return std::make_shared<const std::vector<uint8_t>>(source, source + size);
    auto* bytes = &state_->slots[index].bytes;
    std::shared_ptr<const std::vector<uint8_t>> snapshot(bytes, [state = state_, index](const auto*) {
      std::lock_guard lock(state->mutex);
      state->slots[index].free = true;
    });
    if (size) std::memcpy(bytes->data(), source, size);
    return snapshot;
  }

 private:
  std::shared_ptr<State> state_;
  std::jthread warm_;
};

}  // namespace ae::gpu
