/**
 ******************************************************************************
 * ReXGlue runtime
 ******************************************************************************
 *
 * Minimal host-side periodic callback, replacing xe::threading::PeriodicCallback
 * for the ported online subsystem. Runs `fn` on a background thread every
 * `interval` until destroyed.
 */

#ifndef REX_KERNEL_UTIL_PERIODIC_CALLBACK_H_
#define REX_KERNEL_UTIL_PERIODIC_CALLBACK_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace rex {
namespace kernel {
namespace threading {

class PeriodicCallback {
 public:
  template <typename Rep, typename Period>
  static std::unique_ptr<PeriodicCallback> CreateRepeating(
      std::chrono::duration<Rep, Period> interval, std::function<void()> fn,
      const std::string& name = "") {
    return std::unique_ptr<PeriodicCallback>(new PeriodicCallback(
        std::chrono::duration_cast<std::chrono::milliseconds>(interval),
        std::move(fn)));
  }

  ~PeriodicCallback() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  PeriodicCallback(std::chrono::milliseconds interval, std::function<void()> fn)
      : interval_(interval), fn_(std::move(fn)) {
    thread_ = std::thread([this]() {
      std::unique_lock<std::mutex> lock(mutex_);
      while (!stop_) {
        if (cv_.wait_for(lock, interval_, [this]() { return stop_; })) {
          break;
        }
        lock.unlock();
        fn_();
        lock.lock();
      }
    });
  }

  std::chrono::milliseconds interval_;
  std::function<void()> fn_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace threading
}  // namespace kernel
}  // namespace rex

#endif
