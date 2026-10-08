// Sub-quantum sleep for frame pacing: plain Sleep() rounds to the ~15.6 ms timer tick, a high-resolution waitable timer lands within ~0.5 ms.

#pragma once

#include <chrono>

#include <rex/thread.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace ae::thread {

// The guest thread is the frame's critical path, so it outranks the host's worker threads when cores are short.
inline void RaiseGuestThread() {
#if defined(_WIN32)
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
}

inline void LowerWorkerThread() {
#if defined(_WIN32)
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

inline void SleepPrecise(std::chrono::microseconds duration) {
  if (duration.count() < 100) {
    rex::thread::MaybeYield();
    return;
  }
#if defined(_WIN32)
  // One timer per calling thread, created lazily; the high-resolution flag needs Win10 1803+.
  struct TimerHolder {
    HANDLE handle;
    TimerHolder()
        : handle(CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)) {}
    ~TimerHolder() {
      if (handle) CloseHandle(handle);
    }
  };
  thread_local TimerHolder timer;
  if (timer.handle) {
    LARGE_INTEGER due_time;
    due_time.QuadPart = -static_cast<LONGLONG>(duration.count()) * 10;
    if (SetWaitableTimer(timer.handle, &due_time, 0, nullptr, nullptr, FALSE) &&
        WaitForSingleObject(timer.handle, INFINITE) == WAIT_OBJECT_0) {
      return;
    }
  }
#endif
  // No high-resolution timer: coarse-sleep the bulk, yield-spin the remainder.
  const auto target = std::chrono::steady_clock::now() + duration;
  if (duration > std::chrono::milliseconds(20)) {
    rex::thread::Sleep(std::chrono::duration_cast<std::chrono::milliseconds>(duration - std::chrono::milliseconds(20)));
  }
  while (std::chrono::steady_clock::now() < target) {
    rex::thread::MaybeYield();
  }
}

template <typename Rep, typename Period>
inline void SleepPrecise(std::chrono::duration<Rep, Period> duration) {
  SleepPrecise(std::chrono::duration_cast<std::chrono::microseconds>(duration));
}

}  // namespace ae::thread
