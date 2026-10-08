#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ae::gpu {

// Jobs own their inputs and never access guest memory; a full queue runs work on the submitting thread.
class PreparationQueue {
 public:
  struct Stats {
    uint64_t queued, inline_jobs, work_ns;
  };
  Stats TakeStats() {
    return {queued_.exchange(0), inline_.exchange(0), work_ns_.exchange(0)};
  }

  explicit PreparationQueue(size_t workers = 2, size_t capacity = 64, size_t byte_limit = 64u << 20)
      : capacity_(capacity), byte_limit_(byte_limit) {
    try {
      for (size_t i = 0; i < workers; ++i) workers_.emplace_back([this] { Run(); });
    } catch (...) {
      Stop();
      throw;
    }
  }
  ~PreparationQueue() { Stop(); }
  PreparationQueue(const PreparationQueue&) = delete;
  PreparationQueue& operator=(const PreparationQueue&) = delete;

  template <typename Work>
  auto Post(Work work, size_t bytes) {
    using Result = std::invoke_result_t<Work>;
    auto owned = std::make_shared<std::packaged_task<Result()>>(std::move(work));
    auto result = owned->get_future();
    PostDetached([owned] { (*owned)(); }, bytes);
    return result;
  }

  void PostDetached(std::function<void()> task, size_t bytes) {
    {
      std::lock_guard lock(mutex_);
      if (!stopping_ && !workers_.empty() && jobs_.size() < capacity_ &&
          bytes <= byte_limit_ && bytes_ <= byte_limit_ - bytes) {
        jobs_.push_back(Job{std::move(task), bytes});
        bytes_ += bytes;
        ++pending_;
        queued_.fetch_add(1, std::memory_order_relaxed);
        ready_.notify_one();
        return;
      }
      ++pending_;
    }
    inline_.fetch_add(1, std::memory_order_relaxed);
    Execute(Job{std::move(task), 0});
  }

  void Drain() {
    std::unique_lock lock(mutex_);
    finished_.wait(lock, [this] { return pending_ == 0; });
    auto failure = std::exchange(failure_, {});
    lock.unlock();
    if (failure) std::rethrow_exception(failure);
  }

 private:
  struct Job {
    std::function<void()> task;
    size_t bytes;
  };
  void Execute(Job job) {
    const auto start = std::chrono::steady_clock::now();
    std::exception_ptr failure;
    try { job.task(); } catch (...) { failure = std::current_exception(); }
    work_ns_.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
    std::lock_guard lock(mutex_);
    bytes_ -= job.bytes;
    if (failure && !failure_) failure_ = failure;
    if (--pending_ == 0) finished_.notify_all();
  }
  void Stop() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_) if (worker.joinable()) worker.join();
  }
  void Run() {
    for (;;) {
      Job job;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
        if (jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      Execute(std::move(job));
    }
  }
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable finished_;
  std::deque<Job> jobs_;
  std::vector<std::thread> workers_;
  size_t capacity_, byte_limit_, bytes_ = 0;
  size_t pending_ = 0;
  std::exception_ptr failure_;
  bool stopping_ = false;
  std::atomic<uint64_t> queued_{0}, inline_{0}, work_ns_{0};
};

}  // namespace ae::gpu
