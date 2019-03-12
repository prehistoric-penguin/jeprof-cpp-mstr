#pragma once

#include <condition_variable>
#include <cstdio>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <memory>
#include <mutex>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace detail {
class function_wrapper {
  struct impl_base {
    virtual void call() = 0;
    virtual ~impl_base() {}
  };
  std::unique_ptr<impl_base> impl;
  template <typename F> struct impl_type : impl_base {
    F f;
    impl_type(F &&f_) : f(std::move(f_)) {}
    void call() { f(); }
  };

public:
  template <typename F>
  function_wrapper(F &&f) : impl(new impl_type<F>(std::move(f))) {}

  function_wrapper() = default;

  void operator()() { call(); }

  void call() { impl->call(); }

  operator bool() { return impl.operator bool(); }

  function_wrapper(function_wrapper &&other) : impl(std::move(other.impl)) {}

  function_wrapper &operator=(function_wrapper &&other) {
    impl = std::move(other.impl);
    return *this;
  }

  function_wrapper(const function_wrapper &) = delete;
  function_wrapper(function_wrapper &) = delete;
  function_wrapper &operator=(const function_wrapper &) = delete;
};

template <typename T> class threadsafe_queue {
private:
  mutable std::mutex mut;
  std::deque<T> data_queue;
  std::condition_variable data_cond;
  std::atomic_bool exit_flag{false};

public:
  threadsafe_queue() {}
  threadsafe_queue(threadsafe_queue const &other) {
    std::lock_guard<std::mutex> lk(other.mut);

    data_queue = other.data_queue;
  }

  void push(T &&new_value) {
    std::lock_guard<std::mutex> lk(mut);

    data_queue.push_back(std::move(new_value));
    data_cond.notify_one();
  }

  void wait_and_pop(T &value) {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { 
        return exit_flag || !data_queue.empty() || exit_flag.load(); 
    });
    if (exit_flag.load()) return;
    value = std::move(data_queue.front());
    data_queue.pop_front();
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this] { return exit_flag || !data_queue.empty() || exit_flag.load(); });
    if (exit_flag) return {};

    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop_front();

    return res;
  }

  bool try_pop(T &value) {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty())
      return false;
    value = std::move(data_queue.front());
    data_queue.pop_front();
    return true;
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty())
      return std::shared_ptr<T>();
    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop_front();

    return res;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.empty();
  }

  void clear() { 
    exit_flag = true; 
    std::lock_guard<std::mutex> lk(mut);
    data_cond.notify_all();
  }
};

class thread_pool {
  std::atomic_bool done;
  threadsafe_queue<function_wrapper> work_queue;
  std::vector<std::thread> threads;

  void worker_thread() {
    while (!done) {
      function_wrapper task;
      work_queue.wait_and_pop(task);
      if (task) task();
    }
  }

public:
  thread_pool(size_t pool_size = 0) : done(false) {
    size_t thread_count = 0;
    if (!pool_size) {
      thread_count = std::thread::hardware_concurrency();
    } else {
      thread_count = pool_size;
    }
    try {
      for (unsigned i = 0; i < thread_count; ++i) {
        threads.push_back(std::thread(&thread_pool::worker_thread, this));
      }
    } catch (...) {
      done = true;
      throw;
    }
  }

  ~thread_pool() {
    work_queue.clear();
    done = true;
    for (auto &thr : threads)
      thr.join();
  }

  template <typename FunctionType>
  std::future<typename std::result_of<FunctionType()>::type>
  submit(FunctionType f) {
    typedef typename std::result_of<FunctionType()>::type result_type;

    std::packaged_task<result_type()> task(std::move(f));
    std::future<result_type> res(task.get_future());
    work_queue.push(std::move(task));
    return res;
  }
};

template <typename Iterator, typename T>
void parallel_fill(Iterator first, Iterator last, T val) {
  uint64_t const length = std::distance(first, last);

  if (!length) return;

  uint64_t const block_size = 1024 * 1024 * 1024;
  uint64_t const num_blocks = (length + block_size - 1) / block_size;

  std::vector<std::future<void>> futures(num_blocks - 1);
  thread_pool pool;
  std::cerr << "Now begin with:" << num_blocks << " blocks" << std::endl;

  Iterator block_start = first;
  for (uint64_t i = 0; i < (num_blocks - 1); ++i) {
    Iterator block_end = block_start;
    std::advance(block_end, block_size);
    std::cerr << "block start:" << std::hex
              << reinterpret_cast<intptr_t>(block_start)
              << ", last:" << reinterpret_cast<intptr_t>(block_end)
              << std::endl;

    futures[i] = pool.submit([block_start, block_end, val]() {
      std::fill(block_start, block_end, val);
    });
    block_start = block_end;
  }
  std::cerr << "block start:" << std::hex
            << reinterpret_cast<intptr_t>(block_start)
            << ", last:" << reinterpret_cast<intptr_t>(last) << std::endl;
  std::fill(block_start, last, val);

  for (uint64_t i = 0; i < (num_blocks - 1); ++i) {
    futures[i].get();
  }
}
} // namespace
