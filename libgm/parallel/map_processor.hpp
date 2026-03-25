#pragma once

#include <ankerl/unordered_dense.h>

#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace libgm {

/**
 * A class that processes entries stored in a map in parallel. For each entry,
 * the processor invokes the specified function in a non-deterministic order.
 * The function accepts three arguments: the key, the mapped value, and a state
 * object (one for each thread) that the function can modify. The state objects
 * are maintained by the caller and passed together with the map via operator().
 *
 * See also the template specialization MapProcessor<Key, Result, void> for the
 * version that does not use state.
 *
 * \tparam Key a type that represents a map key
 * \tparam Result a type that represents a mapped result
 * \tparam State a type that represents state
 */
template <typename Key, typename Result, typename State = void>
class MapProcessor {
public:
  using map_type = ankerl::unordered_dense::map<Key, Result>;

  /**
   * Constructs a map processor with the given processing function.
   */
  explicit MapProcessor(std::function<void(const Key&, Result&, State&)> fn)
    : fn_(std::move(fn)) {}

  /**
   * Invokes the processing function on each map entry using the specified state.
   *
   * \param jobs A map of jobs processed in parallel.
   * \param state A vector of state objects. The length of this vector
   *              determines the number of threads. The vector must not
   *              be empty.
   */
  void operator()(map_type& jobs, std::vector<State>& state) {
    assert(!state.empty());
    if (state.size() == 1) {
      for (auto& [key, result] : jobs) {
        fn_(key, result, state.front());
      }
    } else {
      typename map_type::iterator it = jobs.begin();
      const typename map_type::iterator last = jobs.end();
      std::mutex mutex;
      std::vector<std::thread> threads;
      for (std::size_t t = 0; t < state.size(); ++t) {
        threads.emplace_back([this, t, &it, &last, &mutex, &state] {
          while (true) {
            typename map_type::iterator current;
            {
              std::lock_guard<std::mutex> lock(mutex);
              if (it == last) {
                return;
              }
              current = it++;
            }

            auto& [key, result] = *current;
            fn_(key, result, state[t]);
          }  
        });
      }
      for (std::thread& thread : threads) {
        thread.join();
      }
    }
  }

private:
  std::function<void(const Key&, Result&, State&)> fn_;
};

/**
 * A class that processes entries stored in a map in parallel. For each entry,
 * the processor invokes the specified function in a non-deterministic order.
 * The function accepts two arguments: the key and mapped value. To process a
 * map of jobs, pass it via operator(), along with the number of threads.
 *
 * \tparam Key a type that represents a map key
 * \tparam Result a type that represents a mapped result
 */
template <typename Key, typename Result>
class MapProcessor<Key, Result, void> {
public:
  using map_type = ankerl::unordered_dense::map<Key, Result>;

  /**
   * Constructs a map processor with the given processing function.
   */
  explicit MapProcessor(std::function<void(const Key&, Result&)> fn)
    : fn_(std::move(fn)) {}

  /**
   * Invokes the function on each map entry in parallel.
   * \param jobs A map of jobs processed in parallel.
   * \param nthreads The number of worker threads (must be >0).
   */
  void operator()(map_type& jobs, std::size_t nthreads) {
    assert(nthreads > 0);
    if (nthreads == 1) {
      for (auto& [key, result] : jobs) {
        fn_(key, result);
      }
    } else {
      typename map_type::iterator it = jobs.begin();
      const typename map_type::iterator last = jobs.end();
      std::mutex mutex;
      std::vector<std::thread> threads;
      for (std::size_t t = 0; t < nthreads; ++t) {
        threads.emplace_back([this, &it, &last, &mutex] {
          while (true) {
            typename map_type::iterator current;
            {
              std::lock_guard<std::mutex> lock(mutex);
              if (it == last) {
                return;
              }
              current = it++;
            }

            auto& [key, result] = *current;
            fn_(key, result);
          }
        });
      }
      for (std::thread& thread : threads) {
        thread.join();
      }
    }
  }

private:
  std::function<void(const Key&, Result&)> fn_;
};

} // namespace libgm
