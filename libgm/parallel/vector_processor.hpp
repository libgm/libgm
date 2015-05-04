#ifndef LIBGM_VECTOR_PROCESSOR_HPP
#define LIBGM_VECTOR_PROCESSOR_HPP

#include <libgm/global.hpp>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

namespace libgm {

  /**
   * A class that processes jobs stored in a vector in parallel.
   * \tparam Job the type that represents a job
   */
  template <typename Job>
  class vector_processor {
  public:
    /**
     * Constructs a vector processor with the given number of threads
     * and the function to be applied to each job.
     */
    vector_processor(size_t nthreads, std::function<void(const Job&)> fn)
      : nthreads_(nthreads), fn_(std::move(fn)) {
      assert(nthreads_ > 0);
    }
    
    /**
     * Invokes the function on each job in parallel.
     */
    void operator()(const std::vector<Job>& jobs) {
      if (nthreads_ == 1) {
        for (const Job& job : jobs) {
          fn_(job);
        }
      } else {
        index_ = 0;
        std::vector<std::thread> threads;
        for (size_t t = 0; t < nthreads_; ++t) {
          threads.emplace_back([&] {
              size_t i = 0;
              while ((i = index_++) < jobs.size()) {
                fn_(jobs[i]);
              }
            });
        }
        for (size_t t = 0; t < nthreads_; ++t ) {
          threads[t].join();
        }
      }
    }

  private:
    size_t nthreads_;
    std::function<void(const Job&)> fn_;
    std::atomic<size_t> index_;

  }; // class vector_processor
  
} // namespace libgm

#endif
