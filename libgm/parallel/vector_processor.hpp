#ifndef LIBGM_VECTOR_PROCESSOR_HPP
#define LIBGM_VECTOR_PROCESSOR_HPP

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

namespace libgm {

  /**
   * A class that processes jobs stored in a vector in parallel. For each job
   * the processor invokes the specified function in a non-deterministic order.
   * The function accepts two arguments: the job, and state object (one for each
   * thread) that the function can modify. The state objects are maintained by
   * the caller and passed together with jobs via operator().
   *
   * See also the template specialization vector_processor<Job, void> for
   * version that does not use state.
   *
   * \tparam Job a type that represents a job
   * \tparam State a type that represents state
   */
  template <typename Job, typename State = void>
  class vector_processor {
  public:
    /**
     * Constructs a vector processor with the given processing function.
     */
    explicit vector_processor(std::function<void(const Job&, State&)> fn)
      : fn_(std::move(fn)) { }

    /**
     * Invokes the processing function on each job using the specified state.
     *
     * \param jobs A vector of jobs processed in parallel.
     * \param state A vector of state objects. The length of this vector
     *              determines the number of threads. The vector must not
     *              be empty.
     */
    void operator()(const std::vector<Job>& jobs, std::vector<State>& state) {
      assert(!state.empty());
      if (state.size() == 1) {
        for (const Job& job : jobs) {
          fn_(job, state.front());
        }
      } else {
        std::atomic<std::size_t> index(0);
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < state.size(); ++t) {
          threads.emplace_back([&] {
              std::size_t i;
              while ((i = index++) < jobs.size()) {
                fn_(jobs[i], state[i]);
              }
            });
        }
        for (std::size_t t = 0; t < threads.size(); ++t) {
          threads[t].join();
        }
      }
    }

  private:
    //! The function that processes each job.
    std::function<void(const Job&, State&)> fn_;

  }; // class vector_processor


  /**
   * A class that processes jobs stored in a vector in parallel. For each job
   * the processor invokes the specified function in a non-deterministic order.
   * The function accepts a single arguments: the job. To process a vector of
   * jobs, pass them via operator(), along with the number of threads.
   *
   * \tparam Job a type that represents a job
   */
  template <typename Job>
  class vector_processor<Job, void> {
  public:
    /**
     * Constructs a vector processor with the given processing function.
     */
    vector_processor(std::function<void(const Job&)> fn)
      : fn_(std::move(fn)) { }

    /**
     * Invokes the function on each job in parallel.
     * \param jobs A vector of jobs processed in parallel.
     * \param nthreads The number of worker threads (must be >0).
     */
    void operator()(const std::vector<Job>& jobs, std::size_t nthreads) {
      assert(nthreads > 0);
      if (nthreads == 1) {
        for (const Job& job : jobs) {
          fn_(job);
        }
      } else {
        std::atomic<std::size_t> index(0);
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < nthreads; ++t) {
          threads.emplace_back([&] {
              std::size_t i;
              while ((i = index++) < jobs.size()) {
                fn_(jobs[i]);
              }
            });
        }
        for (std::size_t t = 0; t < nthreads; ++t) {
          threads[t].join();
        }
      }
    }

  private:
    std::function<void(const Job&)> fn_;

  }; // class vector_processor<Job, void>

} // namespace libgm

#endif
