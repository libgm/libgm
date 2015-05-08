#ifndef LIBGM_MARKOV_CHAIN_HPP
#define LIBGM_MARKOV_CHAIN_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/base/discrete_process.hpp>
#include <libgm/math/logarithmic.hpp>

namespace libgm {

  template <typename F>
  class markov_chain {
    // Public type declarations
    //==========================================================================
  public:
    typedef typename F::real_type           real_type;
    typedef logarithmic<real_type>          result_type;
    typedef typename F::variable_type       variable_type;
    typedef typename F::domain_type         var_domain_type;
    typedef discrete_process<variable_type> process_type;
    typedef domain<process_type>           domain_type;
    typedef typename F::assignment_type     assignment_type;
    typedef typename F::index_type          var_index_type;
    typedef dynamic_matrix<real_type>       index_type; // for now

    //! Default constructor. Creates an empty chain.
    markov_chain()
      : order_(0) { }

    //! Creates a chain with the given processes.
    explicit markov_chain(const domain_type& processes, std::size_t order = 1)
      : processes_(processes), order_(order) { }

    //! Creates a chain with the given initial distribution and transition model
    markov_chain(const F& initial, const F& transition)
      : processes_(make_vector(discrete_processes(initial.arguments()))) {
      this->initial(initial);
      this->transition(transition);
    }

    //! Returns the processes associated with this chain.
    const domain_type& arguments() const {
      return processes_;
    }

    //! Returns the variables representing the current state of this chain
    var_domain_type current() const {
      return variables(processes_, current_step);
    }

    //! Returns the vairables representing the next state of this chain
    var_domain_type next() const {
      return variables(processes_, next_step);
    }

    //! Returns the order of the chain
    std::size_t order() const {
      return order_;
    }

    //! Returns the initial distribution
    const F& initial() const {
      return initial_;
    }

    //! Return the transition model
    const F& transition() const {
      return transition_;
    }

    //! Sets the initial distribution
    void initial(const F& factor) {
      initial_ = factor.reorder(current());
    }

    //! Sets the transition model
    void transition(const F& factor) {
      transition_ = factor.reorder(concat(next(), current()));
    }

    //! Returns the probability of a chain.
    result_type operator()(const index_type& index) const {
      return result_type(log(index), log_tag());
    }

    //! Returns the log-probability of a chain.
    real_type log(const index_type& index) const {
      real_type result(0);
      if (index.cols() > 0) {
        typedef typename F::ll_type ll_type;
        ll_type init_ll(initial_.param());
        result += init_ll.value(index.col(0));
        ll_type trans_ll(transition_.param());
        for (std::size_t t = 1; t < index.cols(); ++t) {
          result += trans_ll.value(index.col(t), idnex.col(t-1)); // FIXME
        }
      }
      return result;
    }

    //! Draws a random chain of the given length.
    template <typename Generator>
    index_type sample(Generator& rng, std::size_t len) const {
      var_index_type prior = initial_.sample(rng);
      index_type result = replicate(prior, len);
      if (len > 1) {
        auto cpd = transition_.distribution();
        for (std::size_t t = 1; t < len; ++t) {
          result.col(t) = cpd(rng, resul.col(t-1));
        }
      }
      return result;
    }

  private:
    domain_type processes_;
    std::size_t order_;
    F initial_;
    F transition_;

  }; // class markov_chain

} // namespace libgm

#endif
