#ifndef LIBGM_GAUSSIAN_FACTOR_HPP
#define LIBGM_GAUSSIAN_FACTOR_HPP

#include <libgm/argument/traits.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/vector_map.hpp>
#include <libgm/factor/base/factor.hpp>

#include <numeric>
#include <sstream>
#include <vector>

namespace libgm {

  /**
   * The base class of all Guassian factors.
   *
   * \ingroup factor_types
   */
  template <typename Arg>
  class gaussian_factor : public factor {
    static_assert(is_continuous<Arg>::value,
                  "Gaussian factors require Arg to be continuous");

    typedef argument_traits<Arg> arg_traits;

  public:
    typedef domain<Arg> domain_type;

    // Constructors and indexing
    //==========================================================================

    //! Default constructor.
    gaussian_factor() { }

    //! Initializes the argument index to the given arguments.
    explicit gaussian_factor(const domain_type& args) {
      compute_start(args);
    }

    //! Initializes the argument index to the given domains.
    gaussian_factor(const domain_type& args1, const domain_type& args2) {
      compute_start(args1, args2);
    }

    //! Returns the dimensionality of a marginal Gaussian with given arguments.
    static std::size_t param_shape(const domain_type& args) {
      return args.num_dimensions();
    }

    //! Returns the start map.
    const vector_map<Arg, std::size_t>& start() const {
      return start_;
    }

  protected:
    // Protected members
    //==========================================================================

    //! Assigns a starting index span to each argument in an increasing order.
    std::size_t compute_start(const domain_type& args) {
      start_.clear();
      start_.reserve(args.size());
      std::size_t m = args.insert_start(start_);
      start_.sort();
      return m;
    }

    //! Assigns a starting index span to each argument in an increasing order.
    std::pair<std::size_t, std::size_t>
    compute_start(const domain_type& args1, const domain_type& args2) {
      start_.clear();
      start_.reserve(args1.size() + args2.size());
      std::size_t m = args1.insert_start(start_);
      std::size_t n = args2.insert_start(start_);
      start_.sort();
      return std::make_pair(m, n);
    }

    //! Renames the arguments and the variable-index span map
    void subst_args(const std::unordered_map<Arg, Arg>& map) {
      start_.subst_keys(map);
    }

    //! The base implementation of swap.
    void base_swap(gaussian_factor& other) {
      swap(start_, other.start_);
    }

    //! The map from each variable to its index span
    vector_map<Arg, std::size_t> start_;

  }; // class gaussian_factor

} // namespace libgm

#endif
