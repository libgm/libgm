#ifndef LIBGM_MOMENT_GAUSSIAN_RESTRICT_HEAD_HPP
#define LIBGM_MOMENT_GAUSSIAN_RESTRICT_HEAD_HPP

#include <libgm/factor/experimental/expression/moment_gaussian_base.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

namespace libgm { namespace experimental {

  /**
   * A class that represents a restriction of a moment_gaussian to a range of
   * its head dimensions.
   *
   * This class cannot be evaluated unless the restricted moment_gaussian is
   * marginal, but it can be used in moment_gaussian::operator*= to condition
   * on evidence.
   */
  template <typename IndexRange, typename F>
  class moment_gaussian_restrict_head
    : public moment_gaussian_base<
        typename F::real_type,
        moment_gaussian_restrict_head<IndexRange, F> > {
  public:
    // Shortcuts
    using real_type  = typename F::real_type;
    using param_type = moment_gaussian_param<real_type>;

    moment_gaussian_restrict_head(IndexRange dims,
                                  const real_vector<real_type>& values,
                                  const F& f)
      : dims_(dims), values_(values), f_(f) {
      assert(dims.size() == values.size());
    }

    std::size_t head_arity() const {
      return f_.head_arity() - values_.size();
    }

    std::size_t tail_arity() const {
      return f_.tail.arity();
    }

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.param().restrict_head(complement(dims_, f_.head_arity()),
                               dims_, values_, ws_, result);
    }

    template <typename It>
    void multiply_in(index_range<It> join_dims, param_type& result) const {
      if (head_arity() != 0) {
        throw std::invalid_argument(
          "moment_gaussian::operator*= may not introduce arguments"
        );
      }
      f_.param().restrict_head_multiply(join_dims, dims_, values_, result);
    }

  private:
    IndexRange dims_;
    const real_vector<real_type>& values_;
    add_const_reference_if_factor_t<F> f_;
    mutable typename param_type::restrict_workspace ws_;

  }; // class moment_gaussian_restrict_head

} } // namespace libgm::experimental

#endif
