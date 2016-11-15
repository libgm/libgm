#ifndef LIBGM_MOMENT_GAUSSIAN_TRANSFORM_HPP
#define LIBGM_MOMENT_GAUSSIAN_TRANSFORM_HPP

#include <libgm/enable_if.hpp>
#include <libgm/factor/expression/moment_gaussian_base.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

#include <type_traits>

namespace libgm { namespace experimental {

  /**
   * A class that represents a unary transform of a moment_gaussian.
   *
   * \tparam VectorOp
   *         A unary operation accepting an Eigen matrix expression and
   *         returning a matrix expression.
   * \tparam ScalarOp
   *         A unary operation accepting a real type (the log-multiplier) and
   *         returning a real type. Must be associative with addition.
   * \tparam F
   *         A moment_gaussian expression type.
   */
  template <typename VectorOp, typename ScalarOp, typename F>
  class moment_gaussian_transform
    : public moment_gaussian_base<
        std::result_of_t<ScalarOp(typename F::real_type)>,
        moment_gaussian_transform<VectorOp, ScalarOp, F> > {

  public:
    // shortcuts
    using real_type  = std::result_of_t<ScalarOp(typename F::real_type)>;
    using param_type = moment_gaussian_param<real_type>;
    using base = moment_gaussian_base<real_type, moment_gaussian_transform>;

    moment_gaussian_transform(VectorOp vector_op, ScalarOp scalar_op, const F& f)
      : vector_op_(vector_op), scalar_op_(scalar_op), f_(f) { }

    std::size_t head_arity() const {
      return f_.head_arity();
    }

    std::size_t tail_arity() const {
      return f_.tail_arity();
    }

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.param().transform(vector_op_, scalar_op_, result);
    }

    using base::multiply_inplace; // in case the one here is disabled

    LIBGM_ENABLE_IF_N((std::is_same<VectorOp, identity>::value),
                      typename IndexRange)
    void multiply_inplace(const IndexRange& dims, param_type& result) const {
      f_.multiply_inplace(dims, result);
      result.lm = scalar_op_(result.lm);
      // works b/c scalar_op_ is associative with addition
    }

    template <typename OuterVectorOp, typename OuterScalarOp>
    moment_gaussian_transform<compose_t<OuterVectorOp, VectorOp>,
                              compose_t<OuterScalarOp, ScalarOp>,
                              F>
    transform(OuterVectorOp vector_op, OuterScalarOp scalar_op) const {
      return { compose(vector_op, vector_op_),
               compose(scalar_op, scalar_op_),
               f_ };
    }

  private:
    VectorOp vector_op_;
    ScalarOp scalar_op_;
    add_const_reference_if_factor_t<F> f_;

  }; // class moment_gaussian_transform

  /**
   * Constructs a moment_gaussian_transform object, deducing its type.
   * \relates moment_gaussian_transform
   */
  template <typename VectorOp, typename ScalarOp, typename F>
  inline moment_gaussian_transform<VectorOp, ScalarOp, F>
  make_moment_gaussian_transform(VectorOp vector_op, ScalarOp scalar_op,
                                 const F& f) {
    return { vector_op, scalar_op, f };
  }

} } // namespace libgm::experimental

#endif
