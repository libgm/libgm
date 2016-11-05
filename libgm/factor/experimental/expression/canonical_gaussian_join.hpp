#ifndef LIBGM_CANONICAL_GAUSSIAN_JOIN_HPP
#define LIBGM_CANONICAL_GAUSSIAN_JOIN_HPP

#include <libgm/factor/experimental/expression/canonical_gaussian_base.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/compose_assign.hpp>
#include <libgm/math/param/canonical_gaussian_param.hpp>
#include <libgm/range/index_range_map.hpp>

namespace libgm { namespace experimental {

  /**
   * An expression that represents a binary join of two canonical_gaussians
   * f and g. The selected dimensions of f must match the selected dimension
   * of g.
   *
   * \tparam JoinOp
   *         A binary operator that accepts two scalars or matrix expressions.
   * \tparam FDims
   *         A type that models IndexRange and denotes the left join dimensions.
   * \tparam GDims
   *         A type that models IndexRange and denotes the right join dimensions.
   * \tparam F
   *         A canonical_gaussian expression representing the left argument
   *         of the join.
   * \tparam G
   *         A canonical_gaussian expression representing the right argument
   *         of the join.
   */
  template <typename JoinOp,
            typename FDims, typename GDims, typename F, typename G>
  class canonical_gaussian_join
    : public canonical_gaussian_base<
        typename F::real_type,
        canonical_gaussian_join<JoinOp, FDims, GDims, F, G> > {

    static_assert(
      std::is_same<typename F::real_type, typename G::real_type>::value,
      "The joined expressions must have the same real type");

  public:
    // Shortcuts
    using real_type  = typename F::real_type;
    using param_type = canonical_gaussian_param<real_type>;

    //! Constructs a canonical_gaussian_join
    canonical_gaussian_join(JoinOp join_op, FDims fdims, GDims gdims,
                            const F& f, const G& g)
      : join_op_(join_op), fdims_(fdims), gdims_(gdims), f_(f), g_(g) {
      assert(fdims_.size() == gdims_.size());
      arity_ = f_.arity() + g_.arity() - fdims_.size();
    }

    std::size_t arity() const {
      return arity_;
    }

    bool alias(const param_type& param) const {
      return f_.alias(param) || g_.alias(param);
    }

    void eval_to(param_type& result) const {
      result.zero(arity());
      transform_inplace(assign<>(), result);
    }

    template <typename UpdateOp>
    void transform_inplace(UpdateOp update_op, param_type& result) const {
      f_.join_inplace(update_op, front(f_.arity()), result);
      g_.join_inplace(compose_assign(update_op, join_op_),
                      map_right(fdims_, gdims_, f_.arity(), g_.arity()),
                      result);
    }

  private:
    JoinOp join_op_;
    FDims fdims_;
    GDims gdims_;
    add_const_reference_if_factor_t<F> f_;
    add_const_reference_if_factor_t<G> g_;
    std::size_t arity_;

  }; // class canonical_gaussian_join

} } // namespace libgm::experimental

#endif
