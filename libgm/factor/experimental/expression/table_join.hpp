#ifndef LIBGM_TABLE_JOIN_HPP
#define LIBGM_TABLE_JOIN_HPP

#include <libgm/datastructure/table.hpp>
#include <libgm/factor/experimental/expression/table_base.hpp>
#include <libgm/factor/experimental/expression/table_function.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/range/index_range_map.hpp>

namespace libgm { namespace experimental {

  /**
   * An expression that represents a join of two tables f and g using a binary
   * operator. The selected dimensions of f must precisely match the selected
   * dimensions of g. This results in a table whose front dimensions
   * correspond to dimensions of f and the back dimensions correspond to
   * non-selected dimensions of g.
   *
   * \tparam Space
   *         A tag denoting the space of the table (prob_tag or log_tag).
   * \tparam JoinOp
   *         A binary operation accepting real_type and returning real_type.
   * \tparam FDims
   *         A type that models IndexRange and denotes the left join dimensions.
   * \tparam GDims
   *         A type that models IndexRange and denotes the right join dimensions.
   * \tparam F
   *         A table expression representing the left argument of the join.
   * \tparam G
   *         A table expression representing the irhgt argument of the join.
   */
  template <typename Space, typename JoinOp,
            typename FDims, typename GDims, typename F, typename G>
  class table_join
    : public table_base<
        Space,
        typename F::real_type,
        table_join<Space, JoinOp, FDims, GDims, F, G> > {
    static_assert(
      std::is_same<typename F::real_type, typename G::real_type>::value,
      "The joined expressions must have the same real type");

  public:
    // Shortcuts
    using real_type  = typename F::real_type;
    using param_type = table<real_type>;

    table_join(JoinOp join_op, FDims fdims, GDims gdims, const F& f, const G& g)
      : join_op_(join_op), fdims_(fdims), gdims_(gdims), f_(f), g_(g) {
      assert(fdims_.size() == gdims_.size());
    }

    std::size_t arity() const {
      return f_.arity() + g_.arity() - fdims_.size();
    }

    bool alias(const param_type& param) const {
      return f_.alias(param) || g_.alias(param);
    }

    void eval_to(param_type& result) const {
      join(join_op_, f_.param(), g_.param(),
           map_right(fdims_, gdims_, f_.arity(), g_.arity()), result);
    }

    template <typename AggOp>
    real_type accumulate(real_type init, AggOp agg_op) const {
      return join_accumulate(join_op_, agg_op, init, f_.param(), g_.param(),
                             map_right(fdims_, gdims_, f_.arity(), g_.arity()));
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_join<ResultSpace, compose_t<UnaryOp, JoinOp>, FDims, GDims, F, G>
    transform(UnaryOp unary_op) const {
      return { compose(unary_op, join_op_), fdims_, gdims_, f_, g_ };
    }

    template <typename AggOp, typename IndexRange>
    auto aggregate(AggOp agg_op, real_type init, IndexRange retain) const {
      return make_table_function<Space>(
        [=](const F& f, const G& g, param_type& result) {
          return join_aggregate(join_op_, agg_op, init, f.param(), g.param(),
                                map_right(fdims_, gdims_, f.arity(), g.arity()),
                                retain, result);
        }, retain.size(), f_, g_
      );
    }

  private:
    JoinOp join_op_;
    FDims fdims_;
    GDims gdims_;
    add_const_reference_if_factor_t<F> f_;
    add_const_reference_if_factor_t<G> g_;

  }; // class table_join

} } // namespace libgm::experimental

#endif
