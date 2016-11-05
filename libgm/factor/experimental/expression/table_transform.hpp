#ifndef LIBGM_TABLE_TRANSFORM_HPP
#define LIBGM_TABLE_TRANSFORM_HPP

#include <libgm/datastructure/table.hpp>
#include <libgm/factor/experimental/expression/table_base.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>

namespace libgm { namespace experimental {

  /**
   * A class that represents an element-wise transform of one or more tables.
   * The tables must have the same shapes.
   *
   * \tparam Space
   *         A tag denoting the space of the table, e.g., prob_tag or log_tag.
   * \tparam Op
   *         A function object that accepts sizeof...(F) arguments of type
   *         F::real_type and returns a scalar.
   * \tparam F
   *         A non-empty pack of table expressions with identical real_type.
   */
  template <typename Space, typename Op, typename... F>
  class table_transform
    : public table_base<
        Space,
        std::result_of_t<Op(typename F::real_type...)>,
        table_transform<Space, Op, F...> > {

  public:
    // shortcuts
    using real_type  = std::result_of_t<Op(typename F::real_type...)>;
    using param_type = table<real_type>;

    table_transform(Op op, const F&... f)
      : op_(op), data_(f...) { }

    table_transform(Op op, const std::tuple<const F&...>& data)
      : op_(op), data_(data) { }

    std::size_t arity() const {
      return std::get<0>(data_).arity();
    }

    bool alias(const param_type& param) const {
      return false; // table_transform is always safe to evaluate
    }

    void eval_to(param_type& result) const {
      table_transform_assign<real_type, Op> assign(result, op_);
      tuple_apply(assign, tuple_transform(member_param(), data_));
    }

    template <typename JoinOp>
    void transform_inplace(JoinOp op, param_type& result) const {
      table_transform_update<real_type, Op, JoinOp> updater(result, op_);
      tuple_apply(updater, tuple_transform(member_param(), data_));
    }

    template <typename AccuOp>
    real_type accumulate(real_type init, AccuOp accu_op) const {
      table_transform_accumulate<real_type, Op, AccuOp> acc(init, op_, accu_op);
      return tuple_apply(acc, tuple_transform(member_param(), data_));
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_transform<ResultSpace, compose_t<UnaryOp, Op>, F...>
    transform(UnaryOp unary_op) const {
      return { compose(unary_op, op_), data_ };
    }

    friend Op transform_op(const table_transform& f) {
      return f.op_;
    }

    friend std::tuple<const F&...> transform_data(const table_transform& f) {
      return f.data_;
    }

  private:
    Op op_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;

  }; // class table_transform

  /**
   * Constructs a table_transform object, deducing its type.
   * \relates table_transform
   */
  template <typename Space, typename Op, typename... F>
  inline table_transform<Space, Op, F...>
  make_table_transform(Op op, const std::tuple<const F&...>& data) {
    return { op, data };
  }

  /**
   * The default transform associated with a table expression.
   * \relates table_base
   */
  template <typename Space, typename RealType, typename Derived>
  inline identity
  transform_op(const table_base<Space, RealType, Derived>& f) {
    return identity();
  }

  /**
   * The default transform data assciated with a table expression.
   * \relates table_base
   */
  template <typename Space, typename RealType, typename Derived>
  inline std::tuple<const Derived&>
  transform_data(const table_base<Space, RealType, Derived>& f) {
    return std::tie(f.derived());
  }

  /**
   * Transforms two tables with identical Space and RealType.
   * \relates table_base
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op,
            const table_base<Space, RealType, F>& f,
            const table_base<Space, RealType, G>& g) {
    auto fdata = transform_data(f.derived());
    auto gdata = transform_data(g.derived());
    constexpr std::size_t m = std::tuple_size<decltype(fdata)>::value;
    constexpr std::size_t n = std::tuple_size<decltype(gdata)>::value;
    return make_table_transform<Space>(
      compose<m, n>(binary_op, transform_op(f.derived()), transform_op(g.derived())),
      std::tuple_cat(fdata, gdata)
    );
  }

} } // namespace libgm::experimental

#endif
