#ifndef LIBGM_TABLE_RESTRICT_HPP
#define LIBGM_TABLE_RESTRICT_HPP

#include <libgm/enable_if.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factor/experimental/expression/table_base.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/datastructure/uint_vector.hpp>

namespace libgm { namespace experimental {

  /**
   * A class that represents an assignment of a table to a subset of
   * dimensions, followed by an optional transform.
   */
  template <typename Space, typename IndexRange, typename TransOp, typename F>
  class table_restrict
    : public table_base<
        Space,
        typename F::real_type,
        table_restrict<Space, IndexRange, TransOp, F> > {

    static_assert(std::is_trivially_copyable<IndexRange>::value,
                  "The restricted dimensions must be trivially copyable.");

    using base = table_base<Space, typename F::real_type, table_restrict>;

  public:
    // Shortcuts
    using real_type  = typename F::real_type;
    using param_type = table<real_type>;

    table_restrict(IndexRange dims, const uint_vector& values, TransOp trans_op,
                   const F& f)
      : dims_(dims), values_(values), trans_op_(trans_op), f_(f) {
      assert(dims.size() == values.size());
    }

    std::size_t arity() const {
      return f_.arity() - dims_.size();
    }

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.param().restrict(dims_, values_, result);
      result.transform(trans_op_);
    }

    template <typename JoinOp>
    void transform_inplace(JoinOp op, param_type& result) const {
      f_.param().restrict_update(compose_right(op, trans_op_),
                                 dims_, values_, result);
    }

    template <typename JoinOp, typename It>
    void join_inplace(JoinOp join_op, index_range<It> join_dims,
                      param_type& result) const {
      f_.param().restrict_join(compose_right(join_op, trans_op_), join_dims,
                               dims_, values_, result);
    }

    // Allow composing transforms if doing so does not change the real_type
    // of the expression
    LIBGM_ENABLE_IF_D(
      (std::is_same<std::result_of_t<UnaryOp(real_type)>, real_type>::value),
      typename ResultSpace = Space, typename UnaryOp = void)
    table_restrict<ResultSpace, compose_t<UnaryOp, TransOp>, IndexRange, F>
    transform(UnaryOp unary_op) const {
      return { f_, dims_, values_, compose(unary_op, trans_op_) };
    }

    using base::transform; // in case the one above is disabled

  private:
    IndexRange dims_;
    const uint_vector& values_;
    TransOp trans_op_;
    add_const_reference_if_factor_t<F> f_;

  }; // class table_restrict

  /**
   * Constructs a table_restrict object, automatically inferring its type.
   * \relates table_restrict
   */
  template <typename Space, typename IndexRange, typename F>
  table_restrict<Space, IndexRange, identity, F>
  make_table_restrict(IndexRange dims, const uint_vector& vals, const F& f) {
    return { dims, vals, identity(), f };
  }


} } // namespace libgm::experimental

#endif

