#ifndef LIBGM_TABLE_FUNCTION_HPP
#define LIBGM_TABLE_FUNCTION_HPP

#include <libgm/datastructure/table.hpp>
#include <libgm/factor/experimental/expression/table_base.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/traits/nth_type.hpp>

#include <array>

namespace libgm { namespace experimental {

  /**
   * A simple table expression that is a function of one or more underlying
   * table expressions.
   *
   * \tparam Space
   *         A tag denoting the space of the table, e.g., prob_tag or log_tag.
   * \tparam Alias
   *         Determine whether to perform alias checking.
   * \tparam EvalOp
   *         A function accepting all F and a table reference.
   * \tparam F
   *         A non-empty pack of table expressions with identical real_type.
   */
  template <typename Space, bool Alias, typename EvalOp, typename... F>
  class table_function
    : public table_base<
        Space,
        typename nth_type_t<0, F...>::real_type,
        table_function<Space, Alias, EvalOp, F...> > {
  public:
    // Shortcuts
    using real_type  = typename nth_type_t<0, F...>::real_type;
    using param_type = table<real_type>;

    table_function(EvalOp eval_op, std::size_t arity, const F&... f)
      : eval_op_(eval_op), arity_(arity), data_(f...) { }

    std::size_t arity() const {
      return arity_;
    }

    //! Implementation of alias() that checks for aliasing.
    template <bool B = Alias>
    std::enable_if_t<B, bool> alias(const param_type& param) const {
      return tuple_any(make_member_alias(param), data_);
    }

    //! Implementation of alias() that does not check for aliasing.
    template <bool B = Alias>
    std::enable_if_t<!B, bool> alias(const param_type& param) const {
      return false;
    }

    void eval_to(param_type& result) const {
      tuple_apply(eval_op_, tuple_cat(data_, std::tie(result)));
    }

  private:
    EvalOp eval_op_;
    std::size_t arity_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;
  }; // class table_function

  /**
   * Creates a table function that checks for aliasing, automatically deducing
   * its type.
   *
   * \tparam Space  prob_tag or log_tag (must be specified explicitly)
   * \tparam EvalOp a function object evaluating the expression
   * \tparam F      a parameter pack of all constituent expressions
   *
   * \relates table_function
   */
  template <typename Space, typename EvalOp, typename... F>
  inline table_function<Space, true, EvalOp, F...>
  make_table_function(EvalOp eval_op, std::size_t arity, const F&... f) {
    return { eval_op, arity, f... };
  }

  /**
   * Creates a table function that does not check for aliasing, automatically
   * deducing its type.
   */
  template <typename Space, typename EvalOp, typename... F>
  inline table_function<Space, false, EvalOp, F...>
  make_table_function_noalias(EvalOp eval_op, std::size_t arity, const F&... f) {
    return { eval_op, arity, f... };
  }

  /**
   * Returns a special table function that extracts data from a vector.
   *
   * \relates table_function
   */
  template <typename Space, typename F>
  inline auto table_from_vector(const F& f) {
    return make_table_function_noalias<Space>(
      [](const F& f, table<typename F::real_type>& result) {
        auto&& param = f.param();
        std::array<std::size_t, 1> dims = { param.size() };
        result.reset(dims.begin(), dims.end());
        for (std::size_t i = 0; i < param.size(); ++i) {
          result[i] = param[i];
        }
      }, 1, f
    );
  }

  /**
   * Returns a special table function that extracts data from a matrix.
   *
   * \relates table_function
   */
  template <typename Space, typename F>
  inline auto table_from_matrix(const F& f) {
    return make_table_function_noalias<Space>(
      [](const F& f, table<typename F::real_type>& result) {
        auto&& param = f.param();
        std::array<std::size_t, 2> dims = { param.rows(), param.cols() };
        result.reset(dims.begin(), dims.end());
        typename F::real_type* dest = result.data();
        for (std::size_t j = 0; j < param.cols(); ++j) {
          for (std::size_t i = 0; i < param.rows(); ++i) {
            *dest++ = param(i, j);
          }
        }
      }, 2, f
    );
  }

  /**
   * Returns a special table_function object representing the outer join
   * of two table expressions with identical Space.
   */
  template <typename JoinOp,
            typename Space, typename RealType, typename Derived, typename Other>
  inline auto
  outer_join(JoinOp join_op,
             const table_base<Space, RealType, Derived>& f,
             const table_base<Space, RealType, Other>& g) {
    return make_table_function<Space>(
      [join_op](const Derived& f, const Other& g, table<RealType>& result) {
        outer_join(join_op, f.param(), g.param(), result);
      },
      f.derived().arity() + g.derived().arity(),
      f.derived(), g.derived()
    );
  }

} } // namespace libgm::experimental

#endif
