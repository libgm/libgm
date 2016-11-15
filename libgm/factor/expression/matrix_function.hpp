#ifndef LIBGM_MATRIX_FUNCTION_HPP
#define LIBGM_MATRIX_FUNCTION_HPP

#include <libgm/factor/expression/matrix_base.hpp>
#include <libgm/factor/expression/vector_base.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/traits/nth_type.hpp>

namespace libgm { namespace experimental {

  /**
   * A simple matrix expression that is a function of one or more underlying
   * matrix and vector expressions.
   *
   * \tparam Space
   *         A tag denoting the space of the matrix, e.g., prob_tag or log_tag.
   * \tparam EvalOp
   *         A function accepting all F and a matrix reference.
   * \tparam F
   *         A non-empty pack of matrix and vector expressions with identical
   *         real_type.
   */
  template <typename Space, typename EvalOp, typename... F>
  class matrix_function
    : public matrix_base<
        Space,
        typename nth_type_t<0, F...>::real_type,
        matrix_function<Space, EvalOp, F...> > {
  public:
    // Shortcuts
    using real_type  = typename nth_type_t<0, F...>::real_type;
    using param_type = dense_matrix<real_type>;

    matrix_function(EvalOp eval_op, const F&... f)
      : eval_op_(eval_op), data_(f...) { }

    bool alias(const dense_vector<real_type>& param) const {
      // matrix_function can influence a vector only through param(),
      // which returns a dense_matrix temporary and thus acts as a shield
      // against aliasing a vector
      return false;
    }

    bool alias(const dense_matrix<real_type>& param) const {
      // matrix_function can influence its consituent matrices
      // e.g., in the transpose operation
      return tuple_any(make_member_alias(param), data_);
    }

    void eval_to(param_type& result) const {
      tuple_apply(eval_op_, tuple_cat(data_, std::tie(result)));
    }

  private:
    EvalOp eval_op_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;

  }; // class matrix_function

  /**
   * Creates a matrix function, automatically deducing its type.
   *
   * \tparam Space  prob_tag or log_tag (must be specified explicitly)
   * \tparam EvalOp a function object evaluating the expression
   * \tparam F      a parameter pack of all constituent expressions

   * \relates matrix_function
   */
  template <typename Space, typename EvalOp, typename... F>
  inline matrix_function<Space, EvalOp, F...>
  make_matrix_function(EvalOp eval_op, const F&... f) {
    return { eval_op, f... };
  }

  /**
   * Creates a special type of matrix function that computes the outer
   * join of two vectors.
   *
   * \relates matrix_function
   */
  template <typename JoinOp,
            typename Space, typename RealType, typename Derived, typename Other>
  inline auto
  outer_join(JoinOp join_op,
             const vector_base<Space, RealType, Derived>& f,
             const vector_base<Space, RealType, Other>& g) {
    return make_matrix_function<Space>(
      [join_op](const Derived& f, const Other& g, dense_matrix<RealType>& result) {
        auto&& u = f.param();
        auto&& v = g.param();
        result = join_op(u.array().rowwise().replicate(v.size()),
                         v.array().rowwise().replicate(u.size()).transpose());
      }, f.derived(), g.derived()
    );
  }

  /**
   * Creates a special type of matrix function that loads data from
   * tabular factors.
   *
   * \relates matrix_function
   */
  template <typename Space, typename F>
  inline auto matrix_from_table(const F& f) {
    if (f.arity() != 2) {
      throw std::invalid_argument("The factor is not binary");
    }
    return make_matrix_function<Space>(
      [](const F& f, dense_matrix<typename F::real_type>& result) {
        auto&& param = f.param();
        result.resize(param.size(0), param.size(1));
        std::copy(param.begin(), param.end(), result.data());
      }, f
    );
  }

} } // namespace libgm::experimental

#endif
