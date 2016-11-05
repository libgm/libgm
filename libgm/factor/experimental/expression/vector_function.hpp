#ifndef LIBGM_VECTOR_FUNCTION_HPP
#define LIBGM_VECTOR_FUNCTION_HPP

#include <libgm/factor/experimental/expression/vector_base.hpp>

namespace libgm { namespace experimental {

  /**
   * A simple vector expression that is a function of one or more underlying
   * matrix and vector expressions.
   *
   * \tparam Space
   *         A tag denoting the space of the matrix, vector.g., prob_tag or log_tag.
   * \tparam EvalOp
   *         A function accepting all F and a vector reference.
   * \tparam F
   *         A non-empty pack of matrix and vector expressions with identical
   *         real_type.
   */
  template <typename Space, typename EvalOp, typename... F>
  class vector_function
    : public vector_base<
        Space,
        typename nth_type_t<0, F...>::real_type,
        vector_function<Space, EvalOp, F...> > {
  public:
    // Shortcuts
    using real_type  = typename nth_type_t<0, F...>::real_type;
    using param_type = real_vector<real_type>;

    vector_function(EvalOp eval_op, const F&... f)
      : eval_op_(eval_op), data_(f...) { }

    bool alias(const real_vector<real_type>& param) const {
      // vector_function can influence its consituent vectors
      // e.g., in the join-aggregate expressions
      return tuple_any(make_member_alias(param), data_);
    }

    bool alias(const real_matrix<real_type>& param) const {
      // vector_function can influence a matrix only through param(),
      // which returns a real_vector temporary and thus acts as a shield
      // against aliasing a matrix
      return false;
    }

    void eval_to(param_type& result) const {
      tuple_apply(eval_op_, tuple_cat(data_, std::tie(result)));
    }

  private:
    EvalOp eval_op_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;

  }; // class vector_function

  /**
   * Creates a vector function, automatically deducing its type.
   *
   * \tparam Space  prob_tag or log_tag (must be specified explicitly)
   * \tparam EvalOp a function object evaluating the expression
   * \tparam F      a parameter pack of all constituent expressions

   * \relates vector_function
   */
  template <typename Space, typename EvalOp, typename... F>
  inline vector_function<Space, EvalOp, F...>
  make_vector_function(EvalOp eval_op, const F&... f) {
    return { eval_op, f... };
  }

  /**
   * Creates a special type of vector function that loads data from
   * tabular factors.
   * \relates vector_function
   */
  template <typename Space, typename F>
  inline auto vector_from_table(const F& f) {
    if (f.arity() != 1) {
      throw std::invalid_argument("The factor is not unary");
    }
    return make_vector_function<Space>(
      [](const F& f, real_vector<typename F::real_type>& result) {
        auto&& param = f.param();
        result.resize(param.size(0));
        std::copy(param.begin(), param.end(), result.data());
      }, f
    );
  }

} } // namespace libgm::experimental

#endif
