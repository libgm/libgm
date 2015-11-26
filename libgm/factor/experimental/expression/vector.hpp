#ifndef LIBGM_VECTOR_EXPRESSIONS_HPP
#define LIBGM_VECTOR_EXPRESSIONS_HPP

#include <libgm/argument/binary_domain.hpp>
#include <libgm/argument/unary_domain.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/composition.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/traits/nth_type.hpp>
#include <libgm/traits/reference.hpp>

#include <tuple>
#include <type_traits>

namespace libgm { namespace experimental {

  // The base class
  //============================================================================

  template <typename Space, typename Arg, typename RealType, typename Derived>
  class vector_base;

  template <typename F>
  struct is_vector : std::is_same<param_t<F>, real_vector<real_t<F> > > { };

  // Transform expression
  //============================================================================

  /**
   * A class represents an element-wise transform of one or more vectors.
   * The vectors must have the same argument.
   *
   * Examples of a transform:
   * f * 2
   * max(f*2, g)
   *
   * \tparam Space
   *         A tag denoting the space of the vector, e.g., prob_tag or log_tag.
   * \tparam Op
   *         A function object that accepts sizeof...(Expr) dense vectors
   *         and returns a dense vector expression.
   * \tparam Expr
   *         A non-empty pack of (possibly const-reference qualified)
   *         probability_vector or logarithmic_vector expressions with
   *         identical argument_type and real_type.
   */
  template <typename Space, typename Op, typename... Expr>
  class vector_transform
    : public vector_base<
        Space,
        argument_t<nth_type_t<0, Expr...> >,
        real_t<nth_type_t<0, Expr...> >,
        vector_transform<Space, Op, Expr...> > {

  public:
    // shortcuts
    using argument_type = argument_t<nth_type_t<0, Expr...> >;
    using domain_type   = domain_t<nth_type_t<0, Expr...> >;
    using real_type     = real_t<nth_type_t<0, Expr...> >;
    using param_type    = real_vector<real_type>;

    using base = vector_base<Space, argument_type, real_type, vector_transform>;
    using base::param;

    static const std::size_t trans_arity = sizeof...(Expr);

    //! Constructs a vector_transform with the given operator and expressions.
    vector_transform(Op op, std::tuple<Expr...>&& data)
      : op_(op), data_(std::move(data)) { }

    const domain_type& arguments() const {
      return std::get<0>(data_).arguments();
    }

    param_type param() const {
      param_type tmp;
      eval_to(tmp);
      return tmp;
    }

    // Evaluation
    //--------------------------------------------------------------------------

    Op trans_op() const {
      return op_;
    }

    std::tuple<add_const_reference_t<Expr>...> trans_data() const& {
      return data_;
    }

    std::tuple<Expr...> trans_data() && {
      return std::move(data_);
    }

    bool alias(const param_type& param) const {
      return false; // vector_transform is always safe to evaluate
    }

    void eval_to(param_type& result) const {
      result = result_expr();
    }

    template <typename AssignOp>
    void transform_inplace(AssignOp assign_op, param_type& result) const {
      assign_op(result.array(), result_expr());
    }

    template <typename AccuOp>
    void accumulate(AccuOp accu_op) const {
      return accu_op(result_expr());
    }

  private:
    //! Returns the Eigen expression for the result of this transform.
    auto result_expr() const {
      return tuple_apply(
        op_,
        tuple_transform(member_array(), tuple_transform(member_param(), data_))
      );
    }

    //! The array operator applied to the Eigen vectors.
    Op op_;

    //! The transformed vector expressions.
    std::tuple<Expr...> data_;

  }; // class vector_transform

  /**
   * Constructs a vector_transform object, deducing its type.
   *
   * \relates vector_transform
   */
  template <typename Space, typename Op, typename... Expr>
  inline vector_transform<Space, Op, Expr...>
  make_vector_transform(Op op, std::tuple<Expr...>&& expr) {
    return { op, std::move(expr) };
  }

  /**
   * Transforms two vector with identical Space, Arg, and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates vector_transform
   */
  template <typename BinaryOp, typename Space, typename Arg, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op, F&& f, G&& g,
            vector_base<Space, Arg, RealType, std::decay_t<F> >* /* f_tag */,
            vector_base<Space, Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    constexpr std::size_t m = std::decay_t<F>::trans_arity;
    constexpr std::size_t n = std::decay_t<G>::trans_arity;
    return make_vector_transform<Space>(
      compose<m, n>(binary_op, f.trans_op(), g.trans_op()),
      std::tuple_cat(std::forward<F>(f).trans_data(),
                     std::forward<G>(g).trans_data())
    );
  }


  // Raw buffer map
  //============================================================================

  /**
   * An expression that represents a vector via a domain and a raw pointer to
   * the data.
   */
  template <typename Space, typename Arg, typename RealType>
  class vector_map
    : public vector_base<
        Space,
        Arg,
        RealType,
        vector_map<Space, Arg, RealType> > {

  public:
    using base = vector_base<Space, Arg, RealType, vector_map>;
    using base::param;

    vector_map(const unary_domain<Arg>& args, const RealType* data)
      : args_(args), data_(data) {
      assert(data);
    }

    const unary_domain<Arg>& arguments() const {
      return args_;
    }

    real_vector<RealType> param() const {
      return map();
    }

    bool alias(const real_vector<RealType>& param) const {
      return false; // vector_map is always safe to evaluate
    }

    void eval_to(real_vector<RealType>& result) const {
      result = map();
    }

    template <typename AssignOp>
    void transform_inplace(AssignOp assign_op,
                           real_vector<RealType>& result) const {
      assign_op(result.array(), map().array());
    }

    template <typename AssignOp>
    void join_inplace(AssignOp op,
                      const binary_domain<Arg>& result_args,
                      real_matrix<RealType>& result) const {
      if (args_.x() == result_args.x()) {
        op(result.array().colwise(), map().array());
      } else if (args_.x() == result_args.y()) {
        op(result.array().rowwise(), map().array().transpose());
      } else {
        std::ostringstream out;
        out << "vector_map: argument ";
        argument_traits<Arg>::print(out, args_.x());
        out << " not found";
        throw std::invalid_argument(out.str());
      }
    }

    template <typename AccuOp>
    RealType accumulate(AccuOp accu_op) const {
      return accu_op(map());
    }

  private:
    //! Returns the Eigen Map object for this expression.
    Eigen::Map<const real_vector<RealType> > map() const {
      return { data_, std::ptrdiff_t(args_.num_values()) };
    }

    //! The arguments of the expression.
    unary_domain<Arg> args_;

    //! The raw pointer to the data.
    const RealType* data_;

  }; // class vector_map

} } // namespace libgm::experimental

#endif
