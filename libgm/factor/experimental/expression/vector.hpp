#ifndef LIBGM_VECTOR_EXPRESSIONS_HPP
#define LIBGM_VECTOR_EXPRESSIONS_HPP

#include <libgm/datastructure/temporary.hpp>
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

  template <typename Space, typename RealType, typename Derived>
  class vector_base;

  // Forward declaration
  template <typename Space, typename RealType, typename Derived>
  class matrix_base;

  template <typename F>
  struct is_vector
    : std::is_same<param_t<F>, real_vector<real_t<F> > > { };

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
        real_t<nth_type_t<0, Expr...> >,
        vector_transform<Space, Op, Expr...> > {

  public:
    // shortcuts
    using real_type  = real_t<nth_type_t<0, Expr...> >;
    using param_type = real_vector<real_type>;

    static const std::size_t trans_arity = sizeof...(Expr);

    //! Constructs a vector_transform with the given operator and expressions.
    vector_transform(Op op, std::tuple<Expr...>&& data)
      : op_(op), data_(std::move(data)) { }

    std::size_t size() const {
      return std::get<0>(data_).size();
    }

    Op trans_op() const {
      return op_;
    }

    std::tuple<cref_t<Expr>...> trans_data() const& {
      return data_;
    }

    std::tuple<Expr...> trans_data() && {
      return std::move(data_);
    }

    bool alias(const real_vector<real_type>& param) const {
      return false;
      // vector_transform is always safe to evaluate, because it is performed
      // element-wise on the input arrays, and input arrays are trivial
    }

    bool alias(const real_matrix<real_type>& param) const {
      return false;
      // vector_transform evaluates to a vector temporary, so it cannot alias
      // a matrix
    }

    void eval_to(param_type& result) const {
      auto vectors = tuple_transform(member_param(), data_);
      result = expr(vectors);
    }

    template <typename AssignOp>
    void transform_inplace(AssignOp assign_op, param_type& result) const {
      auto vectors = tuple_transform(member_param(), data_);
      assign_op(result.array(), expr(vectors));
    }

  private:
    /**
     * Returns the Eigen expression representing the result of this transform.
     * Because the results of some of the subexpressions may be plain types,
     * these need to be passed explicitly by reference.
     */
    template <typename... Vectors>
    auto expr(std::tuple<Vectors...>& vectors) const {
      return tuple_apply(op_, tuple_transform(member_array(), vectors));
    }

    Op op_;
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
   * Transforms two vector with identical Space and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates vector_transform
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op, F&& f, G&& g,
            vector_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
            vector_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    constexpr std::size_t m = std::decay_t<F>::trans_arity;
    constexpr std::size_t n = std::decay_t<G>::trans_arity;
    return make_vector_transform<Space>(
      compose<m, n>(binary_op, f.trans_op(), g.trans_op()),
      std::tuple_cat(std::forward<F>(f).trans_data(),
                     std::forward<G>(g).trans_data())
    );
  }


  // Join expressions
  //============================================================================

  /**
   * A class that represents an outer join of two vectors F and G.
   */
  template <typename Space, typename JoinOp, typename F, typename G>
  class vector_outer_join
    : public matrix_base<
        Space,
        real_t<F>,
        vector_outer_join<Space, JoinOp, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_vector<F>::value && is_vector<G>::value,
                  "The joined expressions must be both vector expressions");

  public:
    using real_type  = real_t<F>;
    using param_type = real_matrix<real_type>;

    vector_outer_join(JoinOp join_op, F&& f, G&& g)
      : join_op_(join_op),
        f_(std::forward<F>(f)),
        g_(std::forward<G>(g)) { }

    std::size_t rows() const {
      return f_.size();
    }

    std::size_t cols() const {
      return g_.size();
    }

    bool alias(const real_vector<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // vector_outer_join can alias a vector e.g. if f_ or g_ own that vector
    }

    bool alias(const real_matrix<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // in some edge cases, this join could be aliasing a matrix,
      // e.g., when taking an outer product of two matrix restrictions,
      // assigned back to one of those matrices
    }

    auto array() const {
      return join_op_(
        u_.capture(f_.param()).array().rowwise().replicate(g_.size()),
        v_.capture(g_.param()).array().rowwise().replicate(f_.size()).transpose()
      );
    }

  private:
    JoinOp join_op_;
    F f_;
    G g_;

    mutable temporary<real_vector<real_type>, has_param_temporary<F>::value> u_;
    mutable temporary<real_vector<real_type>, has_param_temporary<G>::value> v_;
  }; // class vector_outer_join

  /**
   * Returns an outer join of two vectors with identical Space and RealType.
   * \relates vector_outer_join
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline vector_outer_join<Space, BinaryOp, remove_rref_t<F>, remove_rref_t<G> >
  outer(BinaryOp binary_op, F&& f, G&& g,
        vector_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
        vector_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }


  // Raw buffer map
  //============================================================================

  /**
   * An expression that represents a vector via a elngth and a raw pointer to
   * the data.
   */
  template <typename Space, typename RealType>
  class vector_map
    : public vector_base<Space, RealType, vector_map<Space, RealType> > {

  public:
    using param_type = real_vector<RealType>;

    vector_map(std::size_t length, const RealType* data)
      : length_(length), data_(data) { }

    std::size_t size() const {
      return length_;
    }

    bool alias(const real_vector<RealType>& param) const {
      return false; // vector_map assumes no aliasing
    }

    bool alias(const real_matrix<RealType>& param) const {
      return false; // vector_map assumes no aliasing
    }

    void eval_to(param_type& result) const {
      result = param();
    }

    Eigen::Map<const param_type> param() const {
      return { data_, std::ptrdiff_t(length_) };
    }

    template <typename UnaryPredicate>
    std::size_t find_if(UnaryPredicate pred) const {
      auto it = std::find(data_, data_ + length_, pred);
      if (it == data_ + length_) {
        throw std::out_of_range("Element could not be found");
      } else {
        return it - data_;
      }
    }

  private:
    std::size_t length_;
    const RealType* data_;

  }; // class vector_map

  template <typename Space, typename RealType>
  struct is_primitive<vector_map<Space, RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
