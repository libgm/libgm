#ifndef LIBGM_EXPERIMENTAL_CANONICAL_GAUSSIAN_HPP
#define LIBGM_EXPERIMENTAL_CANONICAL_GAUSSIAN_HPP

#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_base.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_function.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_selector.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_transform.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_function.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/param/canonical_gaussian_param.hpp>
#include <libgm/range/index_range.hpp>

namespace libgm { namespace experimental {

  // Forward declaration of the factor
  template <typename RealType> class canonical_gaussian;
  template <typename RealType> class moment_gaussian;

  // Base class
  //============================================================================

  /**
   * The base class for canonical_gaussian factors and expressions.
   *
   * \tparam RealType
   *         The real type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         The type must implement the following functions:
   *         arity(), alias(), eval_to().
   */
  template <typename RealType, typename Derived>
  class canonical_gaussian_base {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef RealType                     real_type;
    typedef logarithmic<RealType>        result_type;
    typedef canonical_gaussian<RealType> factor_type;

    // ParametricFactor member types
    typedef canonical_gaussian_param<RealType> param_type;
    typedef real_vector<RealType>              vector_type;
    typedef uint_vector                        index_type;

    // ExponentialFamilyFactor member types
    typedef moment_gaussian<RealType> probability_factor_type;

    // Constructors and casts
    //--------------------------------------------------------------------------

    //! Default constructor.
    canonical_gaussian_base() { }

    //! Downcasts this object to the derived type.
    Derived& derived() {
      return static_cast<Derived&>(*this);
    }

    // Downcasts this object to the derived type.
    const Derived& derived() const {
      return static_cast<const Derived&>(*this);
    }

    //! Returns this as void-pointer.
    const void* void_ptr() const {
      return this;
    }

    // Comparison and output operators
    //--------------------------------------------------------------------------

    /**
     * Returns true of the two expressions have the same parameters.
     */
    template <typename Other>
    friend bool
    operator==(const canonical_gaussian_base& f,
               const canonical_gaussian_base<RealType, Other>& g) {
      return f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if the two expressions do not have the same parameters.
     */
    template <typename Other>
    friend bool
    operator!=(const canonical_gaussian_base& f,
               const canonical_gaussian_base<RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const canonical_gaussian_base& f) {
      out << f.derived().param() << std::endl;
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a canonical_gaussian expression representing an element-wise
     * transform of a canonical_gaussian expression with two unary operations
     * applied to the information matrix/vector and the log-multiplier.
     */
    template <typename VectorOp, typename ScalarOp>
    auto transform(VectorOp vector_op, ScalarOp scalar_op) const {
      return make_canonical_gaussian_transform(
        vector_op, scalar_op, std::tie(derived()));
    }

    /**
     * Returns a canonical_gaussian expression representing the product of
     * a canonical_gaussian expression and a scalar.
     */
    friend auto
    operator*(const canonical_gaussian_base& f, logarithmic<RealType> x) {
      return f.derived().transform(identity(), incremented_by<RealType>(x.lv));
    }

    /**
     * Returns a canonical_gaussian expression representing the product of
     * a scalar and a canonical_gaussian expression.
     */
    friend auto
    operator*(logarithmic<RealType> x, const canonical_gaussian_base& f) {
      return f.derived().transform(identity(), incremented_by<RealType>(x.lv));
    }

    /**
     * Returns a canonical_gaussian expression representing the division of
     * a canonical_gaussian expression and a scalar.
     */
    friend auto
    operator/(const canonical_gaussian_base& f, logarithmic<RealType> x) {
      return f.derived().transform(identity(), decremented_by<RealType>(x.lv));
    }

    /**
     * Returns a canonical_gaussian expression representing the division of
     * a scalar and a canonical_gaussian expression.
     */
    friend auto
    operator/(logarithmic<RealType> x, const canonical_gaussian_base& f) {
      return f.derived().transform(std::negate<>(),
                                   subtracted_from<RealType>(x.lv));
    }

    /**
     * Returns a canonical_gaussian expression representing the
     * canonical_gaussian expression raised to an exponent.
     */
    friend auto pow(const canonical_gaussian_base& f, RealType x) {
      return f.derived().transform(multiplied_by<RealType>(x),
                                   multiplied_by<RealType>(x));
    }

    /**
     * Returns a canonical_gaussian expression representing the direct product
     * of two canonical_gaussian expressions.
     */
    template <typename Other>
    friend auto operator*(const canonical_gaussian_base& f,
                          const canonical_gaussian_base<RealType, Other>& g) {
      return libgm::experimental::transform(std::plus<>(), f, g);
    }

    /**
     * Returns a canonical_gaussian expression repreenting the direct division
     * of two canonical_gaussian expressions.
     */
    template <typename Other>
    friend auto operator/(const canonical_gaussian_base& f,
                          const canonical_gaussian_base<RealType, Other>& g) {
      return libgm::experimental::transform(std::minus<>(), f, g);
    }

    /**
     * Returns a canonical_gaussian expression representing f$f^{(1-a)} * g^a\f$
     * for two canonical_gaussian expressions f and g.
     */
    template <typename Other>
    friend auto
    weighted_update(const canonical_gaussian_base& f,
                    const canonical_gaussian_base<RealType, Other>& g,
                    RealType x) {
      return libgm::experimental::transform(weighted_plus<RealType>(1-x, x), f, g);
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a canonical_gaussian expression with the paramters of this
     * expression cast to a different RealType.
     */
    template <typename NewRealType>
    auto cast() const {
      return derived().transform(member_cast<NewRealType>(),
                                 scalar_cast<NewRealType>());
    }

    /**
     * Returns a moment_gaussian expression equivalent to this expression.
     */
    auto moment() const {
      return make_moment_gaussian_function_noalias<void>(
        [](const Derived& f, moment_gaussian_param<RealType>& result) {
          result = f.param();
        }, derived().arity(), 0, derived());
    }

    // Aggregates
    //--------------------------------------------------------------------------

    /**
     * Returns a canonical_gaussian expression representing an aggregate
     * of this expression over a range of dimensions.
     *
     * \throw numerical_error
     *        if the information matrix over the eliminated dimensions
     *        is singular.
     */
    template <typename IndexRange>
    auto aggregate(bool marginal, IndexRange retain) const {
      static_assert(std::is_trivially_copyable<IndexRange>::value,
                    "The retained dimensions must be trivially copyable.");
      using workspace_type = typename param_type::collapse_workspace;
      return make_canonical_gaussian_function<workspace_type>(
        [marginal, retain](const Derived& f, workspace_type& ws,
                           param_type& result) {
          f.param().collapse(marginal, retain, complement(retain, f.arity()),
                             ws, result);
        }, retain.size(), derived());
    }

    /**
     * Returns a canonical_gaussian expression representing the marginal
     * of this expression over a span of dimensions.
     *
     * \throw numerical_error
     *        if the information matrix over the marginalized dimensions
     *        is singular.
     */
    auto marginal(std::size_t start, std::size_t n = 1) const {
      return derived().aggregate(true /* marginal */, span(start, n));
    }

    /**
     * Returns a canonical_gaussian expression representing the maximum
     * of this expression over a span of dimensions.
     *
     * \throw numerical_error
     *        if the information matrix over the maximized dimensions
     *        is singular.
     */
    auto maximum(std::size_t start, std::size_t n = 1) const {
      return derived().aggregate(false /* maximum */ , span(start, n));
    }

    /**
     * Returns a canonical_gaussian expression representing the marginal
     * of this expression over a subset of dimensions.
     *
     * \throw numerical_error
     *        if the information matrix over the marginalized dimensions
     *        is singular.
     */
    auto marginal(const uint_vector& retain) const {
      return derived().aggregate(true /* marginal */, iref(retain));
    }

    /**
     * Returns a canonical_gaussian expression representing the maximum
     * of this expression over a subset of dimensions.
     *
     * \throw numerical_error
     *        if the information matrix over the maximized dimensions
     *        is singular.
     */
    auto maximum(const uint_vector& retain) const {
      return derived().aggregate(false /* maximum */, iref(retain));
    }

    /**
     * Computes the normalization constant of this expression.
     */
    logarithmic<RealType> sum() const {
      return { derived().param().marginal(), log_tag() };
    }

    /**
     * Computes the maximum value of this expression.
     */
    logarithmic<RealType> max() const {
      return { derived().param().maximum(), log_tag() };
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding assignment to a vector.
     */
    logarithmic<RealType> max(real_vector<RealType>& vec) const {
      return { derived().param().maximum(vec), log_tag() };
    }

    /**
     * Returns true if the factor is normalizable.
     */
    bool normalizable() const {
      return std::isfinite(derived().param().marginal());
    }

    // Conditioning
    //--------------------------------------------------------------------------

    /**
     * If this expression represents a marginal distribution, this function
     * returns a moment_gaussian expression representing the conditional
     * distribution with n head (front) dimensions.
     *
     * \throw numerical_error
     *        if the information matrix over the tail dimensions is singular.
     */
    auto conditional(std::size_t nhead) const {
      using workspace_type = typename param_type::collapse_workspace;
      return make_canonical_gaussian_function<workspace_type>(
        [nhead](const Derived& f, workspace_type& ws, param_type& result) {
          f.param().conditional(nhead, ws, result);
        }, derived().arity(), derived());
    }

    /**
     * A generic implementation of the restrict operation.
     */
    template <typename IndexRange>
    auto restrict_dims(IndexRange dims,
                       const real_vector<RealType>& values) const {
      return make_canonical_gaussian_function<void>(
        [dims, &values](const Derived& f, param_type& result) {
          f.param().restrict(complement(dims, f.arity()), dims, values, result);
        }, derived().arity() - dims.size(), derived());
    }

    /**
     * Returns a canonical_gaussian epxression representing the values over the
     * tail dimensions of this expression when the head dimesions are fixed
     * to the specified vector.
     */
    auto restrict_head(const real_vector<RealType>& values) const {
      return restrict_dims(front(values.size()), values);
    }

    /**
     * Returns a canonical_gaussian expression representing the values over the
     * head dimesions of this expression when the tail dimensions are fixed
     * to the specified vector.
     */
    auto restrict_tail(const real_vector<RealType>& values) const {
      return restrict_dims(back(derived().arity(), values.size()), values);
    }

    /**
     * Returns a canonical_gaussian expression resulting from restricting the
     * specified span of dimensions of this expression to the specified values.
     */
    auto restrict(std::size_t start, std::size_t n,
                  const real_vector<RealType>& values) const {
      return restrict_dims(span(start, n), values);
    }

    /**
     * Returns a canonical_gaussian expression resulting from restricting the
     * specified dimensions of this expresssion to the specified values.
     */
    auto restrict(const uint_vector& dims,
                  const real_vector<RealType>& values) const {
      return restrict_dims(iref(dims), values);
    }

    // Ordering
    //--------------------------------------------------------------------------

    /**
     * Returns a canonical_gaussian expression with the dimensions reordered
     * according to the given index.
     */
    auto reorder(const uint_vector& dims) const {
      assert(dims.size() == derived().arity());
      return make_canonical_gaussian_function<void>(
        [&dims](const Derived& f, param_type& result) {
          f.param().reorder(iref(dims), result);
        }, dims.size(), derived());
    }

    // Selectors
    //--------------------------------------------------------------------------

    /**
     * Returns a canonical_gaussian selector referencing the head dimensions
     * of this expression.
     */
    canonical_gaussian_selector<front, const Derived>
    head(std::size_t n) const {
      return { front(n), derived() };
    }

    /**
     * Returns a canonical_gaussian selector referencing the tail dimensions
     * of this expression.
     */
    canonical_gaussian_selector<back, const Derived>
    tail(std::size_t n) const {
      return { back(derived().arity(), n), derived() };
    }

    /**
     * Returns a canonical_gaussian selector referencing a single dimension
     * of this expression.
     */
    canonical_gaussian_selector<single, const Derived>
    dim(std::size_t index) const {
      return { single(index), derived() };
    }

    /**
     * Returns a canonical_gaussian selector referencing a span of dimensions
     * of this expression.
     */
    canonical_gaussian_selector<span, const Derived>
    dims(std::size_t start, std::size_t n) const {
      return { span(start, n), derived() };
    }

    /**
     * Returns a canonical_gaussian selector referencing a subset of dimensions
     * of this expression.
     */
    canonical_gaussian_selector<iref, const Derived>
    dims(const uint_vector& indices) const {
      return { iref(indices), derived() };
    }

    // Entropy and divergences
    //--------------------------------------------------------------------------

    /**
     * Computes the entropy for the distribution represented by this expression.
     */
    RealType entropy() const {
      return derived().param().entropy();
    }

    /**
     * Computes the entropy for a contiguous range of dimensions of the
     * distribution represented by this expression.
     */
    RealType entropy(std::size_t start, std::size_t n = 1) const {
      return derived().marginal(start, n).entropy();
    }

    /**
     * Computes the entropy for a span of dimensions of the distribution
     * represented by this expression.
     */
    RealType entropy(span s) const {
      return derived().marginal(s.start(), s.size()).entropy();
    }

    /**
     * Computes the entropy for a subset of dimensions of the distribution
     * represented by this expression.
     */
    RealType entropy(const uint_vector& dims) const {
      return derived().marginal(dims).entropy();
    }

    /**
     * Computes the mutual information between two contiguous ranges
     * of dimensions of the distribution represented by this expression.
     */
    RealType mutual_information(std::size_t starta, std::size_t startb,
                                std::size_t na = 1, std::size_t nb = 1) const {
      span a(starta, na), b(startb, nb);
      if (contiguous_union(a, b)) {
        return entropy(a) + entropy(b) - entropy(a | b);
      } else {
        return entropy(a) + entropy(b) - entropy(a + b);
      }
    }

    /**
     * Computes the mutual information between two subsets of dimensions
     * of the distribution represented by this expression.
     */
    RealType mutual_information(const uint_vector& a,
                                const uint_vector& b) const {
      return entropy(a) + entropy(b) - entropy(set_union(a, b));
    }

    /**
     * Computes the Kullback-Liebler divergence from p to q.
     * The two distributions must have the same arity.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const canonical_gaussian_base<RealType, Derived>& p,
                  const canonical_gaussian_base<RealType, Other>& q) {
      return kl_divergence(p.derived().param(), q.derived().param());
    }

    /**
     * Computes the maximum of absolute differences between parameters of
     * two moment_gaussians.
     */
    template <typename Other>
    friend RealType
    max_diff(const canonical_gaussian_base<RealType, Derived>& f,
             const canonical_gaussian_base<RealType, Other>& g) {
      return max_diff(f.derived().param(), g.derived().param());
    }

    // Expression evaluation
    //--------------------------------------------------------------------------

    //! Evaluates the parameters to a temporary (may be overriden).
    param_type param() const {
      param_type tmp; derived().eval_to(tmp); return tmp;
    }

    /**
     * Returns the canonical_gaussian factor resulting from evaluating this
     * expression.
     */
    canonical_gaussian<RealType> eval() const {
      return *this;
    }

    /**
     * Updates the parameters of the result using the parameters of this
     * expression.
     */
    template <typename UpdateOp>
    void transform_inplace(UpdateOp update_op, param_type& result) const {
      auto&& param = derived().param();
      result.update(update_op, param.eta, param.lambda, param.lm);
    }

    /**
     * Joins the parameters of this expression into the result, assuming
     * the mapping of arguments to dimensions given by the start map.
     * This function may be overriden by an expression to provide an optimized
     * implementation.
     */
    template <typename UpdateOp, typename It>
    void join_inplace(UpdateOp update_op, index_range<It> join_dims,
                      param_type& result) const {
      auto&& param = derived().param();
      result.update(update_op, join_dims, param.eta, param.lambda, param.lm);
    }

  }; // class canonical_gaussian_base


  // Factor
  //============================================================================

  /**
   * A factor of a multivariate normal (Gaussian) distribution in the natural
   * parameterization of the exponential family. Given an information vector
   * \eta and information matrix \lambda, this factor represents an
   * exponentiated quadratic function exp(-0.5 * x^T \lambda x + x^T \eta + a).
   *
   * \tparam RealType
   *         The real type reprsenting the parameters.
   * \ingroup factor_types
   * \see Factor
   */
  template <typename RealType = double>
  class canonical_gaussian
    : public canonical_gaussian_base<RealType, canonical_gaussian<RealType> > {

    using base = canonical_gaussian_base<RealType, canonical_gaussian>;

  public:
    //! Parameter struct (same as canonical_gaussian_base::param_type).
    typedef canonical_gaussian_param<RealType> param_type;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------

    //! Default constructor. Creates an empty factor.
    canonical_gaussian() { }

    /**
     * Constructs a factor with given arity, with uninitialized elements of the
     * information matrix and vector and zero log-multiplier.
     */
    explicit canonical_gaussian(std::size_t arity)
      : param_(arity) { }

    //! Constructs a factor equivalent to a constant.
    explicit canonical_gaussian(logarithmic<RealType> value)
      : param_(0, value.lv) { }

    //! Constructs a factor with given arity and constant value.
    canonical_gaussian(std::size_t arity, logarithmic<RealType> value)
      : param_(arity, value.lv) { }

    //! Constructs a factor with the given parameters.
    canonical_gaussian(const param_type& param)
      : param_(param) { }

    //! Constructs a factor with the given parameters.
    canonical_gaussian(param_type&& param)
      : param_(std::move(param)) { }

    //! Constructs a factor with the given informatino vector and matrix.
    canonical_gaussian(const real_vector<RealType>& eta,
                       const real_matrix<RealType>& lambda,
                       RealType lv = RealType(0))
      : param_(eta, lambda, lv) { }

    //! Constructs a factor from an expression.
    template <typename Derived>
    canonical_gaussian(const canonical_gaussian_base<RealType, Derived>& f) {
      f.derived().eval_to(param_);
    }

    //! Assigns a constant to this factor.
    canonical_gaussian& operator=(logarithmic<RealType> value) {
      reset(0);
      param_.lm = value.lv;
      return *this;
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    canonical_gaussian&
    operator=(const canonical_gaussian_base<RealType, Derived>& f) {
      assert(!f.derived().alias(param_));
      f.derived().eval_to(param_);
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(canonical_gaussian& f, canonical_gaussian& g) {
      swap(f.param_, g.param_);
    }

    //! Serializes the factor to an archive.
    void save(oarchive& ar) const {
      ar << param_;
    }

    //! Deserializes the factor from an archive.
    void load(iarchive& ar) {
      ar >> param_;
    }

    /**
     * Resets the content of this factor to the given number of dimensions.
     * If the dimensionality of the factor changes, the parameters become
     * invalidated.
     */
    void reset(std::size_t arity) {
      param_.resize(arity);
    }

#if 0
    /**
     * Returns the dimensionality of the parameters for the given
     * argument set.
     */
    static std::size_t param_shape(const domain<Arg>& dom) {
      return dom.num_dimensions();
    }
#endif

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the number of dimensions of this expressions.
    std::size_t arity() const {
      return param_.size();
    }

    //! Returns true if the expression has no arguments (same as arity() = 0).
    bool empty() const {
      return arity() == 0;
    }

    //! Returns the parameter struct.
    param_type& param() {
      return param_;
    }

    //! Returns the parameter struct.
    const param_type& param() const {
      return param_;
    }

    //! Returns the log multiplier.
    RealType log_multiplier() const {
      return param_.lm;
    }

    //! Returns the information vector.
    const real_vector<RealType>& inf_vector() const {
      return param_.eta;
    }

    //! Returns the information matrix.
    const real_matrix<RealType>& inf_matrix() const {
      return param_.lambda;
    }

    //! Returns the information subvector for consecutive dimensions.
    Eigen::VectorBlock<const real_vector<RealType> >
    inf_vector(std::size_t start, std::size_t n = 1) const {
      return param_.eta.segment(start, n);
    }

    //! Returns the information submatrix for consecutive dimensions.
    Eigen::Block<const real_matrix<RealType> >
    inf_matrix(std::size_t start, std::size_t n = 1) const {
      return param_.lambda.block(start, start, n, n);
    }

    //! Returns the information vector for a subset of the dimensions.
    real_vector<RealType> inf_vector(const uint_vector& dims) const {
      return subvec(param_.eta, iref(dims));
    }

    //! Returns the information matrix for a subset of the dimensions.
    real_matrix<RealType> inf_matrix(const uint_vector& dims) const {
      return submat(param_.eta, iref(dims), iref(dims));
    }

    //! Evaluates the factor for a vector.
    logarithmic<RealType> operator()(const real_vector<RealType>& v) const {
      return { log(v), log_tag() };
    }

    //! Returns the log-value of the factor for a vector.
    RealType log(const real_vector<RealType>& v) const {
      return param_(v);
    }

    // Selectors
    //--------------------------------------------------------------------------

    // Bring the immutable selectors from the base into the scope.
    using base::head;
    using base::tail;
    using base::dim;
    using base::dims;

    /**
     * Returns a mutable canonical_gaussian selector referencing the head
     * dimensions of this expression.
     */
    canonical_gaussian_selector<front, canonical_gaussian>
    head(std::size_t n) {
      return { front(n), *this };
    }

    /**
     * Returns a mutable canonical_gaussian selector referencing the tail
     * dimensions of this expression.
     */
    canonical_gaussian_selector<back, canonical_gaussian>
    tail(std::size_t n) {
      return { back(arity(), n), *this };
    }


    /**
     * Returns a mutable canonical_gaussian selector referencing a single
     * dimension of this expression.
     */
    canonical_gaussian_selector<single, canonical_gaussian>
    dim(std::size_t index) {
      return { single(index), *this };
    }

    /**
     * Returns a mutable canonical_gaussian selector referencing a span of
     * dimensions of this expression.
     */
    canonical_gaussian_selector<span, canonical_gaussian>
    dims(std::size_t start, std::size_t n) {
      return { span(start, n), *this };
    }

    /**
     * Returns a canonical_gaussian selector referencing a subset of dimensions
     * of this expression.
     */
    canonical_gaussian_selector<iref, canonical_gaussian>
    dims(const uint_vector& indices) {
      return { iref(indices), *this };
    }

    // Mutations
    //--------------------------------------------------------------------------

    //! Multiplies this expression by a constant in-place.
    canonical_gaussian& operator*=(logarithmic<RealType> x) {
      param_.lm += x.lv;
      return *this;
    }

    //! Divides this expression by a constant in-place.
    canonical_gaussian& operator/=(logarithmic<RealType> x) {
      param_.lm -= x.lv;
      return *this;
    }

    //! Multiplies another expression into this expression element-wise.
    template <typename Other>
    canonical_gaussian&
    operator*=(const canonical_gaussian_base<RealType, Other>& f){
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(plus_assign<>(), param_);
      return *this;
    }

    //! Divides another expression into this expression element-wise.
    template <typename Other>
    canonical_gaussian&
    operator/=(const canonical_gaussian_base<RealType, Other>& f){
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(minus_assign<>(), param_);
      return *this;
    }

    //! Normalizes this expression in-place.
    void normalize() {
      param_.lm -= param_.marginal();
    }

    // Factor evaluation
    //--------------------------------------------------------------------------

    const canonical_gaussian<RealType>& eval() const {
      return *this;
    }

    bool alias(const param_type& param) const {
      return &param == &param_;
    }

  private:
    //! The parameters of the factor, encapsulated as a struct.
    param_type param_;

  }; // class canonical_gaussian

} }  // namespace libgm::experimental

#endif
