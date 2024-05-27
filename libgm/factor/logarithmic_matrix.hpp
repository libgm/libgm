#ifndef LIBGM_FACTOR_LOGARITHMIC_MATRIX_HPP
#define LIBGM_FACTOR_LOGARITHMIC_MATRIX_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/factor/expression/matrix_base.hpp>
#include <libgm/factor/expression/matrix_function.hpp>
#include <libgm/factor/expression/matrix_selector.hpp>
#include <libgm/factor/expression/matrix_transform.hpp>
#include <libgm/factor/expression/matrix_view.hpp>
#include <libgm/factor/expression/vector_view.hpp>
#include <libgm/factor/expression/table_function.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/logarithmic_matrix_ll.hpp>
#include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {


/**
 * The base class for logarithmic_matrix factors and expressions.
 *
 * \tparam RealType
 *         The real type representing the parameters.
 * \tparam Derived
 *         The expression type that derives from this base class.
 *         The type must implement the following functions:
 *         alias(), eval_to().
 */
template <typename T>
class LogarithmicArray2D {
public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    using real_type   = RealType;
    using result_type = logarithmic<RealType>;
    using factor_type = logarithmic_matrix<RealType>;

    // ParametricFactor member types
    using param_type = dense_matrix<RealType>;
    using shape_type = std::pair<std::size_t, std::size_t>;
    using distribution_type = bivariate_categorical_distribution<RealType>;

    // Constructors and casts
    //--------------------------------------------------------------------------

    //! Default constructor.
    matrix_base() { }

    //! Downcasts this object to the derived type.
    Derived& derived() {
      return static_cast<Derived&>(*this);
    }

    //! Downcasts this object to the derived type.
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
     * Returns true if the two expressions have the same arguments and
     * parameters.
     */
    template <typename Other>
    friend bool
    operator==(const matrix_base& f,
               const matrix_base<log_tag, RealType, Other>& g) {
      return f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same arguments or
     * parameters
     */
    template <typename Other>
    friend bool
    operator!=(const matrix_base& f,
               const matrix_base<log_tag, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream& operator<<(std::ostream& out, const matrix_base& f) {
      out << f.derived().param();
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a matrix expression in the specified ResultSpace, representing an
     * element-wise transform of this expression with a unary operation.
     */
    template <typename ResultSpace = log_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) const {
      return make_matrix_transform<ResultSpace>(unary_op, std::tie(derived()));
    }

    /**
     * Returns a logarithmic_matrix expression representing the element-wise
     * product of a logarithmic_matrix expression and a scalar.
     */
    friend auto operator*(const matrix_base& f, logarithmic<RealType> x) {
      return f.derived().transform(incremented_by<RealType>(x.lv));
    }

    /**
     * Returns a logarithmic_matrix expression representing the element-wise
     * product of a scalar and a logarithmic_matrix expression.
     */
    friend auto operator*(logarithmic<RealType> x, const matrix_base& f) {
      return f.derived().transform(incremented_by<RealType>(x.lv));
    }

    /**
     * Returns a logarithmic_matrix expression representing the element-wise
     * division of a logarithmic_matrix expression and a scalar.
     */
    friend auto operator/(const matrix_base& f, logarithmic<RealType> x) {
      return f.derived().transform(decremented_by<RealType>(x.lv));
    }

    /**
     * Returns a logarithmic_matrix expression representing the element-wise
     * division of a scalar and a logarithmic_matrix expression.
     */
    friend auto operator/(logarithmic<RealType> x, const matrix_base& f) {
      return f.derived().transform(subtracted_from<RealType>(x.lv));
    }

    /**
     * Returns a logarithmic_matrix expression representing the
     * logarithmic_matrix expression raised to an exponent element-wise.
     */
    friend auto pow(const matrix_base& f, RealType x) {
      return f.derived().transform(multiplied_by<RealType>(x));
    }

    /**
     * Returns a logarithmic_matrix expression representing the element-wise
     * sum of two logarithmic_matrix expressions.
     */
    template <typename Other>
    friend auto operator+(const matrix_base& f,
                          const matrix_base<log_tag, RealType, Other>& g) {
      return libgm::experimental::transform(log_plus_exp<>(), f, g);
    }

    /**
     * Returns a logarithmic_matrix expression representing the product of
     * two logarithmic_matrix expressions.
     */
    template <typename Other>
    friend auto operator*(const matrix_base& f,
                          const matrix_base<log_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::plus<>(), f, g);
    }

    /**
     * Returns a logarithmic_matrix expression representing the division of
     * two logarithmic_matrix expressions.
     */
    template <typename Other>
    friend auto operator/(const matrix_base& f,
                          const matrix_base<log_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::minus<>(), f, g);
    }

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * maximum of two logarithmic_matrix expressions.
     */
    template <typename Other>
    friend auto max(const matrix_base& f,
                    const matrix_base<log_tag, RealType, Other>& g) {
      return libgm::experimental::transform(member_max(), f, g);
    }

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * minimum of two logarithmic_matrix expressions.
     */
    template <typename Other>
    friend auto min(const matrix_base& f,
                    const matrix_base<log_tag, RealType, Other>& g) {
      return libgm::experimental::transform(member_min(), f, g);
    }

    /**
     * Returns a logarithmic_matrix expression representing \f$f*(1-a) + g*a\f$
     * for two logarithmic_matrix expressions f and g.
     */
    template <typename Other>
    friend auto weighted_update(const matrix_base& f,
                                const matrix_base<log_tag, RealType, Other>& g,
                                RealType x) {
      return libgm::experimental::transform(weighted_plus<RealType>(1-x, x), f, g);
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_matrix expression with the elements of this
     * expression cast to a different RealType.
     */
    template <typename NewRealType>
    auto cast() const {
      return derived().transform(member_cast<NewRealType>());
    }

    /**
     * Returns a probability_matrix expression equivalent to this expression.
     */
    auto probability() const {
      return derived().template transform<prob_tag>(exponent<>());
    }

    /**
     * Returns a logarithmic_table expression equivalent to this matrix.
     */
    auto table() const {
      return table_from_matrix<log_tag>(derived()); // in table_function.hpp
    }

    // Aggregates
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_vector expression representing the aggregate
     * of this expression over a single argument.
     * The second argument, if provided, must be equal to 1.
     */
    template <typename AggOp>
    auto aggregate(AggOp agg_op, std::size_t retain) const {
      assert(retain <= 1);
      return make_vector_function<log_tag>(
        [agg_op, retain](const Derived& f, dense_vector<RealType>& result) {
          if (retain == 0) {
            result = agg_op(f.param().rowwise());
          } else {
            result = agg_op(f.param().colwise()).transpose();
          }
        }, derived());
    }

    /**
     * Returns a logarithmic_vector expression representing the marginal
     * of this of expression over a single argument.
     *
     * The second, optional argument is provided for compatibility with other
     * factors; if provided, it must be equal to 1.
     */
    auto marginal(std::size_t retain, std::size_t n = 1) const {
      assert(n == 1);
      return derived().aggregate(member_logSumExpVectorwise(), retain);
    }

    /**
     * Returns a logarithmic_vector expression representing the maximum
     * of this expression over a single argument.
     *
     * The second, optional argument is provided for compatibility with other
     * factors; if provided, it must be equal to 1.
     */
    auto maximum(std::size_t retain, std::size_t n = 1) const {
      assert(n == 1);
      return derived().aggregate(member_maxCoeff(), retain);
    }

    /**
     * Returns a logarithmic_vector expression representing the minimum
     * of this expression over a single argument.
     *
     * The second, optional argument is provided for compatibility with other
     * factors; if provided, it must be equal to 1.
     */
    auto minimum(std::size_t retain, std::size_t n = 1) const {
      assert(n == 1);
      return derived().aggregate(member_minCoeff(), retain);
    }

    /**
     * Computes the normalization constant of this expression.
     */
    logarithmic<RealType> sum() const {
      return { derived().accumulate(member_logSumExp()), log_tag() };
    }

    /**
     * Computes the maximum value of this expression.
     */
    logarithmic<RealType> max() const {
      return { derived().accumulate(member_maxCoeff()), log_tag() };
    }

    /**
     * Computes the minimum value of this expression.
     */
    logarithmic<RealType> min() const {
      return { derived().accumulate(member_minCoeff()), log_tag() };
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding row and column.
     */
    logarithmic<RealType> max(std::size_t& row, std::size_t& col) const {
      return { derived().accumulate(member_maxCoeffIndex(&row, &col)),
               log_tag() };
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding index to a vector.
     */
    logarithmic<RealType> max(uint_vector& index) const {
      index.resize(2);
      return maximum(index.front(), index.back());
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding row and column.
     */
    logarithmic<RealType> min(std::size_t& row, std::size_t& col) const {
      return { derived().accumulate(member_minCoeffIndex(&row, &col)),
               log_tag() };
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding index to a vector.
     */
    logarithmic<RealType> min(uint_vector& index) const {
      index.resize(2);
      return minimum(index.front(), index.back());
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return max().lv > -inf<RealType>();
    }

    // Conditioning
    //--------------------------------------------------------------------------

    /**
     * If this expression represents a marginal distribution p(x, y), this
     * function returns a probability_matrix expression representing the
     * conditional p(x | y) with 1 tail (front) dimension.
     *
     * The optional argument must be always 1.
     */
    auto conditional(std::size_t nhead = 1) const {
      assert(nhead == 1);
      return make_matrix_function<log_tag>(
        [](const Derived& f, dense_matrix<RealType>& result) {
          f.eval_to(result);
          result.array().rowise() -= member_logSumExp()(result.array().colwise());
        }, derived());
    }

    /**
     * Returns a logarithmic_vector expression representing a row of this
     * expression. The factor provides an optimized version of this expression.
     */
    auto row(std::size_t index) const {
      return make_vector_function<log_tag>(
        [index](const Derived& f, dense_vector<RealType>& result) {
          result = f.row(index).transpose();
        }, derived());
    }

    /**
     * Returns a logarithmic_vector expression representing a column of this
     * expression. The factor provides an optimized version of this expression.
     */
    auto col(std::size_t index) const {
      return make_vector_function<log_tag>(
        [index](const Derived& f, dense_vector<RealType>& result) {
          result = f.col(index);
        }, derived());
    }

    /**
     * Returns a logarithmic_vector expression representing a row of this
     * expression.
     */
    auto restrict_head(std::size_t row) const {
      return derived().row(row);
    }

    /**
     * Returns a logarithmic_vector expression representing a column of this
     * expression.
     */
    auto restrict_tail(std::size_t col) const {
      return derived().col(col);
    }

    /**
     * Returns a probability_vector expression resulting when restricting the
     * specified dimension of this expression to the specified value.
     * Use 0 to restrict the row, 1 to restrict a column.
     */
    auto restrict(std::size_t dim, std::size_t index) const {
      assert(dim <= 1);
      return make_vector_function<log_tag>(
        [dim, index](const Derived& f, dense_vector<RealType>& result) {
          if (dim == 1) {
            result = f.param().col(index);
          } else {
            result = f.param().row(index).transpose();
          }
        }, derived());
    }

    // Reshaping
    //--------------------------------------------------------------------------

    /**
     * Returns the expression representing the transpose of this expression.
     */
    auto transpose() const {
      return make_matrix_function<log_tag>(
        [](const Derived& f, dense_matrix<RealType>& result) {
          result = f.param().transpose();
        }, derived());
    }

    // Selectors
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_matrix selector referencing the rows of
     * this expression (i.e., performing column-wise operations).
     */
    matrix_selector<log_tag, Eigen::Vertical, const Derived>
    colwise() const {
      return derived();
    }

    /**
     * Returns a logarithmic_matrix selector referencing the columns of
     * this of this expression (i.e., performing row-wise operations).
     */
    matrix_selector<log_tag, Eigen::Horizontal, const Derived>
    rowwise() const {
      return derived();
    }

    /**
     * Returns a logarithmic_matrix selector referencing the rows of
     * this expression (i.e., performing column-wise operations).
     *
     * The optional argument is provided for compatibility with other factors
     * and, if specified, must be equal to 1.
     */
    matrix_selector<log_tag, Eigen::Vertical, const Derived>
    head(std::size_t n = 1) const {
      assert(n == 1);
      return derived();
    }

    /**
     * Returns a logarithmic_matrix selector referencing the columns of
     * this of this expression (i.e., performing row-wise operations).
     *
     * The optional argument is provided for compatibility with other factors
     * and, if specified, must be equal to 1.
     */
    matrix_selector<log_tag, Eigen::Horizontal, const Derived>
    tail(std::size_t n = 1) const {
      assert(n == 1);
      return derived();
    }

    /**
     * Returns a logarithmic_matrix selector referencing a single
     * dimension of this expression. The only valid dimensions are 0 and 1,
     * with 0 representing column-wise operations and 1 row-wise operations.
     */
    matrix_selector<log_tag, Eigen::BothDirections, const Derived>
    dim(std::size_t index) const {
      return { index, derived() };
    }

    /**
     * Returns a logarithmic_matrix selector referencing a single
     * dimension of this expression. The only valid dimensions are 0 and 1,
     * with 0 representing column-wise operations and 1 row-wise operations.
     * The second argument must be equal to 1.
     */
    matrix_selector<log_tag, Eigen::BothDirections, const Derived>
    dims(std::size_t index, std::size_t n = 1) const {
      assert(n == 1);
      return { index, derived() };
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a categorical distribution represented by this expression.
     */
    bivariate_categorical_distribution<RealType> distribution() const {
      return { derived().param(), log_tag() };
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression.
     *
     * \throw std::out_of_range
     *        may be thrown if the distribution is not normalized
     */
    template <typename Generator>
    std::pair<std::size_t, std::size_t> sample(Generator& rng) const {
      RealType p = std::uniform_real_distribution<RealType>()(rng);
      return derived().find_if(
        compose(partial_sum_greater_than<RealType>(p), exponent<RealType>())
      );
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression, storing the result in an output vector.
     *
     * \throw std::out_of_range
     *        may be thrown if the distribution is not normalized
     */
    template <typename Generator>
    void sample(Generator& rng, uint_vector& result) const {
      result.resize(2);
      std::tie(result.front(), result.back()) = sample(rng);
    }

    // Entropy and divergences
    //--------------------------------------------------------------------------

    /**
     * Computes the entropy for the distribution represented by this factor.
     */
    RealType entropy() const {
      auto&& param = derived().param();
      auto plus_entropy
        = compose_right(std::plus<RealType>(), entropy_log_op<RealType>());
      return std::accumulate(param.data(), param.data() + param.size(),
                             RealType(0), plus_entropy);
    }

    /**
     * Computes the entropy for a single argument of the distribution
     * represented by this expression.
     * The second argument, if provided, must be equal to 1.
     */
    RealType entropy(std::size_t dim, std::size_t n = 1) const {
      assert(n == 1);
      return derived().marginal(dim).entropy();
    }

    /**
     * Computes the mutual information between the two dimensions (arguments)
     * of the distribution represented by this expression.
     */
    RealType mutual_information() const {
      return entropy(0) + entropy(1) - entropy();
    }

    /**
     * Computes the mutual information between two (not necessarily unique)
     * dimensions of the distribution represented by this expression.
     * The two optional arguments, if provided, must be both equal to 1.
     */
    RealType mutual_information(std::size_t a, std::size_t b,
                                std::size_t na = 1, std::size_t nb = 1) const {
      assert(a <= 1 && b <= 1);
      assert(na == 1 && nb == 1);
      if (a == b) {
        return entropy(a);
      } else {
        return entropy(0) + entropy(1) - entropy();
      }
    }

    /**
     * Computes the cross entropy from p to q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const matrix_base& p,
                  const matrix_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        entropy_log_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Kullback-Leibler divergence from p to q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const matrix_base& p,
                  const matrix_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        kld_log_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Jensen–Shannon divergece between p and q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    js_divergence(const matrix_base& p,
                  const matrix_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        jsd_log_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the sum of absolute differences between parameters of p and q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    sum_diff(const matrix_base& p,
             const matrix_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the max of absolute differences between parameters of p and q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    max_diff(const matrix_base& p,
             const matrix_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), libgm::maximum<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    // Expression evaluations
    //--------------------------------------------------------------------------

    //! Evaluates the expression to a parameter matrix.
    param_type param() const {
      param_type tmp; derived().eval_to(tmp); return tmp;
    }

    /**
     * Returns the logarithmic_matrix object resulting from evaluating this
     * expression.
     */
    logarithmic_matrix<RealType> eval() const {
      return *this;
    }

    /**
     * Updates the result with the given assignment operator. Calling this
     * function is guaranteed to be safe even in the presence of aliasing.
     */
    template <typename AssignOp>
    void transform_inplace(AssignOp op, dense_matrix<RealType>& result) const {
      op(result.array(), derived().param().array());
    }

    /**
     * Accumulates the parameters with the given operator.
     */
    template <typename AccuOp>
    RealType accumulate(AccuOp op) const {
      return op(derived().param());
    }

    /**
     * Identifies the first element in the lexicographical ordering that
     * satisfie the given predicate and returns the corresponding row/column.
     *
     * \throw std::out_of_range if the element cannot be found.
     */
    template <typename UnaryPredicate>
    std::pair<std::size_t, std::size_t> find_if(UnaryPredicate pred) const {
      auto&& p = derived().param();
      std::size_t n = std::find(p.data(), p.data() + p.size(), pred) - p.data();
      if (n == p.size()) {
        throw std::out_of_range("Element could not be found");
      } else {
        return { n % p.rows(), n / p.rows() };
      }
    }

  }; // class logarithmic_matrix_base


  // Factor
  //============================================================================

  /**
   * A factor of a categorical logarithmic distribution whose domain
   * consists of two arguments. The factor represents a non-negative
   * function directly with a parameter array \theta as f(X=x, Y=y | \theta) =
   * \theta_{x,y}. In some cases, this class represents a array of probabilities
   * (e.g., when used as a prior in a hidden Markov model). In other cases,
   * e.g. in a pairwise Markov network, there are no constraints on the
   * normalization of f.
   *
   * \tparam RealType a type of values stored in the factor
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename RealType = double>
  class logarithmic_matrix
    : public matrix_base<log_tag, RealType, logarithmic_matrix<RealType> > {

    using base = matrix_base<log_tag, RealType, logarithmic_matrix>;

  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor member types
    using ll_type = logarithmic_matrix_ll<RealType>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty factor.
    logarithmic_matrix() { }

    //! Constructs a factor with the given shape and uninitialized parameters.
    logarithmic_matrix(std::size_t rows, std::size_t cols)
      : param_(rows, cols) { }

    //! Constructs a factor with the given shape and uninitialized parameters.
    explicit logarithmic_matrix(std::pair<std::size_t, std::size_t> shape)
      : param_(shape.first, shape.second) { }


    //! Constructs a factor with the given shape and constant value.
    logarithmic_matrix(std::size_t rows, std::size_t cols,
                       logarithmic<RealType> x)
      : param_(rows, cols) {
      param_.fill(x.lv);
    }

    //! Constructs a factor with the given shape and constant value.
    logarithmic_matrix(std::pair<std::size_t, std::size_t> shape,
                       logarithmic<RealType> x)
      : param_(shape.first, shape.second) {
      param_.fill(x.lv);
    }

    //! Constructs a factor with the given parameters.
    logarithmic_matrix(const dense_matrix<RealType>& param)
      : param_(param) { }

    //! Constructs a factor with the given argument and parameters.
    logarithmic_matrix(dense_matrix<RealType>&& param)
      : param_(std::move(param)) { }

    //! Constructs a factor with the given arguments and parameters.
    logarithmic_matrix(std::size_t rows,
                       std::size_t cols,
                       std::initializer_list<RealType> values)
      : param_(rows, cols) {
      assert(param_.size() == values.size());
      std::copy(values.begin(), values.end(), param_.data());
    }

    //! Constructs a factor from an expression.
    template <typename Other>
    logarithmic_matrix(const matrix_base<log_tag, RealType, Other>& f) {
      f.derived().eval_to(param_);
    }

    //! Assigns the result of an expression to this factor.
    template <typename Other>
    logarithmic_matrix&
    operator=(const matrix_base<log_tag, RealType, Other>& f) {
      assert(!f.derived().alias(param_));
      f.derived().eval_to(param_);
      return *this;
    }

    //! Swaps the content of two logarithmic_matrix factors.
    friend void swap(logarithmic_matrix& f, logarithmic_matrix& g) {
      f.param_.swap(g.param_);
    }

    //! Serializes the members.
    void save(oarchive& ar) const {
      ar << param_;
    }

    //! Deserializes members.
    void load(iarchive& ar) {
      ar >> param_;
    }

    //! Resets the content of this factor to the given shape.
    void reset(std::size_t rows, std::size_t cols) {
      param_.resize(rows, cols);
    }

    //! Resets the content of this factor to the given shape.
    void reset(std::pair<std::size_t, std::size_t> shape) {
      param_.resize(shape.first, shape.second);
    }

    //! Returns the shape of the matrix corresponding to a domain.
    template <typename Arg>
    static std::pair<std::size_t, std::size_t> shape(const domain<Arg>& dom) {
      assert(dom.size() == 2 &&
             argument_arity(dom.front()) == 1 &&
             argument_arity(dom.back()) == 1);
      return { argument_values(dom.front()), argument_values(dom.back()) };
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the number of arguments of this expression.
    std::size_t arity() const {
      return 2;
    }

    //! Returns the number of rows of the expression.
    std::size_t rows() const {
      return param_.rows();
    }

    //! Returns the number of columns of the expression.
    std::size_t cols() const {
      return param_.cols();
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return param_.size();
    }

    /**
     * Returns the pointer to the first parameter or nullptr if the factor is
     * empty.
     */
    RealType* begin() {
      return param_.data();
    }

    /**
     * Returns the pointer to the first parameter or nullptr if the factor is
     * empty.
     */
    const RealType* begin() const {
      return param_.data();
    }

    /**
     * Returns the pointer past the last parameter or nullptr if the factor is
     * empty.
     */
    RealType* end() {
      return param_.data() + param_.size();
    }

    /**
     * Returns the pointer past the last parameter or nullptr if the factor is
     * empty.
     */
    const RealType* end() const {
      return param_.data() + param_.size();
    }

    //! Returns true if the expression has no data (same as size() == 0).
    bool empty() const {
      return param_.data() == nullptr;
    }

    //! Provides mutable access to the parameter array of this factor.
    dense_matrix<RealType>& param() {
      return param_;
    }

    //! Returns the parameter array of this factor.
    const dense_matrix<RealType>& param() const {
      return param_;
    }

    //! Returns the parameter for the given row and column.
    RealType& param(std::size_t row, std::size_t col) {
      return param_(row, col);
    }

    //! Returns the parameter for the given row.
    const RealType& param(std::size_t row, std::size_t col) const {
      return param_(row, col);
    }

    //! Returns the parameter for the given index.
    RealType& param(const uint_vector& index) {
      assert(index.size() == 2);
      return param_(index[0], index[1]);
    }

    //! Returns the parameter of rthe given index.
    const RealType& param(const uint_vector& index) const {
      assert(index.size() == 2);
      return param_(index[0], index[1]);
    }

    //! Returns the parameter with the given linear index.
    RealType& operator[](std::size_t i) {
      return param_(i);
    }

    //! Returns the parameter with the given linear index.
    const RealType& operator[](std::size_t i) const {
      return param_(i);
    }

    //! Returns the value of the expression for the given row and column.
    logarithmic<RealType> operator()(std::size_t row, std::size_t col) const {
      return { param(row, col), log_tag() };
    }

    //! Returns the value of the expression for the given index.
    logarithmic<RealType> operator()(const uint_vector& index) const {
      return { param(index), log_tag() };
    }

    //! Returns the log-value of the expression for the given row and column.
    RealType log(std::size_t row, std::size_t col) const {
      return param(row, col);
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return param(index);
    }

    // Optimized expressions
    //--------------------------------------------------------------------------

    /**
     * Returns the the logarithmic_matrix expression converting the elements
     * of this factor to the given type.
     */
    template <typename NewRealType>
    auto cast() const {
      return make_matrix_view<log_tag>(
        [](const logarithmic_matrix& f) -> decltype(auto) {
          return f.param().template cast<NewRealType>();
        }, *this);
    }

    /**
     * Returns a probability_vector expression representing a row of this
     * factor.
     */
    auto row(std::size_t index) const {
      return make_vector_view<log_tag>(
        [index](const logarithmic_matrix& f) {
          return f.param().row(index).transpose(); },
        *this);
    }

    /**
     * Returns a probability_vector expression representing a column of this
     * factor.
     */
    auto col(std::size_t index) const {
      return make_vector_view<log_tag>(
        [index](const logarithmic_matrix& f) { return f.param().col(index); },
        *this);
    }

    /**
     * Returns the probability_matrix expression representing the transpose of
     * this factor.
     */
    auto transpose() const {
      return make_matrix_view<log_tag>(
        [](const logarithmic_matrix& f) { return f.param().transpose(); },
        *this);
    }

    // Selectors
    //--------------------------------------------------------------------------

    // Bring the immutable selectors from the base into the scope.
    using base::colwise;
    using base::rowwise;
    using base::head;
    using base::tail;
    using base::dim;
    using base::dims;

    /**
     * Returns a mutable logarithmic_matrix selector referencing the rows of
     * this expression (i.e., performing column-wise operations).
     */
    matrix_selector<log_tag, Eigen::Vertical, logarithmic_matrix>
    colwise() {
      return *this;
    }

    /**
     * Returns a mutable logarithmic_matrix selector referencing the columns of
     * this of this expression (i.e., performing row-wise operations).
     */
    matrix_selector<log_tag, Eigen::Horizontal, logarithmic_matrix>
    rowwise() {
      return *this;
    }

    /**
     * Returns a mutable logarithmic_matrix selector referencing the rows of
     * this expression (i.e., performing column-wise operations).
     *
     * The optional argument is provided for compatibility with other factors
     * and, if specified, must be equal to 1.
     */
    matrix_selector<log_tag, Eigen::Vertical, logarithmic_matrix>
    head(std::size_t n = 1) {
      assert(n == 1);
      return *this;
    }

    /**
     * Returns a mutable logarithmic_matrix selector referencing the columns of
     * this of this expression (i.e., performing row-wise operations).
     *
     * The optional argument is provided for compatibility with other factors
     * and, if specified, must be equal to 1.
     */
    matrix_selector<log_tag, Eigen::Horizontal, logarithmic_matrix>
    tail(std::size_t n = 1) {
      assert(n == 1);
      return *this;
    }

    /**
     * Returns a mutable logarithmic_matrix selector referencing a single
     * dimension of this expression. The only valid dimensions are 0 and 1,
     * with 0 representing column-wise operations and 1 row-wise operations.
     */
    matrix_selector<log_tag, Eigen::BothDirections, logarithmic_matrix>
    dim(std::size_t index) {
      return { index, *this };
    }

    /**
     * Returns a mutable logarithmic_matrix selector referencing a single
     * dimension of this expression. The only valid dimensions are 0 and 1,
     * with 0 representing column-wise operations and 1 row-wise operations.
     * The second argument must be equal to 1.
     */
    matrix_selector<log_tag, Eigen::BothDirections, logarithmic_matrix>
    dims(std::size_t index, std::size_t n = 1) {
      assert(n == 1);
      return { index, *this };
    }

    // Mutations
    //--------------------------------------------------------------------------

    //! Multiplies this factor by a constant.
    logarithmic_matrix& operator*=(logarithmic<RealType> x) {
      param_.array() += x.lv;
      return *this;
    }

    //! Divides this factor by a constant.
    logarithmic_matrix& operator/=(logarithmic<RealType> x) {
      param_.array() -= x.lv;
      return *this;
    }

    //! Multiplies a logarithmic_matrix expression into this factor.
    template <typename Other>
    logarithmic_matrix&
    operator*=(const matrix_base<log_tag, RealType, Other>& f){
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(plus_assign<>(), param_);
      return *this;
    }

    //! Divides a logarithmic_matrix expression into this factor.
    template <typename Other>
    logarithmic_matrix&
    operator/=(const matrix_base<log_tag, RealType, Other>& f){
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(minus_assign<>(), param_);
      return *this;
    }

    //! Divides this factor by its norm inplace.
    void normalize() {
      *this /= this->sum();
    }

    // Evaluation
    //--------------------------------------------------------------------------

    //! Returns this logarithmic_matrix (a noop).
    const logarithmic_matrix& eval() const {
      return *this;
    }

    //! Copies the parameters to the given matrix.
    void eval_to(dense_matrix<RealType>& result) const {
      if (result != param_) {
        result = param_;
      }
    }

    /**
     * Returns true if this logarithmic_matrix aliases the given parameters,
     * if.e., if evaluating an expression involving this logarithmic_matrix
     * to param requires a temporary.
     *
     * This function must be defined by each logarithmic_matrix expression.
     */
    bool alias(const dense_vector<RealType>& param) const {
      return false;
    }

    /**
     * Returns true if this logarithmic_matrix aliases the given parameters,
     * i.e., if evaluating an expression involving this logarithmic_matrix
     * to param requires a temporary.
     *
     * This function must be defined by each logarithmic_matrix expression.
     */
    bool alias(const dense_matrix<RealType>& param) const {
      return &param_ == &param;
    }

  private:
    //! The parameters of the factor, i.e., a matrix of log-probabilities.
    dense_matrix<RealType> param_;

  }; // class logarithmic_matrix

} // namespace libgm

#endif
