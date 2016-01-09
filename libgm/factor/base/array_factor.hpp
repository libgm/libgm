#ifndef LIBGM_ARRAY_FACTOR_HPP
#define LIBGM_ARRAY_FACTOR_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/array_domain.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/factor/base/factor.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/serialization/eigen.hpp>

#include <Eigen/Core>

#include <sstream>
#include <stdexcept>
#include <initializer_list>
#include <numeric>
#include <type_traits>
#include <utility>

namespace libgm {

  /**
   * A base class for discrete factors with a fixed number of either
   * one or two arguments. This class stores the parameters of the factor
   * as an Eigen array and provides the implementations of standard
   * functions on the factors. This class does not model the Factor
   * concept.
   *
   * \tparam N the arity of the factor (must be either 1 or 2).
   * \tparam T the type of parameters stored in the table.
   * \see canonical_array, probability_array
   */
  template <typename Arg, std::size_t N, typename T>
  class array_factor : public factor {
    static_assert(is_discrete<Arg>::value,
                  "Array factors require Arg to be discrete");
    static_assert(is_univariate<Arg>::value,
                  "Array factors require Arg to be univariate");
    static_assert(N == 1 || N == 2,
                  "The arity of factor must be 1 or 2");

  public:
    // Underlying representation
    typedef Eigen::Array<T, Eigen::Dynamic, N == 1 ? 1 : Eigen::Dynamic>
      array_type;

    // Range types
    typedef T*       iterator;
    typedef const T* const_iterator;
    typedef T        value_type;

    // Arguments
    typedef argument_traits<Arg> arg_traits;
    typedef array_domain<Arg, N> domain_type;
    typedef uint_assignment<Arg> assignment_type;

    // Constructors
    //==========================================================================

    /**
     * Default constructor. Creates an empty factor.
     */
    array_factor() { }

    /**
     * Constructs an array_factor with the given arguments and initializes its
     * parameters to the given array.
     */
    array_factor(const domain_type& args, const array_type& param)
      : args_(args), param_(param) {
      check_param();
    }

    /**
     * Constructs an array_factor with the given arguments and moves its
     * parameters from the given array.
     */
    array_factor(const domain_type& args, array_type&& param)
      : args_(args) {
      param_.swap(param);
      check_param();
    }

    /**
     * Constructs an array_factor with the given arguments and parameters.
     */
    array_factor(const domain_type& args, std::initializer_list<T> init) {
      reset(args);
      assert(size() == init.size());
      std::copy(init.begin(), init.end(), begin());
    }

    //! Copy constructor.
    array_factor(const array_factor& other) = default;

    //! Move constructor.
    array_factor(array_factor&& other) {
      swap(other);
    }

    //! Assignment operator.
    array_factor& operator=(const array_factor& other) = default;

    //! Move assignment operator.
    array_factor& operator=(array_factor&& other) {
      swap(other);
      return *this;
    }

    // Serialization and initialization
    //==========================================================================

    //! Serializes members.
    void save(oarchive& ar) const {
      ar << args_ << param_;
    }

    //! Deserializes members.
    void load(iarchive& ar) {
      ar >> args_ >> param_;
      check_param();
    }

    /**
     * Resets the content of this factor to the given arguments.
     * The array elements may be invalidated.
     */
    void reset(const domain_type& args) {
      if (args_ != args || empty()) {
        args_ = args;
        if (N == 1) {
          param_.resize(arg_traits::num_values(x()), 1);
        } else {
          param_.resize(arg_traits::num_values(x()),
                        arg_traits::num_values(y()));
        }
      }
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this factor.
    const domain_type& arguments() const {
      return args_;
    }

    //! Returns the first argument or null if the factor is empty or nullary.
    Arg x() const {
      return args_[0];
    }

    //! Returns the second argument or null if the factor has <=1 arguments.
    Arg y() const {
      return N == 2 ? args_[1] : Arg();
    }

    //! Returns the number of arguments of this factor.
    constexpr std::size_t arity() const {
      return N;
    }

    //! Returns the total number of elements of the factor.
    std::size_t size() const {
      return param_.size();
    }

    //! Returns true if the factor is empty (equivalent to size() == 0).
    bool empty() const {
      return !param_.data();
    }

    //! Returns the pointer to the first element (null if the factor is empty).
    T* begin() {
      return param_.data();
    }

    //! Returns the pointer to the first element (null if the factor is empty).
    const T* begin() const {
      return param_.data();
    }

    //! Returns the pointer past the last element (null if the factor is empty).
    T* end() {
      return param_.data() + param_.size();
    }

    //! Returns the pointer past the last element (null if the factor is empty).
    const T* end() const {
      return param_.data() + param_.size();
    }

    //! Returns the parameter with the given linear index.
    T& operator[](std::size_t i) {
      return param_(i);
    }

    //! Returns the parameter with the given linear index.
    const T& operator[](std::size_t i) const {
      return param_(i);
    }

    //! Provides mutable access to the parameter array of this factor.
    array_type& param() {
      return param_;
    }

    //! Returns the parameter array of this factor.
    const array_type& param() const {
      return param_;
    }

    //! Returns the parameter for the given assignment.
    T& param(const assignment_type& a) {
      return param_(linear_index(a));
    }

    //! Returns the parameter for the given assignment.
    const T& param(const assignment_type& a) const {
      return param_(linear_index(a));
    }

    //! Returns the parameter for the given index.
    T& param(const uint_vector& index) {
      return param_(linear_index(index));
    }

    //! Returns the parameter of rthe given index.
    const T& param(const uint_vector& index) const {
      return param_(linear_index(index));
    }

    // Indexing
    //==========================================================================

    /**
     * Returns the size of the parameter array for a single argument.
     */
    static std::size_t param_shape(const array_domain<Arg, 1>& args) {
      return arg_traits::num_values(args[0]);
    }

    /**
     * Returns the shape of the parmeter array for 2-argument domain.
     */
    static std::pair<std::size_t, std::size_t>
    param_shape(const array_domain<Arg, 2>& args) {
      return {
        arg_traits::num_values(args[0]), arg_traits::num_values(args[1])
      };
    }

    /**
     * Converts a linear index to the corresponding assignment to the
     * factor arguments.
     */
    void assignment(std::size_t linear_index, assignment_type& a) const {
      if (N == 1) {
        a[x()] = linear_index;
      } else {
        a[x()] = linear_index % param_.rows();
        a[y()] = linear_index / param_.rows();
      }
    }

    /**
     * Returns the linear index corresponding to the given assignment.
     */
    std::size_t linear_index(const assignment_type& a) const {
      if (N == 1) {
        return a.at(x());
      } else {
        return a.at(x()) + a.at(y()) * param_.rows();
      }
    }

    /**
     * Returns the linear index corresponding to the given finite index.
     */
    std::size_t linear_index(const uint_vector& index) const {
      if (index.size() != N) {
        throw std::invalid_argument("Index size does not match the arity");
      }
      if (N == 1) {
        return index[0];
      } else {
        return index[0] + index[1] * param_.rows();
      }
    }

    /**
     * Substitutes this factor's arguments according to the given map
     * in place.
     */
    void subst_args(const std::unordered_map<Arg, Arg>& var_map) {
      for (Arg& x : args_) {
        Arg xn = var_map.at(x);
        if (!arg_traits::compatible(x, xn)) {
          std::ostringstream out;
          out << "subst_args: "; arg_traits::print(out, x);
          out << " and "; arg_traits::print(out, xn);
          out << " are not compatible";
          throw std::invalid_argument(out.str());
        }
        x = xn;
      }
    }

    /**
     * Checks if ths dimensions of the parameter array match the factor
     * arguments.
     * \throw std::runtime_error if some of the dimensions do not match
     */
    void check_param() const {
      if (param_.rows() != arg_traits::num_values(x())) {
        throw std::runtime_error("Invalid number of rows");
      }
      if (param_.cols() != (N == 2 ? arg_traits::num_values(y()) : 1)) {
        throw std::runtime_error("Invalid number of columns");
      }
    }

  protected:
    // Protected members
    //========================================================================

    /**
     * Implementation of the swap function. This function must be protected,
     * because it is not type-safe.
     */
    void swap(array_factor& other) {
      if (this != &other) {
        using std::swap;
        swap(args_, other.args_);
        param_.swap(other.param_);
      }
    }

    /**
     * Implementation of operator==(). This function must be protected,
     * because it is not type-safe.
     */
    bool equal(const array_factor& other) const {
      return args_ == other.args_ && std::equal(begin(), end(), other.begin());
    }

    //! The arguments of this factor.
    domain_type args_;

    //! The parameter array.
    array_type param_;

  }; // class array_factor

  // Implementations of common factor operations
  //============================================================================

  /**
   * Throws an std::invalid_argument exception if the two factors do not
   * have the same argument vectors.
   * \relates array_factor
   */
  template <typename Arg, std::size_t N, typename T>
  void check_same_arguments(const array_factor<Arg, N, T>& f,
                            const array_factor<Arg, N, T>& g) {
    if (f.arguments() != g.arguments()) {
      throw std::invalid_argument(
        "Element-wise operations require the two factors "
        "to have the same arguments"
      );
    }
  }

  /**
   * Joins two factors with same arity element-wise.
   * \relates array_factor
   */
  template <typename Result,
            typename T, std::size_t N, typename Arg,
            typename Op>
  Result join(const array_factor<Arg, N, T>& f,
              const array_factor<Arg, N, T>& g,
              Op op) {
    if (f.arguments() == g.arguments()) {
      return Result(f.arguments(), op(f.param(), g.param()));
    }
    if (f.x() == g.y() && f.y() == g.x()) {
      return Result(f.arguments(), op(f.param(), g.param().transpose()));
    }
    throw std::invalid_argument("array_factor:join introduces a new argument");
  }

  /**
   * Joins a binary and a unary factor.
   * \relates array_factor
   */
  template <typename Result, typename T, typename Arg, typename Op>
  Result join(const array_factor<Arg, 2, T>& f,
              const array_factor<Arg, 1, T>& g,
              Op op) {
    const auto& a = f.param(); // 2D array
    const auto& b = g.param(); // 1D array
    typedef Eigen::Array<T, Eigen::Dynamic, 1> b_type;
    if (f.x() == g.x()) { // combine each column of f with g
      Eigen::Replicate<b_type, 1, Eigen::Dynamic> brep(b, 1, a.cols());
      return Result(f.arguments(), op(a, brep));
    }
    if (f.y() == g.x()) { // combine each row of f with g transposed
      Eigen::Replicate<b_type, 1, Eigen::Dynamic> brep(b, 1, a.rows());
      return Result(f.arguments(), op(a, brep.transpose()));
    }
    throw std::invalid_argument("array_factor: join creates a ternary factor");
  }

  /**
   * Joins a unary and a binary factor.
   * \relates array_factor
   */
  template <typename Result, typename T, typename Arg, typename Op>
  Result join(const array_factor<Arg, 1, T>& f,
              const array_factor<Arg, 2, T>& g,
              Op op) {
    const auto& a = f.param(); // 1D array
    const auto& b = g.param(); // 2D array
    typedef Eigen::Array<T, Eigen::Dynamic, 1> a_type;
    if (f.x() == g.x()) { // combine f with each column of g
      Eigen::Replicate<a_type, 1, Eigen::Dynamic> arep(a, 1, b.cols());
      return Result({g.x(), g.y()}, op(arep, b));
    }
    if (f.x() == g.y()) { // combine f with each row of g
      Eigen::Replicate<a_type, 1, Eigen::Dynamic> arep(a, 1, b.rows());
      return Result({g.y(), g.x()}, op(arep, b.transpose()));
    }
    throw std::invalid_argument("array_factor: join creates a ternary factor");
  }

  /**
   * Joins two factors with the same arity element-wise in-place
   * using a mutating operation.
   */
  template <typename T, std::size_t N, typename Arg, typename Op>
  void join_inplace(array_factor<Arg, N, T>& h,
                    const array_factor<Arg, N, T>& f,
                    Op op) {
    if (h.arguments() == f.arguments()) {
      op(h.param(), f.param());
    } else if (h.x() == f.y() && h.y() == f.x()) {
      op(h.param(), f.param().transpose());
    } else {
      throw std::invalid_argument(
        "array_factor:join_inplace introduces a new argument"
      );
    }
  }

  /**
   * Joins a binary factor with a unary factor in-place
   * using a mutating operation.
   */
  template <typename T, typename Arg, typename Op>
  void join_inplace(array_factor<Arg, 2, T>& h,
                    const array_factor<Arg, 1, T>& f, Op op) {
    if (h.x() == f.x()) {
      op(h.param().colwise(), f.param());
    } else if (h.y() == f.x()) {
      op(h.param().rowwise(), f.param().transpose());
    } else {
      throw std::invalid_argument(
        "array_factor:join_inplace introduces a new argument"
        );
    }
  }

  /**
   * Computes the expectation of the parameters of a binary factor
   * under the probabilities given by a unary factor and returns
   * a factor with the remaining variables.
   *
   * \throws std::invalid_argument if f does not contain the argument of g
   */
  template <typename Result, typename T, typename Arg>
  Result expectation(const array_factor<Arg, 2, T>& f,
                     const array_factor<Arg, 1, T>& g) {
    auto a = f.param().matrix(); // matrix
    auto b = g.param().matrix(); // vector
    if (f.y() == g.x()) {
      return Result({f.x()}, (a * b).array());
    }
    if (f.x() == g.x()) {
      return Result({f.y()}, (a.transpose() * b).array());
    }
    throw std::invalid_argument(
      "array_factor expectation: f does not contain the argument of g"
    );
  }

  /**
   * Computes the expectation of the parameters of a binary factor f
   * under the probabilities given by a unary factor g and joins the
   * result into a unary factor h.
   */
  template <typename T, typename Arg, typename Op>
  void join_expectation(array_factor<Arg, 1, T>& h,
                        const array_factor<Arg, 2, T>& f,
                        const array_factor<Arg, 1, T>& g,
                        Op op) {
    auto a = f.param().matrix(); // matrix
    auto b = g.param().matrix(); // vector
    auto c = h.param().matrix(); // vector
    if (f.x() == h.x() && f.y() == g.x()) {
      op(c.noalias(), a * b);
    } else if (f.x() == g.x() && f.y() == h.x()) {
      op(c.noalias(), a.transpose() * b);
    } else {
      throw std::invalid_argument(
        "array_factor join_expectation: unsupported arguments"
      );
    }
  }

  /**
   * Transforms and aggregates the parameter array of a binary factor along
   * dimension different from retain and stores the result to the specified
   * unary factor.
   */
  template <typename T, typename Arg, typename TransOp, typename AggOp>
  void transform_aggregate(const array_factor<Arg, 2, T>& f,
                           const array_domain<Arg, 1>& retain,
                           array_factor<Arg, 1, T>& h,
                           TransOp trans_op,
                           AggOp agg_op) {
    h.reset({retain});
    if (retain[0] == f.x()) {
      h.param() = agg_op(trans_op(f.param()).rowwise());
    } else if (retain[0] == f.y()) {
      h.param() = agg_op(trans_op(f.param()).colwise()).transpose();
    } else {
      throw std::invalid_argument(
        "array_factor: the retained variable not in the factor domain"
      );
    }
  }

  /**
   * Aggregates the parameter array of a binary factor along dimension
   * different from retain (if any) and stores the result to the specified
   * unary factor.
   */
  template <typename T, typename Arg, typename AggOp>
  void aggregate(const array_factor<Arg, 2, T>& f,
                 const array_domain<Arg, 1>& retain,
                 array_factor<Arg, 1, T>& h,
                 AggOp agg_op) {
    transform_aggregate(f, retain, h, identity(), agg_op);
  }

  /**
   * Aggregates the parameter array of a binary factor along dimension
   * different from retain (if any) and returns the result with given type.
   */
  template <typename Result, typename T, typename Arg, typename AggOp>
  Result aggregate(const array_factor<Arg, 2, T>& f,
                   const array_domain<Arg, 1>& retain,
                   AggOp agg_op) {
    Result result;
    aggregate(f, retain, result, agg_op);
    return result;
  }

  /**
   * Restricts a binary factor to an assignment and stores the result
   * to the given unary factor. All variables other than one must be
   * excluded, so that the result is exactly representable by
   * a unary factor.
   */
  template <typename T, typename Arg>
  void restrict_assign(const array_factor<Arg, 2, T>& f,
                       const uint_assignment<Arg>& a,
                       array_factor<Arg, 1, T>& result) {
    auto itx = a.find(f.x());
    auto ity = a.find(f.y());
    if (itx == a.end() && ity != a.end()) {
      result.reset({f.x()});
      result.param() = f.param().col(ity->second);
    } else if (itx != a.end() && ity == a.end()) {
      result.reset({f.y()});
      result.param() = f.param().row(itx->second).transpose();
    } else {
      throw std::invalid_argument(
        "array_factor: assignment must restrict all but one argument"
      );
    }
  }

  /**
   * Restricts a binary factor to an assignment, excluding the variables
   * in the unary factor result, and joins the restriction into result.
   */
  template <typename T, typename Arg, typename Op>
  void restrict_join(const array_factor<Arg, 2, T>& f,
                     const uint_assignment<Arg>& a,
                     array_factor<Arg, 1, T>& result,
                     Op op) {
    auto itx = result.x() != f.x() ? a.find(f.x()) : a.end();
    auto ity = result.x() != f.y() ? a.find(f.y()) : a.end();
    if (itx != a.end() && ity != a.end()) {
      op(result.param(), f.param()(itx->second, ity->second));
    } else if (itx == a.end() && f.x() == result.x()) {
      op(result.param(), f.param().col(ity->second));
    } else if (ity == a.end() && f.y() == result.x()) {
      op(result.param(), f.param().row(itx->second).transpose());
    } else if (itx == a.end() && ity == a.end()) {
      throw std::invalid_argument(
        "array_factor: restrict_join does not restrict anything"
      );
    } else {
      throw std::invalid_argument(
        "array_factor: restrict_join introduces an argument to the result"
      );
    }
  }

  /**
   * Transforms the parameters of two factors using a binary operation
   * and returns the result. The two factors must have the same domains.
   */
  template <typename Result,
            typename T, std::size_t N, typename Arg,
            typename Op>
  Result transform(const array_factor<Arg, N, T>& f,
                   const array_factor<Arg, N, T>& g,
                   Op op) {
    check_same_arguments(f, g);
    Result result(f.arguments());
    std::transform(f.begin(), f.end(), g.begin(), result.begin(), op);
    return result;
  }

  /**
   * Transforms the elements of a single factor using a unary operation
   * and accumulates the result using another operation.
   */
  template <typename T, std::size_t N, typename Arg,
            typename TransOp, typename AccuOp>
  T transform_accumulate(const array_factor<Arg, N, T>& f,
                         TransOp trans_op, AccuOp accu_op) {
    T result(0);
    for (const T& x : f) {
      result = accu_op(result, trans_op(x));
    }
    return result;
  }

  /**
   * Transforms the parameters of two factors using a binary operation
   * and accumulates the result using another operation.
   */
  template <typename T, std::size_t N, typename Arg,
            typename JoinOp, typename AggOp>
  T transform_accumulate(const array_factor<Arg, N, T>& f,
                         const array_factor<Arg, N, T>& g,
                         JoinOp join_op,
                         AggOp agg_op) {
    check_same_arguments(f, g);
    return std::inner_product(f.begin(), f.end(), g.begin(),
                              T(0), agg_op, join_op);
  }

} // namespace libgm

#endif
