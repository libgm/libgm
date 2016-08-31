#ifndef LIBGM_TABLE_FACTOR_HPP
#define LIBGM_TABLE_FACTOR_HPP

#include <libgm/macros.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factor/base/factor.hpp>
#include <libgm/serialization/serialize.hpp>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace libgm {

  /**
   * A base class for table-like factors. Stores the finite arguments and
   * stores the factor parameters in a table. Provides indexing functions
   * and standard join/aggregate/restrict functions on the factors. This
   * class does not model the Factor concept, but it does model the Range
   * concept.
   *
   * \tparam T the type of parameters stored in the table.
   * \see canonical_table, probability_table, hybrid
   */
  template <typename Arg, typename T>
  class table_factor : public factor {
    static_assert(is_discrete<Arg>::value,
                  "Table factors require Arg to be discrete");

  public:
    // Range types
    typedef T*       iterator;
    typedef const T* const_iterator;
    typedef T        value_type;

    // Arguments
    typedef argument_traits<Arg> arg_traits;
    typedef domain<Arg>          domain_type;
    typedef uint_assignment<Arg> assignment_type;

    // Constructors
    //==========================================================================

    //! Default constructor. Creates an empty table factor.
    table_factor() { }

    //! Creates a factor with the given finite arguments and parameters.
    table_factor(const domain_type& args, const table<T>& param)
      : finite_args_(args),
        param_(param) {
      check_param();
    }

    //! Creates a factor with the given finite arguments and parameters.
    table_factor(const domain_type& args, table<T>&& param)
      : finite_args_(args),
        param_(std::move(param)) {
      check_param();
    }

    // Serialization and initialization
    //==========================================================================

    //! Serializes members.
    void save(oarchive& ar) const {
      ar << finite_args_ << param_;
    }

    //! Deserializes members.
    void load(iarchive& ar) {
      ar >> finite_args_ >> param_;
      check_param();
    }

    /**
     * Resets the content of this factor to the given sequence of arguments.
     * If the table size changes, the table elements become invalidated.
     */
    void reset(const domain_type& args = domain_type()) {
      if (empty() || finite_args_ != args) {
        finite_args_ = args;
        param_.reset(param_shape(args));
      }
    }

    /**
     * Resets the content of this factor and fills the table with the given
     * value.
     */
    void reset(const domain_type& args, T value) {
      reset(args);
      param_.fill(value);
    }

    // Accessors
    //==========================================================================

    //! Returns the finite arguments of this factor.
    const domain_type& finite_args() const {
      return finite_args_;
    }

    //! Returns the number of arguments of this factor.
    std::size_t arity() const {
      return finite_args_.size();
    }

    //! Returns the total number of elements of the factor.
    std::size_t size() const {
      return param_.size();
    }

    //! Returns true if the factor has an empty table (same as size() == 0).
    bool empty() const {
      return param_.empty();
    }

    //! Returns the pointer to the first element or nullptr if the factor is empty.
    T* begin() {
      return param_.begin();
    }

    //! Returns the pointer to the first element or nullptr if the factor is empty.
    const T* begin() const {
      return param_.begin();
    }

    //! Returns the pointer past the last element or nullptr if the factor is empty.
    T* end() {
      return param_.end();
    }

    //! Returns the pointer past the last element or nullptr if the factor is empty.
    const T* end() const {
      return param_.end();
    }

    //! Provides mutable access to the parameter with the given linear index.
    T& operator[](std::size_t i) {
      return param_[i];
    }

    //! Returns the parameter with the given linear index.
    const T& operator[](std::size_t i) const {
      return param_[i];
    }

    //! Provides mutable access to the parameter table of this factor.
    table<T>& param() {
      return param_;
    }

    //! Returns the parameter table of this factor.
    const table<T>& param() const {
      return param_;
    }

    //! Provides mutable access to the paramater for the given assignment.
    T& param(const assignment_type& a) {
      return param_[index(a)];
    }

    //! Returns the parameter for the given assignment.
    const T& param(const assignment_type& a) const {
      return param_[index(a)];
    }

    //! Provides mutable access to the parameter for the given index.
    T& param(const uint_vector& index) {
      return param_(index);
    }

    //! Returns the parameter for the given index.
    const T& param(const uint_vector& index) const {
      return param_(index);
    }

    // Indexing
    //==========================================================================

    /**
     * Returns the shape of the table for the given domain. The resulting vector
     * contains the number of values for each argument (when Arg is univariate)
     * or a sequence of value couts for each argument (when Arg is multivariate)
     * in the order specified by args.
     *
     * \fn static uint_vector param_shape(const domain_type& args)
     */
    LIBGM_ENABLE_IF_STATIC(A = Arg, is_univariate<A>::value, uint_vector)
    param_shape(const domain_type& args) {
      uint_vector shape(args.size());
      for (std::size_t i = 0; i < args.size(); ++i) {
        shape[i] = arg_traits::num_values(args[i]);
      }
      return shape;
    }

    LIBGM_ENABLE_IF_STATIC(A = Arg, is_multivariate<A>::value, uint_vector)
    param_shape(const domain_type& args) {
      uint_vector shape(args.num_dimensions());
      uint_vector::iterator it = shape.begin();
      for (Arg arg : args) {
        for (std::size_t i = 0; i < arg_traits::num_dimensions(arg); ++i) {
          *it++ = arg_traits::num_values(arg, i);
        }
      }
      return shape;
    }

    /**
     * Returns the linear index of the cell corresponding to the given
     * assignment. If strict is true, each argument of this factor must be
     * present in the assignment. If strict is false, the missing arguments
     * are assumed to be 0.
     *
     * \fn std::size_t index(const assignment_type& a, bool strict = true) const
     */
    LIBGM_ENABLE_IF_OLD(A = Arg, is_univariate<A>::value, std::size_t)
    index(const assignment_type& a, bool strict = true) const {
      std::size_t result = 0;
      for (std::size_t i = 0; i < finite_args_.size(); ++i) {
        Arg v = finite_args_[i];
        auto it = a.find(v);
        if (it != a.end()) {
          result += param_.offset().multiplier(i) * it->second;
        } else if (strict) {
          std::ostringstream out;
          out << "The assignment does not contain the argument ";
          arg_traits::print(out, v);
          throw std::invalid_argument(out.str());
        }
      }
      return result;
    }

    LIBGM_ENABLE_IF_OLD(A = Arg, is_multivariate<A>::value, std::size_t)
    index(const assignment_type& a, bool strict = true) const {
      std::size_t result = 0;
      std::size_t i = 0;
      for (Arg arg : finite_args_) {
        std::size_t n = arg_traits::num_dimensions(arg);
        auto it = a.find(arg);
        if (it != a.end()) {
          assert(it->second.size() == n);
          for (std::size_t val : it->second) {
            result += param_.offset().multiplier(i++) * val;
          }
        } else if (!strict) {
          i += n;
        } else {
          std::ostringstream out;
          out << "The assignment does not contain the argument ";
          arg_traits::print(out, arg);
          throw std::invalid_argument(out.str());
        }
      }
      return result;
    }

    /**
     * Returns the mapping of this factor's arguments to the given domain.
     * Resulting vector contains, for each dimension of this factor's table,
     * the dimension of the table corresponding to args. If strict is true,
     * all arguments of this factor must be present in the specified domain.
     * If strict is false, the missing arguments will be assigned a NA value,
     * std::numeric_limits<std::size_t>::max().
     *
     * When using this function in factor operations, always call the
     * dim_map function on the factor whose elements will be iterated
     * over in a non-linear fashion. The specified args are the arguments
     * of the table that is iterated over in a linear fashion.
     *
     * \fn uint_vector dim_map(const domain_type& args, bool strict=true) const
     */
    LIBGM_ENABLE_IF_OLD(A = Arg, is_univariate<A>::value, uint_vector)
    dim_map(const domain_type& args, bool strict = true) const {
      uint_vector map(param_.arity(), std::numeric_limits<std::size_t>::max());
      for(std::size_t i = 0; i < map.size(); i++) {
        auto it = std::find(args.begin(), args.end(), finite_args_[i]);
        if (it != args.end()) {
          map[i] = it - args.begin();
        } else if (strict) {
          std::ostringstream out;
          out << "table factor: missing argument ";
          arg_traits::print(out, finite_args_[i]);
          throw std::invalid_argument(out.str());
        }
      }
      return map;
    }

    LIBGM_ENABLE_IF_OLD(A = Arg, is_multivariate<A>::value, uint_vector)
    dim_map(const domain_type& args, bool strict = true) const {
      // compute the first dimension of each argument in args
      std::vector<std::size_t> dim(args.size());
      for (std::size_t i = 1; i < args.size(); ++i) {
        dim[i] = dim[i-1] + arg_traits::num_dimensions(args[i-1]);
      }

      // extract the dimensions for the arguments in this factor
      uint_vector map(param_.arity(), std::numeric_limits<std::size_t>::max());
      uint_vector::iterator dest = map.begin();
      for (Arg arg : finite_args_) {
        auto it = std::find(args.begin(), args.end(), arg);
        std::size_t n = arg_traits::num_dimensions(arg);
        if (it != args.end()) {
          std::iota(dest, dest + n, dim[it - args.begin()]);
        } else if (strict) {
          std::ostringstream out;
          out << "table factor: missing argument ";
          arg_traits::print(out, arg);
          throw std::invalid_argument(out.str());
        }
        dest += n;
      }
      return map;
    }

    /**
     * Substitutes this factor's arguments according to the given map,
     * in place.
     */
    void subst_args(const std::unordered_map<Arg, Arg>& arg_map) {
      for (Arg& arg : finite_args_) {
        Arg new_arg = arg_map.at(arg);
        if (!arg_traits::compatible(arg, new_arg)) {
          std::ostringstream out;
          out << "subst_args: "; arg_traits::print(out, arg);
          out << " and "; arg_traits::print(out, new_arg);
          out << " are not compatible";
          throw std::invalid_argument(out.str());
        }
        arg = new_arg;
      }
    }

    /**
     * Checks if the shape of the table matches this factor's argument vector.
     * \throw std::runtime_error if some of the dimensions do not match
     */
    LIBGM_ENABLE_IF_OLD(A = Arg, is_univariate<A>::value, void)
    check_param() const {
      if (param_.arity() != finite_args_.num_dimensions()) {
        throw std::runtime_error("Invalid table arity");
      }
      for (std::size_t i = 0; i < finite_args_.size(); ++i) {
        if (param_.size(i) != arg_traits::num_values(finite_args_[i])) {
          throw std::runtime_error("Invalid table shape");
        }
      }
    }

    LIBGM_ENABLE_IF_OLD(A = Arg, is_multivariate<A>::value, void)
    check_param() const {
      if (param_.arity() != finite_args_.num_dimensions()) {
        throw std::runtime_error("Invalid table arity");
      }
      std::size_t dim = 0;
      for (Arg arg : finite_args_) {
        for (std::size_t i = 0; i < arg_traits::num_dimensions(arg); ++i) {
          if (param_.size(dim) != arg_traits::num_values(arg, i)) {
            throw std::runtime_error("Invalid table shape");
          }
        }
      }
    }

    // Implementations of common factor operations
    //========================================================================
  protected:
    /**
     * Joins this factor in place with f using the given binary operation.
     * f must not introduce any new arguments into this factor.
     */
    template <typename Op>
    void join_inplace(const table_factor& f, Op op) {
      if (finite_args_ == f.finite_args_) {
        param_.transform(f.param_, op);
      } else {
        uint_vector f_map = f.dim_map(finite_args_);
        table_join_inplace<T, T, Op>(param_, f.param_, f_map, op)();
      }
    }

    /**
     * Performs a binary transform of this factor and another one in place.
     * The two factors must have the same argument vectors.
     */
    template <typename Op>
    void transform_inplace(const table_factor& f, Op op) {
      assert(finite_args_ == f.finite_args_);
      param_.transform(f.param_, op);
    }

    /**
     * Aggregates the parameter table of this factor along all dimensions
     * other than those for the retained arguments using the given binary
     * operation and stores the result to the specified factor. This function
     * avoids reallocation if the target argument vector has not changed.
     */
    template <typename Op>
    void aggregate(const domain_type& retain, T init, Op op,
                   table_factor& result) const {
      result.reset(retain);
      result.param_.fill(init);
      uint_vector result_map = result.dim_map(finite_args_);
      table_aggregate<T, T, Op>(result.param_, param_, result_map, op)();
    }

    /**
     * Restricts this factor to an assignment and stores the result to the
     * given table. This function is protected, because the result is not
     * strongly typed, i.e., we could accidentally restrict a probability_table
     * and store the result in a canonical_table or vice versa.
     */
    void restrict(const assignment_type& a, table_factor& result) const {
      domain_type new_args;
      for (Arg v : finite_args_) {
        if (!a.count(v)) { new_args.push_back(v); }
      }
      result.reset(new_args);
      if (finite_args_.prefix(result.finite_args_)) {
        result.param_.restrict(param_, index(a, false));
      } else {
        uint_vector map = dim_map(result.finite_args_, false);
        table_restrict<T>(result.param_, param_, map, index(a, false))();
      }
    }

    // Protected members
    //========================================================================

    /**
     * Implementation of the swap function (not type-safe).
     */
    void base_swap(table_factor& other) {
      if (this != &other) {
        using std::swap;
        swap(finite_args_, other.finite_args_);
        swap(param_, other.param_);
      }
    }

    //! The sequence of arguments of this factor
    domain_type finite_args_;

    //! The canonical parameters of this factor
    table<T> param_;

  }; // class table_factor


  // Utility functions
  //========================================================================

  /**
   * Joins the parameter tables of two factors using a binary operation.
   * The resulting factor contains the union of f's and g's argument sets.
   */
  template <typename Result, typename Arg, typename T, typename Op>
  Result join(const table_factor<Arg, T>& f,
              const table_factor<Arg, T>& g,
              Op op) {
    if (f.finite_args() == g.finite_args()) {
      Result result(f.finite_args());
      std::transform(f.begin(), f.end(), g.begin(), result.begin(), op);
      return result;
    } else {
      Result result(f.finite_args() + g.finite_args());
      uint_vector f_map = f.dim_map(result.finite_args());
      uint_vector g_map = g.dim_map(result.finite_args());
      table_join<T, T, Op>(result.param(), f.param(), g.param(),
                           f_map, g_map, op)();
      return result;
    }
  }

  /**
   * Transforms the parameters of the factor with a unary operation
   * and returns the result.
   */
  template <typename Result, typename Arg, typename T, typename Op>
  Result transform(const table_factor<Arg, T>& f, Op op) {
    Result result(f.finite_args());
    std::transform(f.begin(), f.end(), result.begin(), op);
    return result;
  }

  /**
   * Transforms the parameters of two factors using a binary operation
   * and returns the result. The two factors must have the same arguments
   */
  template <typename Result, typename Arg, typename T, typename Op>
  Result transform(const table_factor<Arg, T>& f,
                   const table_factor<Arg, T>& g,
                   Op op) {
    assert(f.finite_args() == g.finite_args());
    Result result(f.finite_args());
    std::transform(f.begin(), f.end(), g.begin(), result.begin(), op);
    return result;
  }

  /**
   * Transforms the parameters of two factors using a binary operation
   * and accumulates the result using another operation.
   */
  template <typename Arg, typename T, typename JoinOp, typename AggOp>
  T transform_accumulate(const table_factor<Arg, T>& f,
                         const table_factor<Arg, T>& g,
                         JoinOp join_op,
                         AggOp agg_op) {
    assert(f.finite_args() == g.finite_args());
    return std::inner_product(f.begin(), f.end(), g.begin(),
                              T(0), agg_op, join_op);
  }

} // namespace libgm

#endif
