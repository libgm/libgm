#ifndef LIBGM_TABLE_FACTOR_HPP
#define LIBGM_TABLE_FACTOR_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/finite_assignment.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factor/base/factor.hpp>
#include <libgm/serialization/serialize.hpp>

#include <algorithm>

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
  template <typename T, typename Var>
  class table_factor : public factor {
  public:
    // Range types
    typedef T*       iterator;
    typedef const T* const_iterator;
    typedef T        value_type;

    // Finite domain
    typedef basic_domain<Var> domain_type;
    typedef finite_assignment<Var> assignment_type;

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
        finite_index shape(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
          shape[i] = args[i].size();
        }
        param_.reset(shape);
      }
    }
    
    // Accessors
    //==========================================================================

    //! Returns the finite arguments of this factor.
    const domain_type& finite_args() const {
      return finite_args_;
    }

    //! Returns the number of arguments of this factor.
    size_t arity() const {
      return param_.arity();
    }

    //! Returns the total number of elements of the factor.
    size_t size() const {
      return param_.size();
    }

    //! Returns true if the factor has an empty table (equivalent to size() == 0).
    bool empty() const {
      return param_.empty();
    }

    //! Returns the pointer to the first element or NULL if the factor is empty.
    T* begin() {
      return param_.begin();
    }

    //! Returns the pointer to the first element or NULL if the factor is empty.
    const T* begin() const {
      return param_.begin();
    }

    //! Returns the pointer to past the last element or NULL if the factor is empty.
    T* end() {
      return param_.end();
    }

    //! Returns the pointer to past the last element or NULL if the factor is empty.
    const T* end() const {
      return param_.end();
    }

    //! Provides mutable access to the parameter with the given linear index.
    T& operator[](size_t i) {
      return param_[i];
    }
    
    //! Returns the parameter with the given linear index.
    const T& operator[](size_t i) const {
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
    T& param(const finite_index& index) {
      return param_(index);
    }

    //! Returns the parameter for the given index.
    const T& param(const finite_index& index) const {
      return param_(index);
    }

    // Indexing
    //==========================================================================
    
    /**
     * Converts the index to this factor's arguments to an assignment.
     * The index may be merely a prefix, and the output assignment is not cleared.
     */
    void assignment(const finite_index& index, assignment_type& a) const {
      assert(index.size() <= finite_args_.size());
      for(size_t i = 0; i < index.size(); i++) {
        a[finite_args_[i]] = index[i];
      }
    }

    /**
     * Returns the linear index corresponding to the given assignment.
     * If strict, each argument of this factor must be present in the
     * assignment. If not strict, the missing arguments will be associated
     * with value 0.
     */
    size_t index(const assignment_type& a, bool strict = true) const {
      size_t result = 0;
      for (size_t i = 0; i < arity(); ++i) {
        Var v = finite_args_[i];
        auto it = a.find(v);
        if (it != a.end()) {
          result += param_.offset().multiplier(i) * it->second;
        } else if (strict) {
          throw std::invalid_argument(
            "The assignment does not contain the variable " + v.str()
          );
        }
      }
      return result;
    }

    /**
     * Returns the mapping of this factor's arguments to the given var vector.
     * If strict, all the arguments must be present in the given vector.
     * If not strict, the missing variables will be assigned a NA value,
     * std::numeric_limits<size_t>::max().
     *
     * When using this function in factor operations, always call the
     * dim_map function on the factor whose elements will be iterated
     * over in a non-linear fashion. The vector vars are the arguments
     * of the table that is iterated over in a linear fashion.
     */
    finite_index dim_map(const domain_type& vars, bool strict = true) const {
      finite_index map(arity(), std::numeric_limits<size_t>::max());
      for(size_t i = 0; i < map.size(); i++) {
        auto it = std::find(vars.begin(), vars.end(), finite_args_[i]);
        if (it != vars.end()) {
          map[i] = it - vars.begin();
        } else if (strict) {
          throw std::invalid_argument("Missing variable " + finite_args_[i].str());
        }
      }
      return map;
    }

    /**
     * Substitutes this factor's arguments according to the given map,
     * in place.
     */
    void subst_args(const std::unordered_map<Var, Var>& var_map) {
      for (Var& var : finite_args_) {
        Var new_var = var_map.at(var);
        if (!compatible(var, new_var)) {
          throw std::invalid_argument(
            "subst_args: " + var.str() + " and " + new_var.str() +
            " are not compatible"
          );
        }
        var = new_var;
      }
    }

    /**
     * Checks if the shape of the table matches this factor's argument vector.
     * \throw std::runtime_error if some of the dimensions do not match
     */
    void check_param() const {
      if (param_.arity() != finite_args_.size()) {
        throw std::runtime_error("Invalid table arity");
      }
      for (size_t i = 0; i < finite_args_.size(); ++i) {
        if (param_.size(i) != finite_args_[i].size()) {
          throw std::runtime_error("Invalid table shape");
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
        finite_index f_map = f.dim_map(finite_args_);
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
     * other than those for the retained variables using the given binary
     * operation and stores the result to the specified factor. This function
     * avoids reallocation if the target argument vector has not changed.
     */
    template <typename Op>
    void aggregate(const domain_type& retain, T init, Op op,
                   table_factor& result) const {
      result.reset(retain);
      result.param_.fill(init);
      finite_index result_map = result.dim_map(finite_args_);
      table_aggregate<T, T, Op>(result.param_, param_, result_map, op)();
    }

    /**
     * Restricts this factor to an assignment and stores the result to the
     * given table. This function is protected, because the result is not
     * strongly typed, i.e., we could accidentally restrict a probability_table
     * and store the result in a canonical_table or vice versa.
     */
    void restrict(const assignment_type& a, table_factor& result) const {
      domain_type new_vars;
      for (Var v : finite_args_) {
        if (!a.count(v)) { new_vars.push_back(v); }
      }
      result.reset(new_vars);
      if (prefix(result.finite_args_, finite_args_)) {
        result.param_.restrict(param_, index(a, false));
      } else {
        finite_index map = dim_map(result.finite_args_, false);
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
  template <typename Result, typename T, typename Var, typename Op>
  Result join(const table_factor<T, Var>& f,
              const table_factor<T, Var>& g,
              Op op) {
    if (f.finite_args() == g.finite_args()) {
      Result result(f.finite_args());
      std::transform(f.begin(), f.end(), g.begin(), result.begin(), op);
      return result;
    } else {
      Result result(f.finite_args() | g.finite_args());
      finite_index f_map = f.dim_map(result.finite_args());
      finite_index g_map = g.dim_map(result.finite_args());
      table_join<T, T, Op>(result.param(), f.param(), g.param(),
                           f_map, g_map, op)();
      return result;
    }
  }

  /**
   * Transforms the parameters of the factor with a unary operation
   * and returns the result.
   */
  template <typename Result, typename T, typename Var, typename Op>
  Result transform(const table_factor<T, Var>& f, Op op) {
    Result result(f.finite_args());
    std::transform(f.begin(), f.end(), result.begin(), op);
    return result;
  }

  /**
   * Transforms the parameters of two factors using a binary operation
   * and returns the result. The two factors must have the same argument vectors.
   */
  template <typename Result, typename T, typename Var, typename Op>
  Result transform(const table_factor<T, Var>& f,
                   const table_factor<T, Var>& g,
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
  template <typename T, typename Var, typename JoinOp, typename AggOp>
  T transform_accumulate(const table_factor<T, Var>& f,
                         const table_factor<T, Var>& g,
                         JoinOp join_op,
                         AggOp agg_op) {
    assert(f.finite_args() == g.finite_args());
    return std::inner_product(f.begin(), f.end(), g.begin(), T(0), agg_op, join_op);
  }

} // namespace libgm

#endif
