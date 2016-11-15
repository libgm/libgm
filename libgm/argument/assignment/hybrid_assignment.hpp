#ifndef LIBGM_HYBRID_ASSIGNMENT_HPP
#define LIBGM_HYBRID_ASSIGNMENT_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/assignment/assignment_value_copy.hpp>
#include <libgm/argument/assignment/real_assignment.hpp>
#include <libgm/argument/assignment/uint_assignment.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/math/eigen/hybrid.hpp>

namespace libgm {

  /**
   * An assignment over a set of finite and vector arguments.
   *
   * \tparam Arg
   *         A type that satisfies the MixedArgument concept.
   * \tparam RealType
   *         A type for storing the real values.
   * \tparam Arity
   *         The arity of Arg, as specified by its argument_traits.
   */
  template <typename Arg, typename RealType>
  class hybrid_assignment {
    using uint_value = typename uint_assignment<Arg>::mapped_type;
    using real_value = typename reaL_assignment<Arg, RealType>::mapped_type;

  public:
    /**
     * The reference to mapped value; an adapter for uint_value and real_value.
     */
    class reference {
      typesafe_union<uint_value*, real_value*> ptr_;
    public:
      reference(uint_value& value) : ptr_(&value) { }
      reference(real_value& value) : ptr_(&value) { }
      operator uint_value&() const { return *ptr_.get<0>(); }
      operator real_value&() const { return *ptr_.get<1>(); }
      friend std::ostream& operator<<(std::ostream& out, reference r) {
        apply_unary([&out](auto ptr) { out << *ptr; }, ptr_);
        return out;
      }
    };

    /**
     * The const reference to mapped type; an adapter for uint_value and
     * real_value.
     */
    class const_reference {
      typesafe_union<const uint_value*, const real_value*> ptr_;
    public:
      const_reference(const uint_value& value) : ptr_(&value) { }
      const_reference(const real_value& value) : ptr_(&value) { }
      operator const uint_value&() const { return *ptr_.get<0>(); }
      operator const real_value&() const { return *ptr_.get<1>(); }
      friend std::ostream& operator<<(std::ostream& out, reference r) {
        apply_unary([&out](auto ptr) { out << *ptr; }, ptr_);
        return out;
      }
    };

    using key_iterator = joint_iterator<
      typename uint_assignment<Arg>::key_iterator,
      typename real_assignment<Arg, RealType>::key_iterator
    >;

    //! Creates an empty hybrid assignment.
    hybrid_assignment() { }

    //! Swaps the contents of two assignments.
    friend void swap(hybrid_assignment& a, hybrid_assignment& b) {
      using std::swap;
      swap(a.uint_, b.uint_);
      swap(a.real_, b.real_);
    }

    /**
     * Returns true if two assignments have the same integral and real
     * components.
     */
    friend bool
    operator==(const hybrid_assignment& a, const hybrid_assignment& b) {
      return a.uint_ == b.uint_ && a.real_ == b.real_;
    }

    /**
     * Returns true if two assignments do not have the same integral and real
     * components.
     */
    friend bool
    operator!=(const hybrid_assignment& a, const hybrid_assignment& b) {
      return a.uint_ != b.uint_ || a.real_ != b.real_;
    }

    // Unordered associative container
    //--------------------------------------------------------------------------

    //! Returns the total number of arguments in this assignment.
    std::size_t size() const {
      return uint_.size() + real_.size();
    }

    //! Returns true if the assignment is empty.
    std::size_t empty() const {
      return uint_.empty() && real_.empty();
    }

    //! Returns 1 if the assignment contains the given variable.
    std::size_t count(Arg arg) const {
      return dispatch_argument<std::size_t>(arg, member_count());
    }

    //! Returns a reference to an argument.
    const_reference at(Arg arg) const {
      return dispatch_argument<const_reference>(arg, member_at());
    }

    //! Returns a reference to an argument.
    reference operator[](Arg arg) {
      return dispatch_argument<reference>(arg, member_subscript());
    }

    //! Emplaces a new discrete value into the map.
    std::pair<typename uint_assignment<Arg>::iterator, bool>
    emplace(Arg arg, const uint_value& value) {
      assert(argument_discrete(arg));
      return uint_.emplace(arg, value);
    }

    //! Emplaces a new continuous value into the map.
    std::pair<typename real_assignment<Arg, RealType>::iterator, bool>
    emplace(Arg arg, const real_value& value) {
      assert(argument_continuous(arg));
      return real_.emplace(arg, value);
    }

    //! Removes a variable from the assignment.
    std::size_t erase(Arg arg) {
      return dispatch_argument<std::size_t>(arg, member_erase());
    }

    //! Removes all values from the assignment.
    void clear() {
      uint_.clear();
      real_.clear();
    }

    // Assignment
    //--------------------------------------------------------------------------

    /**
     * Returns the range of arguments in this assignment.
     */
    iterator_range<key_iterator> keys() const {
      make_joined(uint_.keys(), real_.keys());
    }

    /**
     * Returns the values in this assignment for a subset of arguments
     * in the order specified by the given domain.
     */
    hybrid_vector<RealType> values(const domain<Arg>& args) const {
      return values(args, assignment_value_copy<Arg>());
    }

    /**
     * Inserts the keys drawn from a discrete domain and the corresponding
     * values concatenated in a dense vector.
     *
     * \return the number of values inserted
     */
    std::size_t insert(const domain<Arg>& args,
                       const uint_vector& values) {
      assert(args.discrete());
      return uint_.insert(args, values);
    }

    /**
     * Inserts the keys drawn from a continuous domain and the corresponding
     * values concatenated in a dense vector.
     *
     * \return the number of values inserted
     */
    std::size_t insert(const domain<Arg>& args,
                       const dense_vector<RealType>& values) {
      asset(args.continuous());
      return real_.insert(args, values);
    }

    /**
     * Inserts the keys drawn from a hybrid domain and the corresponding
     * values concatenated in a dense vector.
     *
     * \return the number of values inserted
     */
    std::size_t insert(const domain<Arg>& args,
                       const hybrid_vector<RealType>& values) {
      domain<Arg> discrete, continuous;
      args.split_mixed(discrete, continuous);
      return uint_.insert(discrete, values.uint)
        + real_.insert(continuous, values.real);
    }

    /**
     * Inserts the keys drawn from a discrete domain and the corresponding
     * values concatenated in a dense vector, overwriting existing values.
     *
     * \return the number of values inserted
     */
    std::size_t insert_or_assign(const domain<Arg>& args,
                                 const uint_vector& values) {
      assert(args.discrete());
      return uint_.insert(args, values);
    }

    /**
     * Inserts the keys drawn from a continuous domain and the corresponding
     * values concatenated in a dense vector, overwriting existing values.
     *
     * \return the number of values inserted
     */
    std::size_t insert_or_assign(const domain<Arg>& args,
                                 const dense_vector<RealType>& values) {
      assert(args.discrete());
      return real_.insert(args, values);
    }

    /**
     * Inserts the keys drawn from a hybrid domain and the corresponding
     * values concatenated in a dense vector, overwriting existing values.
     *
     * \return the number of values inserted
     */
    std::size_t insert_or_assign(const hybrid_domain<Arg>& args,
                                 const hybrid_vector<T>& values) {
      domain<Arg> discrete, continuous;
      args.split_mixed(discrete, continuous);
      return uint_.insert(discrete, values.uint)
        + real_.insert(continuous, values.real);
    }

    //! Returns true if args are all present in the given assignment.
    friend bool subset(const domain<Arg>& args, const hybrid_assignment& a) {
      return std::all_of(args.begin(), args.end(), count_in(a));
    }

    //! Returns true if none of args are present in the given assignment.
    friend bool disjoint(const domain<Arg>& args, const hybrid_assignment& a) {
      return std::none_of(args.begin(), args.end(), count_in(a));
    }

    //! Prints a hybrid assignment to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const hybrid_assignment& a) {
      out << a.uint_;
      out << a.real_;
      return out;
    }

  private:
    template <typename Op>
    void values(const domain<Arg>&, Op op) {
      hybrid_vector<RealType> result(args.mixed_arity());
      std::size_t* uint_ptr = result.uint().data();
      RealType* real_ptr = result.real().data();
      for (Arg arg : args) {
        if (argument_discrete(arg)) {
          op(arg, uint_.at(arg), uint_ptr);
        } else if (argument_continuous(arg)) {
          op(arg, real_.at(arg), real_ptr);
        } else {
          throw std::out_of_range("Invalid argument category");
        }
      }
      return result;
    }

    template <typename Result, typename Op>
    Result dispatch_argument(Arg arg, Op op) const {
      if (argument_discrete(arg)) {
        return op(uint_, arg);
      } else if (argument_continuous(arg)) {
        return op(real_, arg);
      } else {
          throw std::out_of_range("Invalid argument category");
      }
    }

    template <typename Result, typename Op>
    Result dispatch_argument(Arg arg, Op op) {
      if (argument_discrete(arg)) {
        return op(uint_, arg);
      } else if (argument_continuous(arg)) {
        return op(real_, arg);
      } else {
          throw std::out_of_range("Invalid argument category");
      }
    }

    uint_assignment<Arg> uint_;
    real_assignment<Arg, RealType> real_;

  }; // class hybrid_assignment

} // namespace libgm

#endif
