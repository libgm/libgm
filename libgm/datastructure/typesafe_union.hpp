#ifndef LIBGM_TYPESAFE_UNION_HPP
#define LIBGM_TYPESAFE_UNION_HPP

#include <libgm/enable_if.hpp>
#include <libgm/functional/comparison.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/functional/stream.hpp>
#include <libgm/functional/utility.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>
#include <libgm/traits/count_types.hpp>
#include <libgm/traits/find_type.hpp>
#include <libgm/traits/static_max.hpp>
#include <libgm/traits/nth_type.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <type_traits>

namespace libgm {

  /**
   * A class that represents a union of types. An object of this class behaves
   * like boost::variant, except that its types are required to be trivially
   * copyable. Furthermore, unlike boost::variant, a typesafe_union may be
   * empty (i.e., not be associated with any type). These assumptions simplify
   * the implementation and avoid heap allocations.
   */
  template <typename... Types>
  class typesafe_union {
#ifndef __GNUC__
    static_assert(all_of_types<std::is_trivially_copyable, Types...>::value,
                  "The union types must be all trivially copyable");
#endif
    static_assert(sizeof...(Types) > 0,
                  "Empty typesafe_union is not allowed");
    static_assert(sizeof...(Types) < 128,
                  "A typesafe_union can consist of at most 127 types");

    // Constructors and initialization
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty union.
    typesafe_union()
      : which_(-1) { }

    //! Constructs union from a member that is convertible to one of Types.
    LIBGM_ENABLE_IF_D((count_convertible<T, Types...>::value) == 1, typename T)
    typesafe_union(T&& field) {
      *this = std::forward<T>(field);
    }

    //! Returns a special "deleted" union, used in some datastructures.
    static typesafe_union deleted() {
      typesafe_union result;
      result.which_ = -2;
      return result;
    }

    //! Assigns union to a member that is convertible to one of Types.
    LIBGM_ENABLE_IF_D((count_convertible<T, Types...>::value) == 1, typename T)
    typesafe_union& operator=(T&& field) {
      constexpr std::size_t index = find_convertible<T, Types...>::value;
      which_ = index;
      get<index>() = std::forward<T>(field);
      return *this;
    }

    //! Swaps the content of two unions.
    friend void swap(typesafe_union& x, typesafe_union& y) {
      std::swap(x, y);
    }

    // Serialization
    //--------------------------------------------------------------------------

    //! Serializes a union to an archive.
    void save(oarchive& ar) const {
      ar.serialize_char(which_);
      ar.serialize_buf(data(), length);
    }

    //! Deserializes a union from an archive.
    void load(iarchive& ar) {
      which_ = ar.deserialize_char();
      ar.deserialize_buf(data(), length);
    }

    // Queries
    //--------------------------------------------------------------------------

    //! Returns the index of the assigned field (-1 if empty, -2 if deleted).
    int which() const {
      return which_;
    }

    //! Returns true if the union is empty.
    bool empty() const {
      return which_ == -1;
    }

    //! Returns the pointer to the underlying buffer.
    void* data() {
      return data_;
    }

    //! Returns the pointer to the underlying buffer.
    const void* data() const {
      return data_;
    }

    //! Returns the value in the union with the given index.
    template <std::size_t I>
    typename nth_type<I, Types...>::type& get() {
      assert(which() == I);
      return *static_cast<typename nth_type<I, Types...>::type*>(data());
    }

    //! Returns the value in the union with the given index.
    template <std::size_t I>
    const typename nth_type<I, Types...>::type& get() const {
      assert(which() == I);
      return *static_cast<const typename nth_type<I, Types...>::type*>(data());
    }

    //! Returns the value in the union with the given type.
    LIBGM_ENABLE_IF_D((count_same<T, Types...>::value) == 1, typename T)
    T& get() {
      assert(which() == (find_same<T, Types...>::value));
      return *static_cast<T*>(data());
    }

    //! Returns the value in the union with the given type.
    LIBGM_ENABLE_IF_D((count_same<T, Types...>::value) == 1, typename T)
    const T& get() const {
      assert(which() == (find_same<T, Types...>::value));
      return *static_cast<const T*>(data());
    }

    //! Returns the hash value of the union.
    friend std::size_t hash_value(const typesafe_union& u) {
      std::size_t seed = apply_unary(invoke_hash(), u, std::size_t(0));
      libgm::hash_combine(seed, u.which());
      return seed;
    }

    // Comparisons
    //--------------------------------------------------------------------------

    //! Returns true if two unions have the same type and values.
    friend bool operator==(const typesafe_union& x,
                           const typesafe_union& y) {
      return x.which() == y.which() &&
        apply_binary(equal_to<>(), x, y, true);
    }

    //! Returns true if two unions do not have the same type or values.
    friend bool operator!=(const typesafe_union& x,
                           const typesafe_union& y) {
      return x.which() != y.which() ||
        apply_binary(not_equal_to<>(), x, y, false);
    }

    //! Returns true if x is less than y in the lexicographic ordering.
    friend bool operator<(const typesafe_union& x,
                          const typesafe_union& y) {
      return (x.which() < y.which()) ||
        (x.which() == y.which() && apply_binary(less<>(), x, y, false));
    }

    //! Returns true if x <= y in the lexicographic ordering.
    friend bool operator<=(const typesafe_union& x,
                           const typesafe_union& y) {
      return (x.which() < y.which()) ||
        (x.which() == y.which() && apply_binary(less_equal<>(), x, y, true));
    }

    //! Returns true if x if is greater than y in the lexicographic ordering.
    friend bool operator>(const typesafe_union& x,
                          const typesafe_union& y) {
      return (x.which() > y.which()) ||
        (x.which() == y.which() && apply_binary(greater<>(), x, y, false));
    }

    //! Returns true if x >= y in the lexicographic ordering.
    friend bool operator>=(const typesafe_union& x,
                           const typesafe_union& y) {
      return (x.which() > y.which()) ||
        (x.which() == y.which() && apply_binary(greater_equal<>(), x, y, true));
    }

    //! Prints a union to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const typesafe_union& u){
      switch (u.which()) {
      case -1: out << "null"; break;
      case -2: out << "deleted"; break;
      default: apply_unary(stream_out<std::ostream>(out), u); break;
      }
      return out;
    }

    // Data members
    //--------------------------------------------------------------------------
  private:
    static constexpr std::size_t length = static_max<sizeof(Types)...>::value;

    //! The underlying storage.
    char data_[length];

    //! The indicator of the stored type.
    int8_t which_;

  }; // class typesafe_union


  // Access functions
  //----------------------------------------------------------------------------

  /**
   * Helper function that invokes a function on a set of arguments
   * passed as void pointers.
   */
  template<typename Result, typename Function, typename T, typename... VoidPtrs>
  std::enable_if_t<!std::is_same<Result, std::nullptr_t>::value, Result>
  casting_caller(Function f, VoidPtrs... ptrs) {
    return static_cast<Result>(f(*static_cast<T*>(ptrs)...));
  }

  /**
   * Helper function that invokes a function on a set of arguments
   * passed as void pointers.
   */
  template<typename Result, typename Function, typename T, typename... VoidPtrs>
  std::enable_if_t<std::is_same<Result, std::nullptr_t>::value, Result>
  casting_caller(Function f, VoidPtrs... ptrs) {
    f(*static_cast<T*>(ptrs)...);
    return nullptr;
  }

  /**
   * Applies a unary function to a non-empty union and returns its result.
   * When the provided union is empty, returns empty_result.
   *
   * \tparam Result the result returned by the function
   * \tparam UnaryFunction a function that defines operator() accepting
   *         an argument const T& for each T in Types.
   */
  template <typename Result = std::nullptr_t,
            typename UnaryFunction,
            typename... Types>
  inline Result apply_unary(UnaryFunction f,
                            const typesafe_union<Types...>& u,
                            Result empty_result = Result()) {
    typedef Result(*caller_type)(UnaryFunction f, const void*);

    static caller_type caller[sizeof...(Types)] = {
      &casting_caller<Result, UnaryFunction, const Types, const void*>...
    };

    if (u.which() >= 0) {
      assert(u.which() < sizeof...(Types));
      return (*caller[u.which()])(f, u.data());
    } else {
      return empty_result;
    }
  }

  /**
   * Applies a unary function to a non-empty union and returns its result.
   * When the provided union is empty, return empty_result.
   *
   * \tparam Result the result returned by the function
   * \tparam UnaryFunction a function that defines operator() accepting
   *         an argument T& for each T in Types.
   */
  template <typename Result = std::nullptr_t,
            typename UnaryFunction,
            typename... Types>
  inline Result apply_unary(UnaryFunction f,
                            typesafe_union<Types...>& u,
                            Result empty_result = Result()) {
    typedef Result(*caller_type)(UnaryFunction f, void*);

    static caller_type caller[sizeof...(Types)] = {
      &casting_caller<Result, UnaryFunction, Types, void*>...
    };

    if (u.which() >= 0) {
      assert(u.which() < sizeof...(Types));
      return (*caller[u.which()])(f, u.data());
    } else {
      return empty_result;
    }
  }

  /**
   * Applies a binary function to two unions and returns its result.
   * The unions must be either both empty or both contain a value of
   * the same type. If the unions are empty, returns empty_result.
   *
   * \tparam Result the result returned by the function
   * \tparam Function a function that defines operator() accepting
   *         arguments const T&, const T& for each T in Types.
   * \throw std::invalid_argument if the unions do not contain the same type
   */
  template <typename Result = std::nullptr_t,
            typename BinaryFunction,
            typename... Types>
  inline Result apply_binary(BinaryFunction f,
                             const typesafe_union<Types...>& x,
                             const typesafe_union<Types...>& y,
                             Result empty_result = Result()) {
    typedef Result(*caller_type)(BinaryFunction f, const void*, const void*);

    static caller_type caller[sizeof...(Types)] = {
      &casting_caller<Result, BinaryFunction, const Types, const void*, const void*>...
    };

    if (x.which() == y.which()) {
      int which = x.which();
      if (which >= 0) {
        assert(which < sizeof...(Types));
        return (*caller[which])(f, x.data(), y.data());
      } else {
        return empty_result;
      }
    } else {
      throw std::invalid_argument("Unions have incompatible types");
    }
  }

  /**
   * Applies a binary function to two unions and returns its result.
   * The unions must be either both empty or both contain a value of
   * the same type. If the unions are empty, returns empty_result.
   *
   * \tparam Result the result returned by the function
   * \tparam BinaryFunction a function that defines operator() accepting
   *         arguments T&, T& for each T in Types.
   * \throw std::invalid_argument if the unions do not contain the same type
   */
  template <typename Result = std::nullptr_t,
            typename BinaryFunction,
            typename... Types>
  inline Result apply_binary(BinaryFunction f,
                             typesafe_union<Types...>& x,
                             typesafe_union<Types...>& y,
                             Result empty_result = Result()) {
    typedef Result(*caller_type)(BinaryFunction f, void*, void*);

    static caller_type caller[sizeof...(Types)] = {
      &casting_caller<Result, BinaryFunction, Types, void*, void*>...
    };

    if (x.which() == y.which()) {
      int which = x.which();
      if (which >= 0) {
        assert(which < sizeof...(Types));
        return (*caller[which])(f, x.data(), y.data());
      } else {
        return empty_result;
      }
    } else {
      throw std::invalid_argument("Unions have incompatible types");
    }
  }

  // Traits
  //============================================================================

  /**
   * A specialization of vertex_traits for typesafe_union.
   */
  template <typename... Types>
  struct vertex_traits<typesafe_union<Types...> > {
    typedef typesafe_union<Types...> union_type;

    //! Returns an empty union.
    static union_type null() { return union_type(); }

    //! Returns a special "deleted" union.
    static union_type deleted() { return union_type::deleted(); }

    //! Prints the union to a stream.
    static void print(std::ostream& out, union_type u) { out << u; }

    //! Unions use the default hasher.
    typedef std::hash<union_type> hasher;
  };

} // namespace libgm


namespace std {

  template <typename... Types>
  struct hash<libgm::typesafe_union<Types...>>
    : libgm::default_hash<libgm::typesafe_union<Types...>> { };

} // namespace std

#endif
