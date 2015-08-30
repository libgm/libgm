#ifndef LIBGM_ARRAY_DOMAIN_HPP
#define LIBGM_ARRAY_DOMAIN_HPP

#include <libgm/argument/argument_cast.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/serialization/array.hpp>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <vector>

namespace libgm {

  /**
   * A fixed-size domain that holds exactly the given number of elements.
   * This domain type supports all operations of std::array and can be
   * serialized.
   */
  template <typename Arg, std::size_t N>
  struct array_domain : public std::array<Arg, N> {
  public:
    //! Default constructor. Creates an uninitialized (invalid) domain.
    array_domain() { }

    //! Creates a domain with the given elements.
    array_domain(std::initializer_list<Arg> init) {
      assert(init.size() == N);
      std::copy(init.begin(), init.end(), this->begin());
    }

    //! Creates a domain equivalent to the given vector.
    array_domain(const std::vector<Arg>& elems) {
      assert(elems.size() == N);
      std::copy(elems.begin(), elems.end(), this->begin());
    }

    //! Serializes the domain.
    void save(oarchive& ar) const {
      ar << static_cast<const std::array<Arg, N>&>(*this);
    }

    //! Deserializes the domain.
    void load(iarchive& ar) {
      ar >> static_cast<std::array<Arg, N>&>(*this);
    }

    //! Prints the domain to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const array_domain& a) {
      out << '[';
      for (std::size_t i = 0; i < N; ++i) {
        if (i > 0) { out << ','; }
        argument_traits<Arg>::print(out, a[i]);
      }
      out << ']';
      return out;
    }

    // Sequence operations
    //==========================================================================

    //! Returns true if the given domain is a prefix of this domain.
    template <std::size_t M>
    bool prefix(const array_domain<Arg, M>& dom) const {
      return M <= N && std::equal(dom.begin(), dom.end(), this->begin());
    }

    //! Returnrs true if the given domain is a suffix of this domain.
    template <std::size_t M>
    bool suffix(const array_domain<Arg, M>& dom) const {
      return M <= N && std::equal(dom.begin(), dom.end(), this->end() - M);
    }

    // Set operations
    //==========================================================================

    //! Returns the number of times an argument is present in the domain.
    std::size_t count(const Arg& x) const {
      return std::count(this->begin(), this->end(), x);
    }

    // Argument operations
    //==========================================================================

    //! Returns true if two domains are type-compatible.
    friend bool compatible(const array_domain& a, const array_domain& b) {
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (!argument_traits<Arg>::compatible(a[i], b[i])) {
          return false;
        }
      }
      return true;
    }

    //! Returns the vector dimensionality for a collection of arguments.
    std::size_t num_dimensions() const {
      std::size_t size = 0;
      for (Arg arg : *this) {
        size += argument_traits<Arg>::num_dimensions(arg);
      }
      return size;
    }

    //! Returns the number of values for a collection of discrete arguments.
    template <bool B = is_discrete<Arg>::value>
    typename std::enable_if<B, std::size_t>::type num_values() const {
      std::size_t size = 1;
      for (Arg arg : *this) {
        std::size_t values = argument_traits<Arg>::num_values(arg);
        if (std::numeric_limits<std::size_t>::max() / values <= size) {
          throw std::out_of_range("num_values: possibly overflows std::size_t");
        }
        size *= values;
      }
      return size;
    }

  }; // class array_domain


  // Set operations
  //============================================================================

  /**
   * The concatentation of two fixed-size domains.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  array_domain<Arg, M+N>
  concat(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    array_domain<Arg, M+N> r;
    std::copy(a.begin(), a.end(), r.begin());
    std::copy(b.begin(), b.end(), r.begin() + M);
    return r;
  }

  /**
   * Returns the ordered union of two fixed-size domains.
   * This operation is valid only if the two domains are disjoint.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  array_domain<Arg, M+N>
  operator+(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    assert(disjoint(a, b));
    return concat(a, b);
  }

  /**
   * Returns the difference of two fixed-size domains.
   * This operation is valid only if b is a subset of a.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  array_domain<Arg, M-N>
  operator-(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    static_assert(M > N, "The first argument must be the larger domain");
    array_domain<Arg, M-N> r;
    std::size_t i = 0;
    for (Arg x : a) {
      if (!b.count(x)) {
        assert(i < M-N);
        r[i++] = x;
      }
    }
    assert(i == M-N);
    return r;
  }

  /**
   * Returns the ordered intersection of two fixed-size domains.
   * This operation is valid only if domain a is a strict subset of b.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  typename std::enable_if<(M < N), array_domain<Arg, M>>::type
  operator&(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    assert(subset(a, b));
    return a;
  }

  /**
   * Returns the ordered intersection of two fixed-size domains.
   * This operation is valid only if domain b is a subset of a.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  typename std::enable_if<(M >= N), array_domain<Arg, N> >::type
  operator&(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    array_domain<Arg, N> r;
    std::size_t i = 0;
    for (Arg x : a) {
      if (b.count(x)) {
        assert(i < N);
        r[i++] = x;
      }
    }
    assert(i == N);
    return r;
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  bool disjoint(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    for (Arg x : a) {
      if (b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates array_domain
   */
  template <typename Arg>
  bool disjoint(const array_domain<Arg, 1>& a, const array_domain<Arg, 1>& b) {
    return a[0] != b[0];
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates array_domain
   */
  template <typename Arg>
  bool disjoint(const array_domain<Arg, 2>& a, const array_domain<Arg, 2>& b) {
    return a[0] != b[0] && a[1] != b[0] && a[0] != b[1] && a[1] != b[1];
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates array_domain
   */
  template <typename Arg, std::size_t N>
  bool
  equivalent(const array_domain<Arg, N>& a, const array_domain<Arg, N>& b) {
    array_domain<Arg, N> as = a;
    array_domain<Arg, N> bs = b;
    std::sort(as.begin(), as.end());
    std::sort(bs.begin(), bs.end());
    return as == bs;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  typename std::enable_if<M != N, bool>::type
  equivalent(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    return false;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates array_domain
   */
  template <typename Arg>
  bool
  equivalent(const array_domain<Arg, 1>& a, const array_domain<Arg, 1>& b) {
    return a[0] == b[0];
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates array_domain
   */
  template <typename Arg>
  bool
  equivalent(const array_domain<Arg, 2>& a, const array_domain<Arg, 2>& b) {
    return std::minmax(a[0], a[1]) == std::minmax(b[0], b[1]);
  }

  /**
   * Returns true if all the elements of the first domain are also present in
   * the second domain.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  bool subset(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    for (Arg x : a) {
      if (!b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the first domain are also present in
   * the second domain.
   * \relates array_domain
   */
  template <typename Arg>
  bool subset(const array_domain<Arg, 1>& a, const array_domain<Arg, 2>& b) {
    return a[0] == b[0] || a[0] == b[1];
  }

  /**
   * Returns true if all the elements of the second domain are also present
   * in the first domain.
   * \relates array_domain
   */
  template <typename Arg, std::size_t M, std::size_t N>
  bool superset(const array_domain<Arg, M>& a, const array_domain<Arg, N>& b) {
    for (Arg x : b) {
      if (!a.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the second domain are also present
   * in the first domain.
   * \relates array_domain
   */
  template <typename Arg>
  bool superset(const array_domain<Arg, 2>& a, const array_domain<Arg, 1>& b) {
    return a[0] == b[0] || a[1] == b[0];
  }

  // Argument operations
  //============================================================================

  /**
   * Converts one domain to a domain with another argument type.
   *
   * \tparam Target
   *         The target argument type. Must be convertible from Source using
   *         argument_cast.
   * \tparam Source
   *         The original argument type.
   * \relates array_domain
   */
  template <typename Target, typename Source, std::size_t N>
  array_domain<Target, N> argument_cast(const array_domain<Source, N>& dom) {
    static_assert(is_convertible_argument<Source, Target>::value,
                  "Source must be argument-convertible to Target");

    array_domain<Target, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = argument_cast<Target>(dom[i]);
    }
    return result;
  }

} // namespace libgm

#endif
