#ifndef LIBGM_BASIC_DOMAIN_HPP
#define LIBGM_BASIC_DOMAIN_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/vector.hpp>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <vector>

namespace libgm {

  /**
   * A domain that holds the elements in an std::vector.
   *
   * \tparam Arg a type that satisfies the Argument concept
   */
  template <typename Arg>
  class basic_domain : public std::vector<Arg> {
  public:
    //! Default constructor. Creates an empty domain.
    basic_domain() { }

    //! Constructs a domain with given number of empty arguments.
    explicit basic_domain(std::size_t n)
      : std::vector<Arg>(n) { }

    //! Creates a domain with the given elements.
    basic_domain(std::initializer_list<Arg> init)
      : std::vector<Arg>(init) { }

    //! Creates a domain from the given argument vector.
    basic_domain(const std::vector<Arg>& elems)
      : std::vector<Arg>(elems) { }

    //! Creates a domain from the given argument array.
    template <std::size_t N>
    basic_domain(const std::array<Arg, N>& elems)
      : std::vector<Arg>(elems.begin(), elems.end()) { }

    //! Creates a domain from the given iterator range.
    template <typename Iterator>
    basic_domain(Iterator begin, Iterator end)
      : std::vector<Arg>(begin, end) { }

    //! Creates a domain from the given iterator range.
    template <typename Iterator>
    explicit basic_domain(const iterator_range<Iterator>& range)
      : std::vector<Arg>(range.begin(), range.end()) { }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      ar.serialize_range(this->begin(), this->end());
    }

    //! Laods the domain from an archive.
    void load(iarchive& ar) {
      this->clear();
      ar.deserialize_range<Arg>(std::back_inserter(*this));
    }

    //! Returns the number of times an argument is present in the domain.
    std::size_t count(const Arg& x) const {
      return std::count(this->begin(), this->end(), x);
    }

    /**
     * Partitions this domain into those elements that are present in the
     * given map and those that are not.
     */
    template <typename Map>
    void partition(const Map& map,
                   basic_domain& intersect, basic_domain& difference) const {
      for (Arg arg : *this) {
        if (map.count(arg)) {
          intersect.push_back(arg);
        } else {
          difference.push_back(arg);
        }
      }
    }

    //! Substitutes arguments in-place according to a map.
    template <typename Map>
    void subst(const Map& map) {
      for (Arg& arg : *this) {
        arg = map.at(arg); // TODO: check compatibility
      }
    }

    //! Sorts the elements of the domain in place.
    basic_domain& sort() {
      std::sort(this->begin(), this->end());
      return *this;
    }

    /**
     * Removes the duplicate elements from the domain in place.
     * Does not preserve the relative ordere of elements in the domain.
     */
    basic_domain& unique() {
      std::sort(this->begin(), this->end());
      auto new_end = std::unique(this->begin(), this->end());
      this->erase(new_end, this->end());
      return *this;
    }

    //! Returns the hash value of a domain.
    friend std::size_t hash_value(const basic_domain& dom) {
      return hash_range(dom.begin(), dom.end());
    }

  };

  /**
   * Prints the domain to an output stream.
   * \relates basic_domain
   */
  template <typename Arg>
  std::ostream& operator<<(std::ostream& out, const basic_domain<Arg>& dom) {
    out << '[';
    for (std::size_t i = 0; i < dom.size(); ++i) {
      if (i > 0) { out << ','; }
      argument_traits<Arg>::print(out, dom[i]);
    }
    out << ']';
    return out;
  }

  // Set operations
  //============================================================================

  /**
   * The concatenation of two domains.
   * \relates basic_domain
   */
  template <typename Arg>
  basic_domain<Arg>
  operator+(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    basic_domain<Arg> r;
    r.reserve(a.size() + b.size());
    std::copy(a.begin(), a.end(), std::back_inserter(r));
    std::copy(b.begin(), b.end(), std::back_inserter(r));
    return r;
  }

  /**
   * Returns the difference of two domains.
   * \relates basic_domain
   */
  template <typename Arg>
  basic_domain<Arg>
  operator-(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    basic_domain<Arg> r;
    for (Arg x : a) {
      if (!b.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns the ordered union of two domains.
   * \relates basic_domain
   */
  template <typename Arg>
  basic_domain<Arg>
  operator|(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    basic_domain<Arg> r = a;
    for (Arg x : b) {
      if (!a.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns the ordered intersection of two domains.
   * \relates basic_domain
   */
  template <typename Arg>
  basic_domain<Arg>
  operator&(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    basic_domain<Arg> r;
    for (Arg x : a) {
      if (b.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates basic_domain
   */
  template <typename Arg, typename MapOrSet>
  bool disjoint(const basic_domain<Arg>& a, const MapOrSet& b) {
    for (Arg x : a) {
      if (b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are
   * equivalent if they have the same elements, disregarding the order.
   * \relates basic_domain
   */
  template <typename Arg>
  bool equivalent(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (Arg x : a) {
      if (!b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the first domain are also
   * present in the second domain.
   * \relates basic_domain
   */
  template <typename Arg, typename MapOrSet>
  bool subset(const basic_domain<Arg>& a, const MapOrSet& b) {
    if (a.size() > b.size()) {
      return false;
    }
    for (Arg x : a) {
      if (!b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the second domain are also
   * present in the first domain.
   * \relates basic_domain
   */
  template <typename Arg>
  bool superset(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    return subset(b, a);
  }

  /**
   * Returns true if domain a is a suffix of domain b.
   * \relates basic_domain
   */
  template <typename Arg>
  bool prefix(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
  }

  /**
   * Returns true if domain a is a suffix of domain b.
   * \relates basic_domain
   */
  template <typename Arg>
  bool suffix(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    return a.size() <= b.size()
      && std::equal(a.begin(), a.end(), b.end() - a.size());
  }

  // Argument operations
  //============================================================================

  /**
   * Returns true if two domains are compatible.
   * \relates basic_domain
   */
  template <typename Arg>
  bool compatible(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (!argument_traits<Arg>::compatible(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns the number of values for a collection of discrete arguments.
   * This is equal to to the product of the numbers of values of the argument.
   *
   * \tparam Arg a type that satisfies the DiscreteArgument concept
   * \throws std::out_of_range in case of overflow
   * \relates basic_domain
   */
  template <typename Arg>
  std::size_t num_values(const basic_domain<Arg>& dom) {
    std::size_t size = 1;
    for (Arg arg : dom) {
      std::size_t values = argument_traits<Arg>::num_values(arg);
      if (std::numeric_limits<std::size_t>::max() / values <= size) {
        throw std::out_of_range("num_values: possibly overflows std::size_t");
      }
      size *= values;
    }
    return size;
  }

  /**
   * Returns the dimensionality for a collection of continuous arguments.
   * This is equal to the sum of the numbers of dimensions of the arguments.
   *
   * \relates basic_domain
   * \tparam Arg a type that satisfies the ContinuousArgument concept
   * \throws std::out_of_range in case of overflow
   */
  template <typename Arg>
  std::size_t num_dimensions(const basic_domain<Arg>& dom) {
    std::size_t size = 0;
    for (Arg arg : dom) {
      size += argument_traits<Arg>::num_dimensions(arg);
    }
    return size;
  }

} // namespace libgm


namespace std {

  template <typename Arg>
  struct hash<libgm::basic_domain<Arg>>
    : libgm::default_hash<libgm::basic_domain<Arg>> { };

} // namespace std

#endif
