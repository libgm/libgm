#ifndef LIBGM_DOMAIN_HPP
#define LIBGM_DOMAIN_HPP

#include <libgm/config.hpp>
#include <libgm/argument/argument_cast.hpp>
#include <libgm/argument/traits.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/functional/utility.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/vector.hpp>
#include <libgm/traits/missing.hpp>

#include <boost/serialization/vector.hpp>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace libgm {

/**
 * A domain that holds the arguments in an std::vector.
 * Domains can be sorted or unsorted (TBD), which affects the
 * speed of their operations.
 */
class Domain : public std::vector<Arg> {
public:
  /// Default constructor. Creates an empty domain.
  Domain() { }

  /// Constructs a domain with given number of empty arguments.
  explicit Domain(size_t n)
    : std::vector<Arg>(n) { }

  /// Creates a domain with the given arguments.
  Domain(std::initializer_list<Arg> init)
    : std::vector<Arg>(init) { }

  /// Creates a domain from the given iterator range.
  template <typename Iterator>
  Domain(Iterator begin, Iterator end)
    : std::vector<Arg>(begin, end) { }

  /// Returns the hash value of a domain.
  friend size_t hash_value(const Domain& dom) {
    return hash_range(dom.begin(), dom.end());
  }

  /// Prints the domain to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const Domain& dom);

  // Sequence operations
  //--------------------------------------------------------------------------

  /// Returns a prefix of this domain.
  Domain prefix(size_t n) const;

  /// Returns a suffix of this domain.
  Domain suffix(size_t n) const;

  /// Returns true if the given domain is a prefix of this domain.
  bool has_prefix(const Domain& dom) const;

  /// Returns true if the given domain is a suffix of this domain.
  bool has_suffix(const Domain& dom) const;

  // Set operations
  //--------------------------------------------------------------------------

  /**
   * Returns the number of times an argument is present in the domain.
   * This operation has a logarithmic time complexity.
   */
  bool contains(Arg x) const;

  /**
   * Partitions this domain into those arguments that are present in the
   * given associative container (set or map) and those that are not.
   */
  void partition(const Assignment& a, Domain& present, Domain& absent) const;

  /**
   * Returns the ordered intersection of two domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend domain operator&(const Domain& a, const Domain& b);

  /**
   * Returns the union of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend Domain operator|(const Domain& a, const Domain& b);

  /**
   * Returns the difference of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend Domain operator-(const Domain& a, const Domain& b);

  /**
   * Returns true if two domains do not have any arguments in common.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool disjoint(const Domain& a, const Domain& b);

  /**
   * Returns true if all the arguments of the first domain are also
   * present in the second domain.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool subset(const Domain& a, const Domain& b);

  /**
   * Returns true if all the arguments of the second domain are also
   * present in the first domain.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool superset(const Domain& a, const Domain& b);

  // Argument operations
  //--------------------------------------------------------------------------

  /**
   * Returns the overall dimensionality for a collection of arguments.
   * This operation has a linear time complexity.
   */
  size_t arity(const ShapeMap& map, size_t start = 0) const;

  /**
   * Returns the shape of the arguments in this domain.
   * This operation has a linear time complexity.
   */
  std::vector<size_t> shape(const ShapeMap& map) const;

  /**
   * Computes the dims of the specified arguments in this domain.
   * This operation has a linear time complexity.
   */
  std::vector<unsigned> dims(const Domain& args) const;

}; // class Domain

} // namespace libgm

namespace std {

struct hash<libgm::Domain>
  : libgm::default_hash<libgm::Domain> { };

} // namespace std

#endif
