#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/argument/dims.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/datastructure/subrange.hpp>

#include <boost/functional/hash.hpp>

#include <cereal/types/vector.hpp>

#include <iostream>
#include <memory>
#include <vector>

namespace libgm {

/**
 * A domain that holds the arguments in an std::vector.
 */
class Domain : public std::vector<Arg> {
public:
  // Bring in constructors.
  using std::vector<Arg>::vector;

  // Bring in th eshadowed operation.
  using std::vector<Arg>::erase;

  /**
   * Constructs a domain from a range.
   */
  template <typename IT>
  Domain(const SubRange<IT>& range)
    : std::vector<Arg>(range.begin(), range.end()) {}

  /**
   * Converts an argument set to an (ordered) domain.
   */
  explicit Domain(const ArgSet& set);

  // /// Returns the hash value of a domain.
  // friend size_t hash_value(const Domain& dom) {
  //   return boost::hash_range(dom.begin(), dom.end());
  // }

  /**
   * Prints the domain to an output stream.
   */
  friend std::ostream& operator<<(std::ostream& out, const Domain& dom);

  // Sequence operations
  //----------------------------------------------------------------------------

  /**
   * Returns a prefix of this domain.
   * \throws std::invalid_argument if n is greater than the length of this domain
   */
  Domain prefix(size_t n) const;

  /**
   * Returns a suffix of this domain with given length
   * \throws std::invalid_argument if n is greater than the length of this domain
   */
  Domain suffix(size_t n) const;

  /**
   * Returns true if the given domain is a prefix of this domain.
   * This operation has a linear complexity O(|a|).
   */
  bool has_prefix(const Domain& a) const;

  /**
   * Returns true if the given domain is a suffix of this domain.
   * This operation has a linear complexity O(|a|).
   */
  bool has_suffix(const Domain& a) const;

  /**
   * Appends another domain to this one.
   */
  void append(const Domain& other);

  // Set operations
  //----------------------------------------------------------------------------

  /**
   * Sorts this domain.
   */
  void sort();

  /**
   * Sorts this domain and extract unique elements.
   */
  void unique();

  /**
   * Returns true if all the arguments of the domains follow the ordering.
   * This operation has a linear time complexity, O(|this|).
   */
  bool is_sorted() const;

  /**
   * Returns true if an argument is present in this sorted domain.
   * This operation has a logarithmic time complexity.
   */
  bool contains(Arg x) const;

  /**
   * Removes the specified argument from this domain. The argument must exist.
   */
  void erase(Arg x);

  /**
   * Returns true if all the arguments of the first domain are also
   * present in the second domain.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool is_subset(const Domain& a, const Domain& b);

  /**
   * Returns true if all the arguments of the second domain are also
   * present in the first domain.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool is_superset(const Domain& a, const Domain& b);

  /**
   * Returns true if two ordered domains do not have any arguments in common.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool are_disjoint(const Domain& a, const Domain& b);

  /**
   * Returns the size of the intersection of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend size_t intersection_size(const Domain& a, const Domain& b);

  /**
   * Returns the union of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend Domain operator|(const Domain& a, const Domain& b);

  /**
   * Returns the ordered intersection of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend Domain operator&(const Domain& a, const Domain& b);

  /**
   * Returns the set difference of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend Domain operator-(const Domain& a, const Domain& b);

  /**
   * Computes the ordered intersection of two ordered domains, storing the result in this.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  Domain& operator&=(const Domain& other);

  /**
   * Returns the set difference of two ordered domains, storing the result in this.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  Domain& operator-=(const Domain& other);

  // Factor operations
  //----------------------------------------------------------------------------

  /**
   * Returns the shape of the arguments in this domain.
   * This operation has a linear time complexity.
   */
  Shape shape(const ShapeMap& map) const;

  /**
   * Computes the dims of the specified arguments in this domain.
   * This operation has a linear time complexity.
   */
  Dims dims(const Domain& args) const;

  /**
   * Computes the dims of all but the specified argument in this domain.
   * This operation has a linear time complexity.
   */
  Dims dims_omit(Arg arg) const;

}; // class Domain

} // namespace libgm

namespace std {

template<>
struct hash<libgm::Domain> : boost::hash<libgm::Domain> { };

} // namespace std
