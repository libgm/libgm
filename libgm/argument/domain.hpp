#pragma once

#include <libgm/argument/argument_cast.hpp>
#include <libgm/argument/traits.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/vector.hpp>

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
 */
class Domain : public std::vector<Arg> {
public:
  // Bring in constructors.
  using std::vector<Arg>::vector;

  /// Returns the hash value of a domain.
  friend size_t hash_value(const Domain& dom) {
    return hash_range(dom.begin(), dom.end());
  }

  /// Prints the domain to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const Domain& dom);

  /// Returns a prefix of this domain.
  Domain prefix(size_t n) const;

  /// Returns a suffix of this domain.
  Domain suffix(size_t n) const;

  /// Returns true if the given domain is a prefix of this domain.
  bool has_prefix(const Domain& dom) const;

  /// Returns true if the given domain is a suffix of this domain.
  bool has_suffix(const Domain& dom) const;

  /**
   * Partitions this domain into those arguments that are present in the
   * given associative container (set or map) and those that are not.
   */
  void partition(const Assignment& a, Domain& present, Domain& absent) const;

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

}; // class Domain

} // namespace libgm

namespace std {

struct hash<libgm::Domain>
  : libgm::default_hash<libgm::Domain> { };

} // namespace std
