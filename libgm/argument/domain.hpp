#pragma once

#include <libgm/argument/concepts/argument.hpp>
#include <libgm/argument/dims.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/iterator/counting_output_iterator.hpp>

#include <boost/container_hash/hash.hpp>

#include <cereal/types/vector.hpp>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace libgm {

/**
 * A domain that holds the arguments in an std::vector.
 */
template <Argument Arg>
class Domain : public std::vector<Arg> {
public:
  using argument_type = Arg;
  using std::vector<Arg>::vector;
  using std::vector<Arg>::erase;

  /**
   * Constructs a domain from a range.
   */
  template <typename It, typename Sentinel>
  Domain(std::ranges::subrange<It, Sentinel> range)
    : std::vector<Arg>(range.begin(), range.end()) {}

  /// Returns the hash value of a domain.
  friend std::size_t hash_value(const Domain& dom) {
    return boost::hash_range(dom.begin(), dom.end());
  }

  /**
   * Prints the domain to an output stream.
   */
  friend std::ostream& operator<<(std::ostream& out, const Domain& dom) {
    out << '[';
    for (std::size_t i = 0; i < dom.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << dom[i];
    }
    out << ']';
    return out;
  }

  /**
   * Returns a prefix of this domain.
   * \throws std::invalid_argument if n is greater than the length of this domain
   */
  Domain prefix(std::size_t n) const {
    if (n > this->size()) {
      throw std::invalid_argument("Domain: Prefix out of bounds");
    }
    return {this->begin(), this->begin() + n};
  }

  /**
   * Returns a suffix of this domain with given length
   * \throws std::invalid_argument if n is greater than the length of this domain
   */
  Domain suffix(std::size_t n) const {
    if (n > this->size()) {
      throw std::invalid_argument("Domain: Suffix out of bounds");
    }
    return {this->end() - n, this->end()};
  }

  /**
   * Returns true if the given domain is a prefix of this domain.
   * This operation has a linear complexity O(|a|).
   */
  bool has_prefix(const Domain& a) const {
    return a.size() <= this->size() && std::ranges::equal(a.begin(), a.end(), this->begin(), this->begin() + a.size());
  }

  /**
   * Returns true if the given domain is a suffix of this domain.
   * This operation has a linear complexity O(|a|).
   */
  bool has_suffix(const Domain& a) const {
    return a.size() <= this->size() && std::ranges::equal(a.begin(), a.end(), this->end() - a.size(), this->end());
  }

  /**
   * Appends another domain to this one.
   */
  void append(const Domain& other) {
    this->insert(this->end(), other.begin(), other.end());
  }

  /**
   * Returns the index of the given argument in this domain.
   */
  std::size_t index(const Arg& arg) const {
    auto it = std::ranges::find(*this, arg);
    if (it == this->end()) {
      throw std::invalid_argument("Domain::index: Argument not found.");
    }
    return static_cast<std::size_t>(it - this->begin());
  }

  /**
   * Sorts this domain.
   */
  void sort() {
    std::ranges::sort(*this);
  }

  /**
   * Sorts this domain and extract unique elements.
   */
  void unique() {
    sort();
    this->erase(std::ranges::unique(*this).begin(), this->end());
  }

  /**
   * Returns true if all the arguments of the domains follow the ordering.
   * This operation has a linear time complexity, O(|this|).
   */
  bool is_sorted() const {
    return std::ranges::is_sorted(*this);
  }

  /**
   * Returns true if an argument is present in this sorted domain.
   * This operation has a logarithmic time complexity.
   */
  bool contains(const Arg& arg) const {
    return std::ranges::binary_search(*this, arg);
  }

  /**
   * Removes the specified argument from this domain. The argument must exist.
   */
  void erase(const Arg& arg) {
    auto it = std::ranges::lower_bound(*this, arg);
    if (it != this->end() && *it == arg) {
      this->erase(it);
      return;
    }
    throw std::invalid_argument("Domain::erase: Argument not found.");
  }

  /**
   * Returns true if all the arguments of the first domain are also
   * present in the second domain.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool is_subset(const Domain& a, const Domain& b) {
    return std::ranges::includes(b, a);
  }

  /**
   * Returns true if all the arguments of the second domain are also
   * present in the first domain.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool is_superset(const Domain& a, const Domain& b) {
    return std::ranges::includes(a, b);
  }

  /**
   * Returns true if two ordered domains do not have any arguments in common.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend bool are_disjoint(const Domain& a, const Domain& b) {
    return intersection_size(a, b) == 0;
  }

  /**
   * Returns the size of the intersection of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend std::size_t intersection_size(const Domain& a, const Domain& b) {
    CountingOutputIterator out;
    return std::ranges::set_intersection(a, b, out).out.count();
  }

  /**
   * Returns the union of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend Domain operator|(const Domain& a, const Domain& b) {
    Domain result(a.size() + b.size());
    auto end = std::ranges::set_union(a, b, result.begin()).out;
    result.resize(static_cast<std::size_t>(end - result.begin()));
    return result;
  }

  /**
   * Returns the ordered intersection of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend Domain operator&(const Domain& a, const Domain& b) {
    Domain result(std::min(a.size(), b.size()));
    auto end = std::ranges::set_intersection(a, b, result.begin()).out;
    result.resize(static_cast<std::size_t>(end - result.begin()));
    return result;
  }

  /**
   * Returns the set difference of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  friend Domain operator-(const Domain& a, const Domain& b) {
    Domain result(a.size());
    auto end = std::ranges::set_difference(a, b, result.begin()).out;
    result.resize(static_cast<std::size_t>(end - result.begin()));
    return result;
  }

  /**
   * Computes the ordered intersection of two ordered domains, storing the result in this.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  Domain& operator&=(const Domain& other) {
    auto it = std::ranges::set_intersection(*this, other, this->begin()).out;
    this->erase(it, this->end());
    return *this;
  }

  /**
   * Returns the set difference of two ordered domains, storing the result in this.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  Domain& operator-=(const Domain& other) {
    auto it = std::ranges::set_difference(*this, other, this->begin()).out;
    this->erase(it, this->end());
    return *this;
  }

  /**
   * Returns the shape of the arguments in this domain.
   * This operation has a linear time complexity.
   */
  Shape shape(const ShapeMap<Arg>& map) const {
    Shape result(this->size());
    for (std::size_t i = 0; i < this->size(); ++i) {
      result[i] = map((*this)[i]);
    }
    return result;
  }

  /**
   * Computes the dims of the specified arguments in this domain.
   * This operation has a linear time complexity.
   */
  Dims dims(const Domain& args) const {
    Dims result;
    std::size_t j = 0;
    for (std::size_t i = 0; i < this->size() && j < args.size(); ++i) {
      if ((*this)[i] == args[j]) {
        result.set(i);
        ++j;
      }
    }
    if (j != args.size()) {
      throw std::invalid_argument("Domain::dims: The specified arguments are not an ordered subset");
    }
    return result;
  }

  /**
   * Computes the dims of all but the specified argument in this domain.
   * This operation has a linear time complexity.
   */
  Dims dims_omit(const Arg& arg) const {
    Dims result;
    bool found = false;
    for (std::size_t i = 0; i < this->size(); ++i) {
      if ((*this)[i] != arg) {
        result.set(i);
      } else {
        found = true;
      }
    }
    if (!found) {
      throw std::invalid_argument("Domain::dims_omit: The specified argument could not be found");
    }
    return result;
  }
}; // class Domain

} // namespace libgm

namespace std {

template <libgm::Argument Arg>
struct hash<libgm::Domain<Arg>> : boost::hash<libgm::Domain<Arg>> { };

} // namespace std
