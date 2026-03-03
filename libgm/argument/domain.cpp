#include "domain.hpp"

#include <libgm/iterator/counting_output_iterator.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace libgm {

Domain::Domain(const ArgSet& set)
  : std::vector<Arg>(set.values()) {
  sort();
}

std::ostream& operator<<(std::ostream& out, const Domain& dom) {
  out << '[';
  for (size_t i = 0; i < dom.size(); ++i) {
    if (i > 0) { out << ", "; }
    out << dom[i];
  }
  out << ']';
  return out;
}

Domain Domain::prefix(size_t n) const {
  if (n > size()) throw std::invalid_argument("Domain: Prefix out of bounds");
  return {begin(), begin() + n};
}

Domain Domain::suffix(size_t n) const {
  if (n > size()) throw std::invalid_argument("Domain: Suffix out of bounds");
  return {end() - n, end()};
}

bool Domain::has_prefix(const Domain& dom) const {
  return dom.size() <= size() && std::equal(dom.begin(), dom.end(), begin());
}

bool Domain::has_suffix(const Domain& dom) const {
  return dom.size() <= size() && std::equal(dom.begin(), dom.end(), end() - dom.size());
}

void Domain::append(const Domain& other) {
  insert(end(), other.begin(), other.end());
}

void Domain::sort() {
  std::sort(begin(), end());
}

void Domain::unique() {
  sort();
  erase(std::unique(begin(), end()), end());
}

bool Domain::is_sorted() const {
  return std::is_sorted(begin(), end());
}

bool Domain::contains(Arg x) const {
  return std::binary_search(begin(), end(), x);
}

void Domain::erase(Arg x){
  auto it = std::lower_bound(begin(), end(), x);
  if (it != end() && *it == x) {
    erase(it);
  } else {
    throw std::invalid_argument("Domain::erase: Argument not found.");
  }
}

bool is_subset(const Domain& a, const Domain& b) {
  return std::includes(b.begin(), b.end(), a.begin(), a.end());
}

bool is_superset(const Domain& a, const Domain& b) {
  return std::includes(a.begin(), a.end(), b.begin(), b.end());
}

bool are_disjoint(const Domain& a, const Domain& b) {
  return intersection_size(a, b) == 0;
}

size_t intersection_size(const Domain& a, const Domain& b) {
  CountingOutputIterator out;
  return std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), out).count();
}

Domain operator|(const Domain& a, const Domain& b) {
  Domain result(a.size() + b.size());
  auto end = std::set_union(a.begin(), a.end(), b.begin(), b.end(), result.begin());
  result.resize(end - result.begin());
  return result;
}

Domain operator&(const Domain& a, const Domain& b) {
  Domain result(std::min(a.size(), b.size()));
  auto end = std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), result.begin());
  result.resize(end - result.begin());
  return result;
}

Domain operator-(const Domain& a, const Domain& b) {
  Domain result(a.size());
  auto end = std::set_difference(a.begin(), a.end(), b.begin(), b.end(), result.begin());
  result.resize(end - result.begin());
  return result;
}

Domain& Domain::operator&=(const Domain& other) {
  auto it = std::set_intersection(begin(), end(), other.begin(), other.end(), begin());
  erase(it, end());
  return *this;
}

Domain& Domain::operator-=(const Domain& other) {
  auto it = std::set_difference(begin(), end(), other.begin(), other.end(), begin());
  erase(it, end());
  return *this;
}

Shape Domain::shape(const ShapeMap& map) const {
  Shape shape(size());
  for (size_t i = 0; i < size(); ++i) {
    shape[i] = map((*this)[i]);
  }
  return shape;
}

Dims Domain::dims(const Domain& args) const {
  Dims result;
  size_t j = 0;
  for (size_t i = 0; i < size() && j < args.size(); ++i) {
    if ((*this)[i] == args[j]) {
      result.set(i);
      ++j;
    }
  }
  if (j < args.size()) return result;

  throw std::invalid_argument("Domain::dims: The specified arguments are not an ordered subset");
}

Dims Domain::dims_omit(Arg arg) const {
  Dims result;
  bool found = false;
  for (size_t i = 0; i < size(); ++i) {
    if ((*this)[i] != arg) {
      result.set(i);
    } else {
      found = true;
    }
  }

  if (found) return result;

  throw std::invalid_argument("Domain::dims_omit: The specified argument could not be found");
}

} // namespace libgm
