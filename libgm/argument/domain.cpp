#include "domain.hpp"

namespace libgm {

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
  assert(n <= size());
  return {begin(), begin() + n};
}

Domain Domain::suffix(size_t n) const {
  assert(n <= size());
  return {end() - n, end()};
}

bool Domain::has_prefix(const Domain& dom) const {
  return dom.size() <= size() && std::equal(dom.begin(), dom.end(), begin());
}

/// Returns true if the given domain is a suffix of this domain.
bool Domain::has_suffix(const Domain& dom) const {
  return dom.size() <= size() && std::equal(dom.begin(), dom.end(), end() - dom.size());
}

void Domain::partition(const Assignment& a, Domain& present, Domain& absent) const {
  for (Arg arg : *this) {
    if (a.count(arg)) {
      present.push_back(arg);
    } else {
      absent.push_back(arg);
    }
  }
}

Shape Domain::shape(const ShapeMap& map) const {
  Shape shape(size());
  for (size_t i = 0; i < size(); ++i) {
    shape[i] = map((*this)][i]);
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

} // namespace libgm
