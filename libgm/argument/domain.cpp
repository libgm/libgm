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

bool Domain::contains(Arg x) const {
  return std::binary_search(begin(), end(), x);
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

Domain operator&(const Domain& a, const Domain& b) {
  Domain result(std::min(a.size(), b.size()));
  auto end = std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), result.begin());
  result.resize(end - result.begin());
  return result;
}

Domain operator|(const Domain& a, const Domain& b) {
  Domain result(a.size() + b.size());
  auto end = std::set_union(a.begin(), a.end(), b.begin(), b.end(), result.begin());
  result.resize(end - result.begin());
  return result;
}

Domain operator-(const Domain& a, const Domain& b) {
  Domain result(a.size());
  auto end = std::set_difference(a.begin(), a.end(), b.begin(), b.end(), result.begin());
  result.resize(end - result.begin());
  return result;
}

bool subset(const Domain& a, const Domain& b) {
  return std::includes(b.begin(), b.end(), a.begin(), a.end());
}

bool superset(const Domain& a, const Domain& b) {
  return std::includes(a.begin(), a.end(), b.begin(), b.end());
}

size_t Domain::arity(const ShapeMap& map, size_t start = 0) const {
  size_t size = 0;
  for (size_t i = start; i < size(); ++i) {
    size += map((*this)[i]).size();
  }
  return size;
}

Shape Domain::shape(const ShapeMap& map, std::vector<size_t>& vec) const {
  vec.clear();
  size_t real_size = 0;
  for (Arg arg : *this) {
    Shape shape = map(arg);
    if (shape.is_discrete()) {
      vec.insert(vec.back(), shape.begin(), shape.end());
    } else {
      real_size += shape.size();
    }
  }
  if (vec.empty()) {
    return Shape(real_size);
  } else if (real_size == 0) {
    return Shape(vec.data(), vec.size());
  } else {
    throw std::invalid_argument("Hybrid domains not supported yet");
  }
}

Dims Domain::dims(const ShapeMap& map, const Domain& args) const {
  if (args == *this) {
    return Dims::all();
  }
  if (has_prefix(args)) {
    return Dims::head(args.arity(map));
  }
  if (has_suffix(args)) {
    return Dims::tail(args.arity(map));
  }
  if (subset(args, *this)) {
    std::vector<unsigned> result;
    boost::counting_iterator<unsigned> it(0);
    for (size_t i = 0, j = 0; i < size() && j < size(); ++i) {
      Shape shape = map((*this)[i]);
      if ((*this)[i] == args[j]) {
        std::copy(it, it + shape.size(), std::back_inserter(result));
        ++j;
      }
      it += shape.size();
    }
    return Dims::list(std::move(result));
  }
  throw std::invalid_argument("The specified domain is not a subset of this");
}

/**
 * Computes the start indexes of this domain in a linear ordering
 * of arguments.
 *
 * \tparam Map A map object with keys Arg and values size_t.
 * \return the number of dimensions of this domain.
 */
template <typename Map>
size_t insert_start(Map& start) const {
  size_t pos = 0;
  for (Arg arg : *this) {
    start.emplace(arg, pos);
    pos += argument_arity(arg);
  }
  return pos;
}

} // namespace libgm
