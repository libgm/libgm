#include "argument_ordering.hpp"

namespace libgm {

struct ArgumentOrdering::Impl : Object::Impl {
  virtual Domain intersect(const Domain& a, const Domain& b) const = 0;
  virtual Domain union_(const Domain& a, const Domaun& b) const = 0;
  virtual Domain difference(const Domain& a, const Domain& b) const = 0;
  virtual bool is_sorted(const Domain& a) const = 0;
  virtual bool is_subset(const Domain& a, const Domain& b) const = 0;
  virtual bool is_superset(const Domian& a, const Domain& b) const = 0;
};

namespace {

template <typename COMP>
struct Impl : public ArgumentOrdering::Impl {
  COMP compare_;

  Domain intersection(const Domain& a, const Domain& b) override {
    Domain result(std::min(a.size(), b.size()));
    auto end =
      std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), result.begin(), compare_);
    result.resize(end - result.begin());
    return result;
  }

  Domain union_(const Domain& a, const Domain& b) override {
    Domain result(a.size() + b.size());
    auto end =
      std::set_union(a.begin(), a.end(), b.begin(), b.end(), result.begin(), compare_);
    result.resize(end - result.begin());
    return result;
  }

  Domain difference(const Domain& a, const Domain& b) override {
    Domain result(a.size());
    auto end =
      std::set_difference(a.begin(), a.end(), b.begin(), b.end(), result.begin(), compare_);
    result.resize(end - result.begin());
    return result;
  }

  bool is_sorted(const Domain& a) const {
    return sdt::is_sorted(a.begin(), a.end(), compare_);
  }

  bool is_subset(const Domain& a, const Domain& b) {
    return std::includes(b.begin(), b.end(), a.begin(), a.end(), compare_);
  }

  bool is_superset(const Domain& a, const Domain& b) {
    return std::includes(a.begin(), a.end(), b.begin(), b.end(), compare_);
  }

bool Domain::contains(Arg x) const {
  return std::binary_search(begin(), end(), x);
}

};

// /**
//  * Returns the size of the intersection of two sets with given handles.
//  */
// size_t intersection_size(Handle a, Handle b) {
//   counting_output_iterator out;
//   const vector_type& veca = *sets_.at(a);
//   const vector_type& vecb = *sets_.at(b);
//   return std::set_intersection(veca.begin(), veca.end(),
//                                 vecb.begin(), vecb.end(),
//                                 out).count();
// }

// /**
//  * Returns the sorted intersection of two sets with given handles.
//  */
// Range intersection(Handle a, Handle b) {
//   Range result;
//   const vector_type& veca = *sets_.at(a);
//   const vector_type& vecb = *sets_.at(b);
//   std::set_intersection(veca.begin(), veca.end(),
//                         vecb.begin(), vecb.end(),
//                         std::inserter(result, result.end()));
//   return result;
// }


}
