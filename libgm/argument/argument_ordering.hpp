#include <libgm/argument/domain.hpp>

namespace libgm {

class ArgumentOrdering : public Object {
public:
  /**
   * Returns true if all the arguments of the domains follow the ordering.
   * This operation has a linear time complexity, O(|a|).
   */
  bool is_sorted(const Domain& a) const;

  /**
   * Returns true if all the arguments of the first domain are also
   * present in the second domain.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  bool is_subset(const Domain& a, const Domain& b) const;

  /**
   * Returns true if all the arguments of the second domain are also
   * present in the first domain.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  bool is_superset(const Domain& a, const Domain& b) const;

  /**
   * Returns true if two domains do not have any arguments in common.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  bool are_disjoint(const Domain& a, const Domain& b) const;

  /**
   * Returns true if an argument is present in the domain.
   * This operation has a logarithmic time complexity.
   */
  bool contains(Arg x, const Domain& a) const;

  /**
   * Returns the ordered intersection of two domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  Domain intersect(const Domain& a, const Domain& b) const;

  /**
   * Returns the union of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  Domain union_(const Domain& a, const Domaun& b) const;

  /**
   * Returns the difference of two ordered domains.
   * This operation has a linear time complexity, O(|a| + |b|).
   */
  Domain difference(const Domain& a, const Domain& b) const;

private:
  struct Impl;
};

std::shared_ptr<ArgumentOrdering> alphabetical_ordering();
std::shared_ptr<ArgumentOrdering> prioritized_ordering();

} // namespace libgm
