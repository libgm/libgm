#ifndef LIBGM_DOMAIN_INDEX_HPP
#define LIBGM_DOMAIN_INDEX_HPP

#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>

#include <algorithm>
#include <functional>
#include <iosfwd>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace libgm {

/**
 * An index over domains that efficiently processes intersection and
 * superset queries. Each domain is associated with a pointer handle
 * that stores that domain.
 */
class DomainIndex {
  // Public types
  //==========================================================================
public:
  /// The handle of (i.e., pointer to) the domain stored in this index.
  using Handle = const Domain*;

  /// An unordered set of handles. // FIXME: consistent hashing?
  using HandleSet = ankerl::unordered_dense::set<Handle>;

  /// Maps each element to the vector of containing pointers.
  using AdjacencyMap = ankerl::unordered_dense::map<Arg, std::vector<Handle>>;

  /// Iterators over arguments contained in this index.
  using argument_iterator = MapKeyIterator<AdjacencyMap>;

  // Constructors and destructors
  //==========================================================================
public:
  /// Default constructor. Creates an empty set index.
  DomainIndex() = default;

  /// Swaps the content of two index sets in constant time.
  friend void swap(DomainIndex& a, DomainIndex& b);

  // Queries
  //==========================================================================

  /// Returns true if the index contains no domains.
  bool empty() const;

  /// Returns the number of domains stored in this index.
  size_t size() const;

  /// Returns the number of arguments stored in this index.
  size_t num_arguments() const;

  /// Returns the range of arguments in this index.
  boost::iterator_range<argument_iterator> arguments() const;

  /// Returns the handle of the first domain stored element.
  Handle front() const;

  /**
   * Returns a handle for any domain that contains the specified argument.
   * \throw std::out_of_range if no there is no domain containing the argument
   */
  Handle operator[](Arg arg) const;

  /**
   * Returns the number of domains that contain an argument.
   */
  size_t count(Arg arg) const;

  /**
   * Visits all domains that include (are superset of) the supplied arguments.
   */
  void visit_covers(const Domain& args, std::function<void(Handle)> visitor) const;

  /**
   * Visits all domains that intersect (meet) the given arguments.
   */
  HandleSet find_intersections(const Domain& args) const;

  /**
   * Maximal intersection query: finds a domain whose intersection with the supplied
   * arguments is maximal and non-zero.
   *
   * \return The handle with a non-zero maximal intersection with the supplied arguments.
   *         Of the handles with the same intersection size, returns the smallest one.
   *         Otherwise, returns front().
   */
  Handle find_max_intersection(const Domain& args) const;

  /**
   * Minimal superset query: find the domain which is a superset of the supplied arguments.
   * If the supplied arguments are empty, this operation may return any set.
   *
   * \return the handle for a minimal superset if a superset exists.
   *         Handle() if no superset exists.
   */
  Handle find_min_cover(const Domain& args) const;

  /**
   * Returns true if there are no (non-strict) supersets of the supplied arguments in this index.
   */
  bool is_maximal(const Domain& args) const;

  /**
   * Prints the index to an output stream.
   */
  friend std::ostream& operator<<(std::ostream& out, const DomainIndex& index);

  // Mutating operations
  //==========================================================================

  /**
   * Inserts a new domain in the index. This function is linear in the number of arguments of the
   * domain. The domain must not be empty, must not be present in the index.
   */
  void insert(Handle handle);

  /// Removes the set with the given handle from the index.
  void erase(Handle handle);

  /// Removes all sets from this index.
  void clear();

  // Private members
  //==========================================================================
private:

  // /// Returns the edges from a single value to the sets it is contained in/
  // const std::unordered_map<Handle, vector_type*>&
  // adjacency(Arg arg) const {
  //   static std::unordered_map<Handle, vector_type*> empty;
  //   auto it = adjacency_.find(arg);
  //   if (it == adjacency_.end()) {
  //     return empty;
  //   } else {
  //     return it->second;
  //   }
  // }

  /// The mapping from arguments to domains.
  AdjacencyMap adjacency_;

  /// The number of domain stpred in this index.
  size_t num_domains_ = 0;

}; // class DomainIndex

} // namespace libgm

#endif
