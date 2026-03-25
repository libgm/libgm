#pragma once

#include <boost/functional/hash.hpp>

#include <iosfwd>
#include <utility>

namespace libgm {

/**
 * An edge of a bipartite graph, represented as the source and target vertex.
 *
 * \ingroup graph_types
 */
template <typename Source, typename Target>
class BipartiteEdge {
public:
  /// Constructs an empty edge with null source and target.
  BipartiteEdge()
    : source_(), target_() {}

  /// Constructs a special "root" edge with empty source and given target.
  explicit BipartiteEdge(Target target)
    : source_(), target_(target) {}

  /// Constructs a standard edge.
  BipartiteEdge(Source source, Target target)
    : source_(source), target_(target) {}

  /// Conversion to bool indicating if this edge is empty.
  explicit operator bool() const {
    return source_ != Source() && target_ != Target();
  }

  /// Returns the source vertex.
  Source source() const {
    return source_;
  }

  /// Returns the target vertex.
  Target target() const {
    return target_;
  }

  /// Returns the pair consisting of source and target vertex.
  std::pair<Source, Target> pair() const {
    return {source_, target_};
  }

  /// Returns the pair consisting of target and source vertex.
  std::pair<Target, Source> reverse_pair() const {
    return {target_, source_};
  }

  /// Returns true if two edges have the same source and target.
  friend bool operator==(const BipartiteEdge& a, const BipartiteEdge& b) {
    return a.pair() == b.pair();
  }

  /// Returns true if two edges do not have the same source or target.
  friend bool operator!=(const BipartiteEdge& a, const BipartiteEdge& b) {
    return a.pair() != b.pair();
  }

  /// Compares two bipartite edges.
  friend bool operator<=(const BipartiteEdge& a, const BipartiteEdge& b) {
    return a.pair() <= b.pair();
  }

  /// Compares two bipartite edges.
  friend bool operator>=(const BipartiteEdge& a, const BipartiteEdge& b) {
    return a.pair() >= b.pair();
  }

  /// Compares two bipartite edges.
  friend bool operator<(const BipartiteEdge& a, const BipartiteEdge& b) {
    return a.pair() < b.pair();
  }

  /// Compares two bipartite edges.
  friend bool operator>(const BipartiteEdge& a, const BipartiteEdge& b) {
    return a.pair() > b.pair();
  }

  /// Prints the edge to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const BipartiteEdge& e) {
    out << e.source() << " --> " << e.target();
    return out;
  }

private:
  /// Vertex from which the edge originates.
  Source source_;

  /// Vertex to which the edge emanates.
  Target target_;
};

} // namespace libgm


namespace std {

/// \relates BipartiteEdge
template <typename Source, typename Target>
struct hash<libgm::BipartiteEdge<Source, Target>> {
  size_t operator()(const libgm::BipartiteEdge<Source, Target>& e) const {
    size_t seed = 0;
    boost::hash_combine(seed, e.source());
    boost::hash_combine(seed, e.target());
    return seed;
  }
};

} // namespace std
