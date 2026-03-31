#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/graph/vector_graph.hpp>

#include <vector>

namespace libgm {

/**
 * A vector-backed Markov graph that associates each vertex index with an
 * argument.
 */
class MarkovStructure : public VectorGraph {
public:
  /// Adds a vertex for the given argument and returns its descriptor.
  size_t add_vertex(Arg u) {
    arguments_.push_back(u);
    return VectorGraph::add_vertex();
  }

  /// Returns the argument associated with the given vertex.
  Arg argument(size_t vertex) const {
    return arguments_.at(vertex);
  }

  /// Returns the arguments adjacent to the given vertex.
  Domain adjacent_arguments(size_t vertex) const {
    Domain result;
    for (size_t neighbor : adjacent_vertices(vertex)) {
      result.push_back(argument(neighbor));
    }
    return result;
  }

private:
  std::vector<Arg> arguments_;
};

} // namespace libgm
