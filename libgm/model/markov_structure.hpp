#pragma once

#include <libgm/argument/concepts/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/graph/vector_graph.hpp>

#include <vector>

namespace libgm {

/**
 * A vector-backed Markov graph that associates each vertex index with an
 * argument.
 */
template <Argument Arg>
class MarkovStructure : public VectorGraph {
public:
  using argument_type = Arg;
  using domain_type = Domain<Arg>;

  size_t add_vertex(Arg u) {
    arguments_.push_back(u);
    return VectorGraph::add_vertex();
  }

  Arg argument(size_t vertex) const {
    return arguments_.at(vertex);
  }

  domain_type adjacent_arguments(size_t vertex) const {
    domain_type result;
    for (size_t neighbor : adjacent_vertices(vertex)) {
      result.push_back(argument(neighbor));
    }
    return result;
  }

private:
  std::vector<Arg> arguments_;
};

} // namespace libgm
