#ifndef LIBGM_MEAN_FIELD_BIPARTITE_HPP
#define LIBGM_MEAN_FIELD_BIPARTITE_HPP

#include <libgm/factor/utility/traits.hpp>
#include <libgm/graph/bipartite_graph.hpp>
#include <libgm/parallel/vector_processor.hpp>

#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace libgm {

/**
 * A class that runs the mean field algorithm for a factor graph.
 * The computation is performed synchronously, first for all
 * arguments and then for all factors. The number of worker
 * threads is controlled by a parameter to the constructor.
 */
class MeanFieldBipartite {
public:
  // Factor types
  struct NodePotential {

  };

  struct EdgePotential {

  };

  struct Belief {

  };

  using graph_type = FactorGraph<NodePotential, EdgePotential>;

  /**
   * Creates a mean field engine for the given graph.
   * The graph vertices must not change after initialization
   * (the potentials may).
   *
   * \param num_threads the number of worker threads
   */
  explicit MeanFieldBipartite(const graph_type* graph, size_t nthreads = 1)
    : graph_(*graph), nthreads_(nthreads) {
    beliefs1_.reserve(graph_.num_vertices1());
    beliefs2_.reserve(graph_.num_vertices2());

    for (Vertex1 v : graph_.vertices1()) {
      vertices1_.push_back(v);
      beliefs1_[v] = belief_type(NodeF::shape(v), real_type(1));
    }

    for (Vertex2 v : graph_.vertices2()) {
      vertices2_.push_back(v);
      beliefs2_[v] = belief_type(NodeF::shape(v), real_type(1));
    }
  }

  /**
   * Performs a single iteration of mean field.
   */
  Real iterate() {
    Real diff1 = update_all(vertices1_);
    real_type diff2 = update_all(vertices2_);
    return (diff1 + diff2) / graph_.num_vertices();
  }

  /**
   * Returns the belief for a type-1 vertex.
   */
  const belief_type& belief(Vertex1 v) const {
    return beliefs1_.at(v);
  }

  /**
   * Returns the belief for a type-2 vertex.
   */
  const belief_type& belief(Vertex2 v) const {
    return beliefs2_.at(v);
  }

  // Private members
  //==========================================================================
private:
  /**
   * Updates the given range of vertices and returns the sum of the
   * factor differences.
   * \tparam Vertex the vertex type
   */
  template <typename Vertex>
  real_type update_all(std::vector<Vertex>& vertices) {
    std::vector<real_type> sums(nthreads_, real_type(0));
    vector_processor<Vertex, real_type> process([&](Vertex v, real_type& sum){
        sum += update(v);
      });
    process(vertices, sums);
    return std::accumulate(sums.begin(), sums.end(), real_type(0));
  }

  /**
   * Updates a single vertex.
   * \tparam Vertex the vertex type
   */
  template <typename Vertex>
  real_type update(Vertex v) {
    NodeF result = graph_[v];
    for (bipartite_edge<Vertex1, Vertex2> e : graph_.in_edges(v)) {
      if (e.forward()) {
        result *= graph_[e].head().expected_log(belief(e.v1()));
      } else {
        result *= graph_[e].tail().expected_log(belief(e.v2()));
      }
    }
    result /= result.max();
    belief_type new_belief = result.probability();
    new_belief.normalize();
    swap(const_cast<belief_type&>(belief(v)), new_belief);
    return sum_diff(new_belief, belief(v));
  }

  /// The underlying graphical model.
  const graph_type& graph_;

  /// The number of worker threads.
  size_t nthreads_;

  /// A map of current beliefs for arguments
  ankerl::unordered_dense::map<Arg, NodeBelief> node_beliefs_;

  /// A map of current beliefs for factors
  std::vector<EdgeBelief> edge_beliefs;

};

} // namespace libgm

#endif
