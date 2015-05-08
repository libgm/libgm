#ifndef LIBGM_MIN_SPANNING_TREE_HPP
#define LIBGM_MIN_SPANNING_TREE_HPP

#include <libgm/graph/algorithm/functor_property_map.hpp>
#include <libgm/graph/algorithm/vertex_index_map.hpp>
#include <libgm/graph/boost_graph_helpers.hpp>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

namespace libgm {

  /**
   * Kruskal Minimum Spanning Tree (MST) algorithm.
   *
   * \param graph the underlying undirected graph
   * \param weightfn a function object that maps edges to weights
   * \param out an output iterator to which the edges are stored
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename UnaryFn, typename OutIt>
  void kruskal_minimum_spanning_tree(const Graph& graph,
                                     UnaryFn weightfn,
                                     OutIt out) {
    typedef typename Graph::edge_type edge_type;
    boost::kruskal_minimum_spanning_tree(
      graph,
      out,
      boost::vertex_index_map(vertex_index_map<Graph>(graph)).
      weight_map(make_functor_property_map<edge_type>(weightfn)));
  }

} // namespace libgm

#endif
