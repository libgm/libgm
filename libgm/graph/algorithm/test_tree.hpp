#ifndef LIBGM_TEST_TREE_HPP
#define LIBGM_TEST_TREE_HPP

#include <libgm/functional/algorithm.hpp>

#include <functional>
#include <queue>
#include <unordered_set>

namespace libgm {

  /**
   * Tests if the connected component containing the root forms a tree.
   *
   * \tparam Graph an undirected graph type
   * \param pred a function returns false if the given edge is to be ignored.
   *             If null, no edges are ignored.
   * \return the number of reachable nodes or 0
   *         if the connected component is not a tree.
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename Predicate = constant<bool, true> >
  std::size_t
  test_tree(const Graph& g,
            typename Graph::vertex_type root,
            Predicate pred = Predicate()) {
    typedef typename Graph::vertex_type vertex_type;
    typedef typename Graph::edge_type edge_type;
    std::unordered_set<vertex_type> visited;
    visited.insert(root);
    std::queue<edge_type> q;
    q.push(edge_type(root));
    while (!q.empty()) {
      edge_type in = q.front();
      q.pop();
      for (edge_type out : g.out_edges(in.target())) {
        vertex_type v = out.target();
        if (v != in.source() && pred(out)) {
          if (visited.count(v)) { return 0; }
          visited.insert(v);
          q.push(out);
        }
      }
    }
    return visited.size();
  }

} // namespace libgm

#endif
