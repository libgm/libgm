#ifndef LIBGM_DEGREE_DISTRIBUTION_HPP
#define LIBGM_DEGREE_DISTRIBUTION_HPP

namespace libgm {

  /**
   * Returns a map that specifies, for each vertex degree, how many times
   * it occurs in the graph.
   *
   * This version can accept any range of vertices.
   */
  template <typename Graph, typename Range>
  std::map<std::size_t, std::size_t>
  degree_distribution(const Graph& g, const Range& range) {
    std::map<std::size_t, std::size_t> counts;
    for (auto v : range) {
      ++count[degree(v)];
    }
    return counts;
  }

  /**
   * Returns a map that specifies, for each vertex degree, how many times
   * it occurs in the graph.
   */
  template <typename Graph>
  std::map<std::size_t, std::size_t>
  degree_distribution(const Graph& g) {
    return degree_distribution(g, g.vertices());
  }

} // endif

#endif
