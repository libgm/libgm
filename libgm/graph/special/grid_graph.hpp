#ifndef LIBGM_GRID_GRAPHS_HPP
#define LIBGM_GRID_GRAPHS_HPP

#include <armadillo>

#include <libgm/graph/undirected_graph.hpp>

namespace libgm {

  //! Creates a grid graph for a sequence of vertices {1, ..., m*n}
  //! \ingroup graph_special
  template <typename VP, typename EP>
  arma::umat
  make_grid_graph(std::size_t m, std::size_t n,
                  undirected_graph<std::size_t, VP, EP>& g) {
    std::size_t ind = 1;

    // create the vertices
    arma::umat vertex(m, n);
    for (std::size_t j = 0; j < n; j++) {
      for (std::size_t i = 0; i < m; i++) {
        vertex(i,j) = ind; g.add_vertex(ind++);
      }
    }

    // create the edges
    for (std::size_t i = 0; i < m; i++) {
      for (std::size_t j = 0; j < n; j++) {
        if (j < n-1) g.add_edge(vertex(i,j), vertex(i,j+1));
        if (i < m-1) g.add_edge(vertex(i,j), vertex(i+1,j));
      }
    }

    return vertex;
  }

  //! Creates a grid graph and a map of the corresponding vertices
  //! \ingroup graph_special
  template <typename Vertex, typename VP, typename EP>
  arma::field<Vertex>
  make_grid_graph(const std::vector<Vertex>& vertices,
                  std::size_t m, std::size_t n,
                  undirected_graph<Vertex, VP, EP>& g) {
    assert(vertices.size() == m*n);

    // create the vertices
    std::size_t k = 0;
    arma::field<Vertex> vertex(m, n);
    for (std::size_t j = 0; j < n; j++) {
      for (std::size_t i = 0; i < m; i++) {
        g.add_vertex(vertices[k]);
        vertex(i,j) = vertices[k];
        ++k;
      }
    }

    // create the edges
    for (std::size_t i = 0; i < m; i++) {
      for(std::size_t j = 0; j < n; j++) {
        if (j < n-1) g.add_edge(vertex(i,j), vertex(i,j+1));
        if (i < m-1) g.add_edge(vertex(i,j), vertex(i+1,j));
      }
    }

    return vertex;
  }

} // namespace libgm

#endif
