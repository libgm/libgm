#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/graph/markov_network.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace libgm {

Arg make_argument(size_t row, size_t col);

template <typename VP, typename EP = VP>
MarkovNetworkT<VP, EP> make_grid_graph(
    size_t rows,
    size_t cols,
    const std::function<Arg(size_t, size_t)>& make_argument) {
  MarkovNetworkT<VP, EP> graph(rows * cols);
  std::vector<Arg> args(rows * cols);

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      const size_t idx = row * cols + col;
      Arg u = make_argument(row, col);
      args[idx] = u;
      graph.add_vertex(u);

      if (row > 0) {
        graph.add_edge(u, args[(row - 1) * cols + col]);
      }
      if (col > 0) {
        graph.add_edge(u, args[idx - 1]);
      }
    }
  }

  return graph;
}

} // namespace libgm
