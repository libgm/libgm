#pragma once

#include <libgm/argument/grid_argument.hpp>
#include <libgm/model/markov_network.hpp>

#include <cstddef>

namespace libgm {

template <typename VP, typename EP = VP>
MarkovNetwork<GridArg, VP, EP> make_grid_graph(size_t rows, size_t cols) {
  MarkovNetwork<GridArg, VP, EP> graph(rows * cols);

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      GridArg u{row, col};
      graph.add_vertex(u);

      if (row > 0) {
        graph.add_edge(GridArg{row - 1, col}, u);
      }
      if (col > 0) {
        graph.add_edge(GridArg{row, col - 1}, u);
      }
    }
  }

  return graph;
}

} // namespace libgm
