#ifndef LIBGM_TEST_MN_FIXTURE_HPP
#define LIBGM_TEST_MN_FIXTURE_HPP

#include <libgm/factor/probability_table.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/math/generator/table_generator.hpp>

#include <random>

namespace libgm {

struct Fixture {
  using PTable = ProbabilityTable<double>;

  Fixture(size_t rows = 5, size_t cols = 4, unsigned seed = 17)
    : rows(rows),
      cols(cols),
      rng(seed),
      mn(make_grid_graph<PTable, PTable>(rows, cols, make_argument)) {
    UniformTableGenerator<double> unary_gen(0.1, 1.0);
    DiagonalTableGenerator<std::uniform_real_distribution<double>> pairwise_gen(0.1, 0.2, 1.0);

    mn.init_vertices([&](Arg) {
      return PTable(unary_gen(Shape{2}, rng));
    });
    mn.init_edges([&](UndirectedEdge<Arg>) {
      return PTable(pairwise_gen(2, 2, rng));
    });
  }

  size_t rows;
  size_t cols;
  std::mt19937 rng;
  ShapeMap shape_map = [](Arg) { return size_t(2); };
  MarkovNetworkT<PTable, PTable> mn;
};

} // namespace libgm

#endif
