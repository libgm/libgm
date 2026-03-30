#pragma once

#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/inference/exact/junction_tree_engine.hpp>
#include <libgm/inference/exact/variable_elimination.hpp>
#include <libgm/math/generator/matrix_generator.hpp>
#include <libgm/math/generator/vector_generator.hpp>

#include <random>

namespace libgm {

struct Fixture {
  using PTable = ProbabilityTable<double>;
  using PMatrix = ProbabilityMatrix<double>;
  using PVector = ProbabilityVector<double>;
  using Factor = typename FactorGraph<PTable, PTable>::Factor;

  Fixture(size_t rows = 5, size_t cols = 4, unsigned seed = 17)
    : rows(rows),
      cols(cols),
      rng(seed),
      mn(make_grid_graph<PVector, PMatrix>(rows, cols, make_argument)) {
    UniformVectorGenerator<double> unary_gen(0.1, 1.0);
    DiagonalMatrixGenerator<std::uniform_real_distribution<double>> binary_gen(0.1, 0.2, 1.0);
    mn.init_vertices([&](Arg) { return unary_gen(2, rng); });
    mn.init_edges([&](UndirectedEdge<Arg>) { return binary_gen(2, rng); });
  }

  void init_engine(JunctionTreeEngine<PTable>& engine) {
    engine.reset(mn.without_properties(), min_fill, shape_map);
    for (Arg arg : mn.vertices()) {
      // TODO: multiply by unary factor directly without constructing a temporary table
      engine.multiply_in({arg}, mn[arg].table());
      for (UndirectedEdge<Arg> e : mn.out_edges(arg)) {
        if (e.is_nominal()) {
          engine.multiply_in({e.source(), e.target()}, mn[e].table());
        }
      }
    }
  }

  PTable expected_belief(const Domain& retain, bool normalize) const {
    FactorGraph<PTable, PTable> fg(mn, [](auto&& factor) { return factor.table(); });
    VariableElimination<PTable> ve(shape_map, min_fill, sum_product);
    PTable expected = ve.eliminate_join(fg, retain);
    if (normalize) {
      expected.normalize();
    }
    return expected;
  }

  size_t rows;
  size_t cols;
  std::mt19937 rng;
  ShapeMap shape_map = [](Arg) { return size_t(2); };
  SumProduct<PTable> sum_product;
  MinFillStrategy min_fill;
  MarkovNetworkT<PVector, PMatrix> mn;
  DiscreteAssignment evidence = {
    {make_argument(1, 1), 0},
    {make_argument(0, 3), 1},
    {make_argument(1, 3), 0},
  };
};

} // namespace libgm
