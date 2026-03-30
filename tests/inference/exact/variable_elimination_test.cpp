#define BOOST_TEST_MODULE variable_elimination
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/model/factor_graph.hpp>
#include <libgm/inference/exact/variable_elimination.hpp>
#include "mn_fixture.hpp"

namespace libgm {

BOOST_FIXTURE_TEST_CASE(test_grid, Fixture) {
  FactorGraph<PTable, PTable> fg(mn, [](auto&& factor) { return factor.table(); });
  MinFillStrategy strategy;
  SumProduct<PTable> semiring;
  VariableElimination<PTable> ve(shape_map, min_fill, sum_product);
  Domain arguments = mn.vertices();
  arguments.sort();

  auto joint = ve.join(fg, arguments);

  for (Arg u : mn.vertices()) {
    for (UndirectedEdge<Arg> e : mn.out_edges(u)) {
      if (!e.is_nominal()) {
        continue;
      }

      Domain retain{e.source(), e.target()};
      BOOST_CHECK(retain.is_sorted());

      FactorGraph<PTable, PTable> reduced = fg;
      PTable via_elimination = ve.eliminate_join(reduced, retain);
      PTable direct = joint.marginal_dims(arguments.dims(retain));
      BOOST_CHECK_SMALL(max_diff(via_elimination, direct), 1e-8);
    }
  }
}

} // namespace libgm
