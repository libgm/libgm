#define BOOST_TEST_MODULE variable_elimination
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/graph/factor_graph.hpp>
#include <libgm/inference/exact/variable_elimination.hpp>
#include "mn_fixture.hpp"

namespace libgm {
namespace {

using PTable = Fixture::PTable;

} // namespace

BOOST_FIXTURE_TEST_CASE(test_grid, Fixture) {
  FactorGraphT<int, PTable> fg(mn);
  MinFillStrategy strategy;
  SumProduct<PTable> semiring;
  VariableElimination<PTable> ve;

  auto joint = ve.combine_all(fg, shape_map, semiring);

  for (Arg u : mn.vertices()) {
    for (UndirectedEdge<Arg> e : mn.out_edges(u)) {
      if (e.source() > e.target()) {
        continue;
      }

      Domain retain{e.source(), e.target()};
      BOOST_CHECK(retain.is_sorted());

      FactorGraphT<int, PTable> reduced = fg;
      ve.eliminate(reduced, retain, shape_map, strategy, semiring);
      auto eliminated = ve.combine_all(reduced, shape_map, semiring);

      PTable direct = joint.factor.marginal_dims(joint.domain.dims(retain));
      PTable via_elimination = eliminated.factor.marginal_dims(eliminated.domain.dims(retain));
      BOOST_CHECK_SMALL(max_diff(via_elimination, direct), 1e-8);
    }
  }
}

} // namespace libgm
