#define BOOST_TEST_MODULE belief_update_calibrate
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/graph/factor_graph.hpp>
#include <libgm/inference/exact/belief_update_calibrate.hpp>
#include <libgm/inference/exact/variable_elimination.hpp>
#include "mn_fixture.hpp"

namespace libgm {
namespace {

using PTable = Fixture::PTable;

} // namespace

BOOST_FIXTURE_TEST_CASE(test_calibrate, Fixture) {
  FactorGraphT<int, PTable> fg(mn);

  MinFillStrategy strategy;

  BeliefUpdateCalibrate<PTable> engine;
  engine.reset(mn, strategy, shape_map);
  for (FactorGraph::Factor* f : fg.factors()) {
    engine.multiply_in(fg.arguments(f), fg[f]);
  }
  engine.calibrate();

  VariableElimination<PTable> ve;
  SumProduct<PTable> semiring;

  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());

    FactorGraphT<int, PTable> reduced = fg;
    ve.eliminate(reduced, retain, shape_map, strategy, semiring);
    auto expected_joint = ve.combine_all(reduced, shape_map, semiring);
    PTable expected = expected_joint.factor.marginal_dims(expected_joint.domain.dims(retain));

    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }

  engine.normalize();
  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());

    FactorGraphT<int, PTable> reduced = fg;
    ve.eliminate(reduced, retain, shape_map, strategy, semiring);
    auto expected_joint = ve.combine_all(reduced, shape_map, semiring);
    PTable expected = expected_joint.factor.marginal_dims(expected_joint.domain.dims(retain));
    expected.normalize();

    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }
}

} // namespace libgm
