#define BOOST_TEST_MODULE sum_product_calibrate
#include <boost/test/unit_test.hpp>

#include <libgm/assignment/discrete_assignment.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/graph/factor_graph.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>
#include <libgm/inference/exact/variable_elimination.hpp>
#include "mn_fixture.hpp"

namespace libgm {
namespace {

using PTable = Fixture::PTable;

PTable expected_belief(
    const FactorGraphT<int, PTable>& fg,
    const Domain& retain,
    const ShapeMap& shape_map,
    const EliminationStrategy& strategy,
    const SumProduct<PTable>& semiring,
    VariableElimination<PTable>& ve,
    bool normalize) {
  FactorGraphT<int, PTable> reduced = fg;
  ve.eliminate(reduced, retain, shape_map, strategy, semiring);
  auto expected_joint = ve.combine_all(reduced, shape_map, semiring);
  PTable expected = expected_joint.factor.marginal_dims(expected_joint.domain.dims(retain));
  if (normalize) {
    expected.normalize();
  }
  return expected;
}

void add_evidence_factors(FactorGraphT<int, PTable>& fg, const DiscreteAssignment& evidence) {
  for (auto [arg, value] : evidence) {
    PTable indicator(Shape{2}, 0.0);
    indicator.param().data()[value] = 1.0;
    fg.add_factor(Domain{arg}, std::move(indicator));
  }
}

} // namespace

BOOST_FIXTURE_TEST_CASE(test_calibrate, Fixture) {
  FactorGraphT<int, PTable> fg(mn);

  MinFillStrategy strategy;
  SumProduct<PTable> semiring;
  VariableElimination<PTable> ve;

  SumProductCalibrate<PTable> engine;
  engine.reset(mn, strategy, shape_map);
  for (FactorGraph::Factor* f : fg.factors()) {
    engine.multiply_in(fg.arguments(f), fg[f]);
  }

  engine.calibrate();
  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    PTable expected = expected_belief(fg, retain, shape_map, strategy, semiring, ve, false);
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }

  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    std::cout << v << engine.belief(v) << std::endl;
  }

  engine.normalize();
  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    PTable expected = expected_belief(fg, retain, shape_map, strategy, semiring, ve, true);
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }

  for (Arg u : mn.vertices()) {
    for (UndirectedEdge<Arg> e : mn.out_edges(u)) {
      if (e.source() <= e.target()) {
        Domain retain{e.source(), e.target()};
        BOOST_CHECK(retain.is_sorted());
        PTable belief = engine.belief(retain);
        BOOST_CHECK(!belief.param().empty());
      }
    }
  }

  DiscreteAssignment evidence = {
    {make_argument(1, 1), 0},
    {make_argument(0, 3), 1},
    {make_argument(1, 3), 0},
  };

  engine.condition(evidence);
  engine.calibrate();
  engine.normalize();

  FactorGraphT<int, PTable> conditioned_fg = fg;
  add_evidence_factors(conditioned_fg, evidence);

  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    PTable expected = expected_belief(
        conditioned_fg, retain, shape_map, strategy, semiring, ve, true);
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }
}

} // namespace libgm
