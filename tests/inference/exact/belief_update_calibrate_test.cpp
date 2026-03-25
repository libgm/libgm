#define BOOST_TEST_MODULE belief_update_calibrate
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_table.hpp>
#include <libgm/graph/factor_graph.hpp>
#include <libgm/inference/exact/belief_update_calibrate_test.hpp>
#include "mn_fixture.hpp"

namespace libgm {

BOOST_FIXTURE_TEST_CASE(test_calibrate, Fixture) {
  BeliefUpdateCalibrate<PTable> engine;
  init_engine(engine);
  engine.calibrate();

  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected_belief(retain, false)), 1e-8);
  }

  engine.normalize();
  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected_belief(retain, true)), 1e-8);
  }
}

BOOST_FIXTURE_TEST_CASE(test_conditioning, Fixture) {
  BeliefUpdateCalibrate<PTable> engine;
  init_engine(engine);
  engine.condition(evidence);
  engine.calibrate();
  engine.normalize();

  for (auto [arg, value] : evidence) {
    mn[arg].param()[1 - value] = 0.0;
  }

  for (ClusterGraph::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    PTable expected = expected_belief(retain, true);
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }
}

}
