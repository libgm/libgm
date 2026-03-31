#define BOOST_TEST_MODULE sum_product_calibrate
#include <boost/test/unit_test.hpp>

#include <libgm/model/factor_graph.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>
#include "mn_fixture.hpp"

namespace libgm {

BOOST_FIXTURE_TEST_CASE(test_calibrate, Fixture) {
  SumProductCalibrate<PTable> engine;
  init_engine(engine);
  engine.calibrate();
  for (ClusterGraph<>::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    PTable expected = expected_belief(retain, false);
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }

  engine.normalize();

  for (ClusterGraph<>::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    PTable expected = expected_belief(retain, true);
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }

  for (auto e : mn.edges()) {
    Domain retain = mn.domain(e);
    BOOST_CHECK(retain.is_sorted());
    PTable belief = engine.belief(retain);
    BOOST_CHECK(!belief.param().empty());
  }
}

BOOST_FIXTURE_TEST_CASE(test_conditioning, Fixture) {
  SumProductCalibrate<PTable> engine;
  init_engine(engine);
  engine.condition(evidence);
  engine.calibrate();
  engine.normalize();

  for (auto [arg, value] : evidence) {
    mn[arg].param()[1 - value] = 0.0;
  }

  for (ClusterGraph<>::vertex_descriptor v : engine.jt().vertices()) {
    Domain retain = engine.jt().cluster(v);
    BOOST_CHECK(retain.is_sorted());
    PTable expected = expected_belief(retain, true);
    BOOST_CHECK_SMALL(max_diff(engine.belief(v), expected), 1e-8);
  }
}

} // namespace libgm
