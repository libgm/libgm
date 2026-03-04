#define BOOST_TEST_MODULE belief_update_calibrate
#include <boost/test/unit_test.hpp>

#include <libgm/inference/exact/belief_update_calibrate.hpp>

#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>

namespace libgm {
  template class belief_update_calibrate<probability_table<var> >;
  template class belief_update_calibrate<canonical_gaussian<var> >;
}

#include "mn_fixture.hpp"

BOOST_FIXTURE_TEST_CASE(test_calibrate, fixture) {
  using libgm::id_t;
  belief_update_calibrate<ptable> engine(mn);

  // check if clique marginals are correct
  engine.calibrate();
  for (id_t v : engine.jt().vertices()) {
    check_belief(engine.belief(v), 1e-8);
  }

  // check if clique marginals are correct after normalization
  engine.normalize();
  for (id_t v : engine.jt().vertices()) {
    check_belief_normalized(engine.belief(v), 1e-8);
  }
}
