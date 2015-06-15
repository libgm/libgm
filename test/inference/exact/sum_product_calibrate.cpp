#define BOOST_TEST_MODULE sum_product_calibrate
#include <boost/test/unit_test.hpp>

#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>

namespace libgm {
  template class sum_product_calibrate<ptable>;
  template class sum_product_calibrate<cgaussian>;
}

#include "mn_fixture.hpp"

BOOST_FIXTURE_TEST_CASE(test_calibrate, fixture) {
  using libgm::id_t;
  sum_product_calibrate<ptable_type> engine(mn);

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

  // check if all the MN edges are present in the junction tree
  for (auto e : mn.edges()) {
    BOOST_CHECK(!engine.belief({e.source(), e.target()}).empty());
  }

  // condition on an assignment
  uint_assignment<std::pair<int, int>> a = {
    {{1, 1}, 0}, {{0, 3}, 1}, {{1, 3}, 0}
  };
  engine.condition(a);
  engine.calibrate();
  engine.normalize();
  mn.condition(a);
  for (id_t v : engine.jt().vertices()) {
    check_belief_normalized(engine.belief(v), 1e-10);
  }
}
