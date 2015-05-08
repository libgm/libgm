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
  sum_product_calibrate<ptable> engine(mn);

  // check if clique marginals are correct
  engine.calibrate();
  for (std::size_t v : engine.jt().vertices()) {
    check_belief(engine.belief(v), 1e-8);
  }

  // check if clique marginals are correct after normalization
  engine.normalize();
  for (std::size_t v : engine.jt().vertices()) {
    check_belief_normalized(engine.belief(v), 1e-8);
  }

  // check if all the MN edges are present in the junction tree
  for (auto e : mn.edges()) {
    BOOST_CHECK(!engine.belief({e.source(), e.target()}).empty());
  }

  // condition on an assignment
  finite_assignment<> a = {{vars[6], 0}, {vars[15], 1}, {vars[16], 0}};
  engine.condition(a);
  engine.calibrate();
  engine.normalize();
  mn.condition(a);
  for (std::size_t v : engine.jt().vertices()) {
    check_belief_normalized(engine.belief(v), 1e-10);
  }
}
