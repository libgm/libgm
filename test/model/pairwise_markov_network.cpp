#define BOOST_TEST_MODULE markov_network
#include <boost/test/unit_test.hpp>

#include <libgm/model/pairwise_markov_network.hpp>

#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>

namespace libgm {
  template class pairwise_markov_network<cgaussian>;
  template class pairwise_markov_network<ptable>;
}
