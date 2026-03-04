#define BOOST_TEST_MODULE mixture_em
#include <boost/test/unit_test.hpp>

#include <libgm/learning/parameter/mixture_em.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/learning/dataset/real_dataset.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

namespace libgm {
  template class mixture_em<moment_gaussian<vec> >;
}

using namespace libgm;
typedef moment_gaussian<vec> mgaussian;

BOOST_AUTO_TEST_CASE(test_convergence) {
  using namespace libgm;
  using namespace std;

  std::size_t k = 3;
  std::size_t nsamples = 5000;
  std::size_t niters = 50;

  universe u;
  vec x = vec::continuous(u, "x", 2);

  // construct the ground truth
  mixture<mgaussian> truth({x}, k);
  truth.param(0).mean << -2, 0;
  truth.param(1).mean << 2, -2;
  truth.param(2).mean << 2, 2;
  truth.param(0).cov << 1, 0.5, 0.5, 1;
  truth.param(1).cov << 1, 0.2, 0.2, 1;
  truth.param(2).cov << 1, -0.2, -0.2, 1;
  truth.normalize();
  auto sample = truth.distribution();

  // generate some data
  real_dataset<vec> data({x}, nsamples);
  std::mt19937 rng;
  for (std::size_t i = 0; i < nsamples; ++i) {
    data.insert(sample(rng), 1.0);
  }

  // learn the model
  em_parameters<> param;
  param.verbose = true;
  mixture_em<mgaussian> learner(param);
  mixture<mgaussian> estimate = learner.fit(data, {x}, k);

  // retrieve the indices in the canonical order
  uint_vector indices = {0, 1, 2};
  std::sort(indices.begin(), indices.end(), [&](std::size_t i, std::size_t j) {
      const dense_vector<>& mi = estimate.param(i).mean;
      const dense_vector<>& mj = estimate.param(j).mean;
      return std::make_pair(mi[0], mi[1]) < std::make_pair(mj[0], mj[1]);
    });

  // compute the KL divergence for each component
  for (std::size_t i = 0; i < k; ++i) {
    std::size_t j = indices[i];
    double kl = kl_divergence(truth.param(i), estimate.param(j));
    cout << i << " "
	 << j << ": "
         << estimate.param(j).mean.transpose() << "\t"
	 << kl << std::endl;
    BOOST_CHECK_SMALL(kl, 0.02);
  }
}
