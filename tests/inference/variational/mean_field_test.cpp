#define BOOST_TEST_MODULE mean_field
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/graph/factor_graph.hpp>
#include <libgm/graph/markov_network.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>
#include <libgm/inference/variational/mean_field.hpp>
#include <libgm/math/generator/matrix_generator.hpp>
#include <libgm/math/generator/vector_generator.hpp>

#include <random>
#include <string>
#include <vector>

using namespace libgm;

namespace {

using LogTable = LogarithmicTable<double>;
using LogVector = LogarithmicVector<double>;
using ProbMatrix = ProbabilityMatrix<double>;
using ProbTable = ProbabilityTable<double>;
using ProbVector = ProbabilityVector<double>;

Arg make_arg(const std::string& name) {
  return NamedFactory::default_factory().make(name);
}

} // namespace

BOOST_AUTO_TEST_CASE(test_convergence) {
  constexpr size_t nvertices = 5;
  constexpr size_t nedges = 8;
  constexpr size_t niters = 20;

  std::mt19937 rng(0);
  std::uniform_int_distribution<size_t> side_dist(0, nvertices - 1);
  UniformVectorGenerator<double> unary_gen(0.1, 1.0);
  UniformMatrixGenerator<double> pairwise_gen(0.1, 1.0);
  ShapeMap shape_map = [](Arg) { return size_t(2); };
  MinFillStrategy strategy;
  SumProductCalibrate<ProbTable> sp;

  MarkovNetworkT<ProbVector, ProbMatrix> exact_mn;
  FactorGraphT<LogVector, LogTable> mf_fg;
  std::vector<Arg> xs;
  std::vector<Arg> ys;

  for (size_t i = 0; i < nvertices; ++i) {
    Arg x = make_arg("x" + std::to_string(i));
    Arg y = make_arg("y" + std::to_string(i));
    xs.push_back(x);
    ys.push_back(y);

    ProbVector unary_x(unary_gen(2, rng));
    ProbVector unary_y(unary_gen(2, rng));

    exact_mn.add_vertex(x, unary_x);
    exact_mn.add_vertex(y, unary_y);
    mf_fg.add_argument(x, unary_x.logarithmic());
    mf_fg.add_argument(y, unary_y.logarithmic());
  }

  size_t inserted_edges = 0;
  while (inserted_edges < nedges) {
    size_t i = side_dist(rng);
    size_t j = side_dist(rng);

    Arg x = xs[i];
    Arg y = ys[j];
    ProbMatrix pairwise(pairwise_gen(2, 2, rng));
    Domain domain{x, y};
    auto [e, inserted] = exact_mn.add_edge(x, y, pairwise);
    if (!inserted) {
      continue;
    }
    ++inserted_edges;
    mf_fg.add_factor(domain, pairwise.logarithmic().table());
  }

  sp.reset(exact_mn.without_properties(), strategy, shape_map);
  for (Arg u : exact_mn.vertices()) {
    sp.multiply_in(Domain{u}, exact_mn[u].table());
    for (UndirectedEdge<Arg> e : exact_mn.out_edges(u)) {
      if (e.is_nominal()) {
        sp.multiply_in(Domain{e.source(), e.target()}, exact_mn[e].table());
      }
    }
  }
  sp.calibrate();
  sp.normalize();

  MeanField<LogVector, LogTable> mf(mf_fg, shape_map);
  double diff = 0.0;
  for (size_t it = 0; it < niters; ++it) {
    diff = mf.iterate();
  }
  BOOST_CHECK_LT(diff, 1e-4);

  double kl = 0.0;
  for (Arg u : exact_mn.vertices()) {
    ProbVector exact = sp.belief(Domain{u}).vector();
    kl += exact.kl_divergence(mf.belief(u));
  }
  kl /= exact_mn.num_vertices();
  BOOST_CHECK_LT(kl, 0.02);
}
