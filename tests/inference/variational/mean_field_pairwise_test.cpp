#define BOOST_TEST_MODULE mean_field_pairwise
#include <boost/test/unit_test.hpp>

#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/model/factor_graph.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>
#include <libgm/inference/variational/mean_field_pairwise.hpp>
#include <libgm/math/generator/matrix_generator.hpp>
#include <libgm/math/generator/vector_generator.hpp>

#include <random>

using namespace libgm;

namespace {

using LogVector = LogarithmicVector<double>;
using LogMatrix = LogarithmicMatrix<double>;
using ProbMatrix = ProbabilityMatrix<double>;
using ProbVector = ProbabilityVector<double>;
using ProbTable = ProbabilityTable<double>;

} // namespace

BOOST_AUTO_TEST_CASE(test_convergence) {
  constexpr size_t rows = 8;
  constexpr size_t cols = 5;
  constexpr size_t niters = 20;

  std::mt19937 rng(0);
  ShapeMap shape_map = [](Arg) { return size_t(2); };
  MinFillStrategy strategy;

  auto mn = make_grid_graph<LogVector, LogMatrix>(rows, cols, make_argument);
  SumProductCalibrate<ProbTable> sp;
  sp.reset(mn.structure(), strategy, shape_map);

  UniformVectorGenerator<double> unary_gen(0.1, 1.0);
  mn.init_vertices([&](Arg u) {
    ProbVector factor(unary_gen(2, rng));
    sp.multiply_in(Domain{u}, factor.table());
    return factor.logarithmic();
  });

  DiagonalMatrixGenerator<std::uniform_real_distribution<double>> pairwise_gen(0.1, 0.2, 1.0);
  mn.init_edges([&](auto e) {
    ProbMatrix factor(pairwise_gen(2, rng));
    sp.multiply_in(mn.domain(e), factor.table());
    return factor.logarithmic();
  });

  sp.calibrate();
  sp.normalize();

  MeanFieldPairwise<LogVector, LogMatrix> mf(mn, shape_map);
  double diff = 0.0;
  for (size_t it = 0; it < niters; ++it) {
    diff = mf.iterate();
  }
  BOOST_CHECK_LT(diff, 1e-4);

  double kl = 0.0;
  for (auto* v : mn.vertices()) {
    Arg u = mn.argument(v);
    ProbVector exact = sp.belief(Domain{u}).vector();
    kl += exact.kl_divergence(mf.belief(u));
  }
  kl /= mn.num_vertices();
  BOOST_CHECK_LT(kl, 0.02);
}
