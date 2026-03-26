#define BOOST_TEST_MODULE pairwise_bp
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>
#include <libgm/inference/loopy/pairwise_bp.hpp>
#include <libgm/inference/loopy/pairwise_bp_schedules.hpp>
#include <libgm/math/generator/matrix_generator.hpp>
#include <libgm/math/generator/vector_generator.hpp>

#include <random>

using namespace libgm;

namespace {

using ProbMatrix = ProbabilityMatrix<double>;
using ProbTable = ProbabilityTable<double>;
using ProbVector = ProbabilityVector<double>;
using BP = PairwiseBeliefPropagation<ProbVector, ProbMatrix>;

void test(
    PairwiseBeliefSchedule<ProbVector>& schedule,
    BP& engine,
    const MarkovNetworkT<ProbVector, ProbMatrix>& mn,
    const SumProductCalibrate<ProbTable>& sp,
    size_t niters,
    double residual_error,
    double node_error,
    double edge_error,
    double consistency_error) {
  double residual = 0.0;
  for (size_t i = 0; i < niters; ++i) {
    residual = schedule.iterate();
  }
  BOOST_CHECK_LT(residual, residual_error);

  double node_diff = 0.0;
  for (Arg u : mn.vertices()) {
    ProbVector exact = sp.belief(Domain{u}).vector();
    node_diff += engine.belief(u).sum_diff(exact);
  }
  BOOST_CHECK_LT(node_diff / mn.num_vertices(), node_error);

  double edge_diff = 0.0;
  for (Arg u : mn.vertices()) {
    for (auto e : mn.out_edges(u)) {
      if (e.is_nominal()) {
        ProbMatrix exact = sp.belief(Domain{e.source(), e.target()}).matrix();
        edge_diff += engine.belief(e).sum_diff(exact);
      }
    }
  }
  BOOST_CHECK_LT(edge_diff / mn.num_edges(), edge_error);

  double consistency_diff = 0.0;
  size_t consistency_terms = 0;
  for (Arg u : mn.vertices()) {
    ProbVector node_belief = engine.belief(u);
    for (auto e : mn.in_edges(u)) {
      ProbVector edge_belief = engine.belief(e).marginal_back(1);
      consistency_diff += node_belief.sum_diff(edge_belief);
      ++consistency_terms;
    }
  }
  BOOST_CHECK_LT(consistency_diff / consistency_terms, consistency_error);
}

} // namespace

BOOST_AUTO_TEST_CASE(test_convergence) {
  constexpr size_t rows = 5;
  constexpr size_t cols = 4;

  std::mt19937 rng(0);
  ShapeMap shape_map = [](Arg) { return size_t(2); };
  MinFillStrategy strategy;

  auto mn = make_grid_graph<ProbVector, ProbMatrix>(rows, cols, make_argument);
  SumProductCalibrate<ProbTable> sp;
  sp.reset(mn.without_properties(), strategy, shape_map);

  UniformVectorGenerator<double> unary_gen(0.1, 1.0);
  mn.init_vertices([&](Arg arg) {
    ProbVector factor(unary_gen(2, rng));
    sp.multiply_in({arg}, factor.table());
    return factor;
  });

  DiagonalMatrixGenerator<std::uniform_real_distribution<double>> pairwise_gen(0.1, 0.2, 1.0);
  mn.init_edges([&](UndirectedEdge<Arg> e) {
    ProbMatrix factor(pairwise_gen(2, rng));
    sp.multiply_in({e.source(), e.target()}, factor.table());
    return factor;
  });

  sp.calibrate();
  sp.normalize();

  BeliefUpdate<ProbVector> update = Assign<>();
  BeliefDiff<ProbVector> diff = std::mem_fn(&ProbVector::sum_diff);

  {
    BP engine(mn);
    SynchronousPropagationSchedule<ProbVector> schedule(engine, update, diff);
    schedule.initialize(shape_map);
    test(schedule, engine, mn, sp, 100, 1e-6, 1e-3, 1e-3, 1e-6);
  }

  {
    BP engine(mn);
    AsynchronousPropagationSchedule<ProbVector> schedule(engine, update, diff);
    schedule.initialize(shape_map);
    test(schedule, engine, mn, sp, 100, 1e-6, 1e-3, 1e-3, 1e-6);
  }

  {
    BP engine(mn);
    ResidualPropagationSchedule<ProbVector> schedule(engine, update, diff);
    schedule.initialize(shape_map);
    test(schedule, engine, mn, sp, rows * cols * 100, 1e-5, 2e-3, 2e-3, 1e-6);
  }
}
