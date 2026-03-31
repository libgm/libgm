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

namespace {

using Arg = libgm::GridArg;
using Domain = libgm::Domain<Arg>;
using ShapeMap = libgm::ShapeMap<Arg>;
using ProbMatrix = libgm::ProbabilityMatrix<double>;
using ProbTable = libgm::ProbabilityTable<double>;
using ProbVector = libgm::ProbabilityVector<double>;
using BP = libgm::PairwiseBeliefPropagation<Arg, ProbVector, ProbMatrix>;

void test(
    libgm::PairwiseBeliefSchedule<Arg, ProbVector>& schedule,
    BP& engine,
    const libgm::MarkovNetwork<Arg, ProbVector, ProbMatrix>& mn,
    const libgm::SumProductCalibrate<Arg, ProbTable>& sp,
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
  for (auto* v : mn.vertices()) {
    Arg u = mn.argument(v);
    ProbVector exact = sp.belief(Domain{u}).vector();
    node_diff += engine.belief(u).sum_diff(exact);
  }
  BOOST_CHECK_LT(node_diff / mn.num_vertices(), node_error);

  double edge_diff = 0.0;
  for (auto e : mn.edges()) {
    ProbMatrix exact = sp.belief(mn.domain(e)).matrix();
    edge_diff += engine.belief(e).sum_diff(exact);
  }
  BOOST_CHECK_LT(edge_diff / mn.num_edges(), edge_error);

  double consistency_diff = 0.0;
  size_t consistency_terms = 0;
  for (auto* v : mn.vertices()) {
    Arg u = mn.argument(v);
    ProbVector node_belief = engine.belief(u);
    for (auto e : mn.in_edges(v)) {
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
  libgm::MinFillStrategy strategy;

  auto mn = libgm::make_grid_graph<ProbVector, ProbMatrix>(rows, cols);
  libgm::SumProductCalibrate<Arg, ProbTable> sp;
  sp.reset(mn.structure(), strategy, shape_map);

  libgm::UniformVectorGenerator<double> unary_gen(0.1, 1.0);
  mn.init_vertices([&](Arg arg) {
    ProbVector factor(unary_gen(2, rng));
    sp.multiply_in({arg}, factor.table());
    return factor;
  });

  libgm::DiagonalMatrixGenerator<std::uniform_real_distribution<double>> pairwise_gen(0.1, 0.2, 1.0);
  mn.init_edges([&](auto e) {
    ProbMatrix factor(pairwise_gen(2, rng));
    sp.multiply_in(mn.domain(e), factor.table());
    return factor;
  });

  sp.calibrate();
  sp.normalize();

  libgm::BeliefUpdate<ProbVector> update = libgm::Assign<>();
  libgm::BeliefDiff<ProbVector> diff = std::mem_fn(&ProbVector::sum_diff);

  {
    BP engine(mn);
    libgm::SynchronousPropagationSchedule<Arg, ProbVector> schedule(engine, update, diff);
    schedule.initialize(shape_map);
    test(schedule, engine, mn, sp, 100, 1e-6, 1e-3, 1e-3, 1e-6);
  }

  {
    BP engine(mn);
    libgm::AsynchronousPropagationSchedule<Arg, ProbVector> schedule(engine, update, diff);
    schedule.initialize(shape_map);
    test(schedule, engine, mn, sp, 100, 1e-6, 1e-3, 1e-3, 1e-6);
  }

  {
    BP engine(mn);
    libgm::ResidualPropagationSchedule<Arg, ProbVector> schedule(engine, update, diff);
    schedule.initialize(shape_map);
    test(schedule, engine, mn, sp, rows * cols * 100, 1e-5, 2e-3, 2e-3, 1e-6);
  }
}
