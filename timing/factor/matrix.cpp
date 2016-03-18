#include <libgm/argument/var.hpp>
#include <libgm/factor/probability_array.hpp>
#include <libgm/factor/canonical_array.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>

#include <functional>
#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

const char* version(double) {
  return "probability";
}

const char* version(logd) {
  return "canonical  ";
}

template <typename Matrix, std::size_t N, typename Op>
void time_transform(Op op,
                    const uint_vector& num_values,
                    std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    auto f = tuple_rep<N>(Matrix({x, y}));
    Matrix g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = tuple_apply(op, f);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename JoinOp>
void time_matmat_join(JoinOp join_op,
                      bool transpose,
                      const uint_vector& num_values,
                      std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    array_domain<var, 2> xy = {x, y}, yx = {y, x};
    Matrix f(xy);
    Matrix g(transpose ? yx : xy);
    Matrix h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename JoinOp>
void time_matvec_join(JoinOp join_op,
                      bool transpose,
                      const uint_vector& num_values,
                      std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    Matrix f({x, y});
    Vector g({transpose ? y : x});
    Matrix h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename JoinOp>
void time_vecmat_join(JoinOp join_op,
                      bool transpose,
                      const uint_vector& num_values,
                      std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    array_domain<var, 2> xy = {x, y}, yx = {y, x};
    Vector f({x});
    Matrix g(transpose ? yx : xy);
    Matrix h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename JoinOp>
void time_matvec_outer_join(JoinOp join_op,
                            const uint_vector& num_values,
                            std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    Matrix f({x, y});
    Vector gx({x});
    Vector gy({y});
    Matrix h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(join_op(f, gx), gy);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename AggOp>
void time_aggregate(AggOp agg_op,
                    bool rowwise,
                    const uint_vector& num_values,
                    std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    Matrix f({x, y});
    Vector g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = agg_op(f, array_domain<var, 1>({rowwise ? x : y}));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename Op>
void time_assignment(Op op,
                     const uint_vector& num_values,
                     std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    Matrix f({var::discrete(u, "x", n), var::discrete(u, "y", n)});
    uint_assignment<var> a;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      result_t<Matrix> value = op(f, a);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename JoinOp, typename AggOp>
void time_join_aggregate(JoinOp join_op,
                         AggOp agg_op,
                         bool right,
                         const uint_vector& num_values,
                         std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    Matrix f({x, y});
    Vector g({right ? y : x});
    Vector h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = agg_op(join_op(f, g), array_domain<var, 1>({right ? x : y}));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector>
void time_restrict(bool colwise,
                   const uint_vector& num_values,
                   std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    Matrix f({x, y});
    Vector g;
    uint_assignment<var> a = {{colwise ? y : x, 0}};
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f.restrict(a);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename UpdateOp>
void time_restrict_update(bool colwise,
                          UpdateOp update_op,
                          const uint_vector& num_values,
                          std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Matrix>()) << std::flush;
  for (std::size_t n : num_values) {
    var x = var::discrete(u, "x", n);
    var y = var::discrete(u, "y", n);
    Matrix f({x, y});
    Vector g({colwise ? x : y});
    uint_assignment<var> a = {{colwise ? y : x, 0}};
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      update_op(g, f.restrict(a));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  std::size_t min_values;
  std::size_t step_size;
  std::size_t max_values = 0;
  std::size_t num_reps = 0;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("min-values", po::value<std::size_t>(&min_values)->default_value(2),
     "the initial number of values")
    ("max-values", po::value<std::size_t>(&max_values),
     "the maximum number of values")
    ("step-size", po::value<std::size_t>(&step_size)->default_value(1),
     "the increment in the number of values")
    ("num-reps", po::value<std::size_t>(&num_reps),
     "the number of repetitions");

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  notify(vm);

  if (vm.count("help") || !max_values || !num_reps) {
    std::cout << desc;
    return 0;
  }

  std::cout << "Measuring matrix factors with "
            << " min_values=" << min_values
            << " max_values=" << max_values
            << " step_size=" << step_size
            << " num_reps=" << num_reps
            << std::endl;

  typedef probability_array<var, 1> pvector;
  typedef probability_array<var, 2> pmatrix;
  typedef canonical_array<var, 1> lvector;
  typedef canonical_array<var, 2> lmatrix;
  uint_vector num_values;
  for (std::size_t i = min_values; i <= max_values; i += step_size) {
    num_values.push_back(i);
  }
  std::multiplies<> mult;

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "matrix * constant" << std::endl;
  time_transform<pmatrix, 1>(multiplied_by<double>(2), num_values, num_reps);
  time_transform<lmatrix, 1>(multiplied_by<logd>(logd(2)), num_values, num_reps);

  std::cout << std::endl << "matrix + matrix" << std::endl;
  time_transform<pmatrix, 2>(std::plus<>(), num_values, num_reps);
  time_transform<lmatrix, 2>(std::plus<>(), num_values, num_reps / 10);

  std::cout << std::endl << "matrix * matrix -- direct" << std::endl;
  time_matmat_join<pmatrix>(mult, false, num_values, num_reps);
  time_matmat_join<lmatrix>(mult, false, num_values, num_reps);

  std::cout << std::endl << "matrix * matrix -- transposed" << std::endl;
  time_matmat_join<pmatrix>(mult, true, num_values, num_reps);
  time_matmat_join<lmatrix>(mult, true, num_values, num_reps);

  std::cout << std::endl << "matrix * vector -- direct" << std::endl;
  time_matvec_join<pmatrix, pvector>(mult, false, num_values, num_reps);
  time_matvec_join<lmatrix, lvector>(mult, false, num_values, num_reps);

  std::cout << std::endl << "matrix * vector -- transposed" << std::endl;
  time_matvec_join<pmatrix, pvector>(mult, true, num_values, num_reps);
  time_matvec_join<lmatrix, lvector>(mult, true, num_values, num_reps);

  std::cout << std::endl << "vector * matrix -- direct" << std::endl;
  time_vecmat_join<pmatrix, pvector>(mult, false, num_values, num_reps);
  time_vecmat_join<lmatrix, lvector>(mult, false, num_values, num_reps);

  std::cout << std::endl << "vector * matrix -- transposed" << std::endl;
  time_vecmat_join<pmatrix, pvector>(mult, true, num_values, num_reps);
  time_vecmat_join<lmatrix, lvector>(mult, true, num_values, num_reps);

  std::cout << std::endl << "matrix * vector * vector -- outer" << std::endl;
  time_matvec_outer_join<pmatrix, pvector>(mult, num_values, num_reps);
  time_matvec_outer_join<lmatrix, lvector>(mult, num_values, num_reps);

  std::cout << std::endl << "matrix.marginal(dom) -- row-wise" << std::endl;
  time_aggregate<pmatrix, pvector>(member_marginal(), true, num_values, num_reps);
  time_aggregate<lmatrix, lvector>(member_marginal(), true, num_values, num_reps / 10);

  std::cout << std::endl << "matrix.marginal(dom) -- column-wise" << std::endl;
  time_aggregate<pmatrix, pvector>(member_marginal(), false, num_values, num_reps);
  time_aggregate<lmatrix, lvector>(member_marginal(), false, num_values, num_reps / 10);

  std::cout << std::endl << "matrix.maximum(a)" << std::endl;
  time_assignment<pmatrix, pvector>(member_maximum(), num_values, num_reps);
  time_assignment<lmatrix, lvector>(member_maximum(), num_values, num_reps);

  std::cout << std::endl << "(matrix * vector).marginal(dom) -- right"
            << std::endl;
  time_join_aggregate<pmatrix, pvector>(mult, member_marginal(), true,
                                        num_values, num_reps);
  time_join_aggregate<lmatrix, lvector>(mult, member_marginal(), true,
                                        num_values, num_reps / 10);

  std::cout << std::endl << "(matrix * vector).marginal(dom) -- left"
            << std::endl;
  time_join_aggregate<pmatrix, pvector>(mult, member_marginal(), false,
                                        num_values, num_reps);
  time_join_aggregate<lmatrix, lvector>(mult, member_marginal(), false,
                                        num_values, num_reps / 10);

  std::cout << std::endl << "(matrix * vector).maximum(dom) -- right"
            << std::endl;
  time_join_aggregate<pmatrix, pvector>(mult, member_maximum(), true,
                                        num_values, num_reps);
  time_join_aggregate<lmatrix, lvector>(mult, member_maximum(), true,
                                        num_values, num_reps);

  std::cout << std::endl << "(matrix * vector).maximum(dom) -- left"
            << std::endl;
  time_join_aggregate<pmatrix, pvector>(mult, member_maximum(), false,
                                        num_values, num_reps);
  time_join_aggregate<lmatrix, lvector>(mult, member_maximum(), false,
                                        num_values, num_reps);

  std::cout << std::endl << "matrix.restrict(a) -- row-wise" << std::endl;
  time_restrict<pmatrix, pvector>(false, num_values, num_reps);
  time_restrict<lmatrix, lvector>(false, num_values, num_reps);

  std::cout << std::endl << "matrix.restrict(a) -- column-wise" << std::endl;
  time_restrict<pmatrix, pvector>(true, num_values, num_reps);
  time_restrict<lmatrix, lvector>(true, num_values, num_reps);

  std::cout << std::endl << "vector *= matrix.restrict(a) -- row-wise"
            << std::endl;
  time_restrict_update<pmatrix, pvector>(false, multiplies_assign<>(),
                                         num_values, num_reps);
  time_restrict_update<lmatrix, lvector>(false, multiplies_assign<>(),
                                         num_values, num_reps);

  std::cout << std::endl << "vector *= matrix.restrict(a) -- column-wise"
            << std::endl;
  time_restrict_update<pmatrix, pvector>(true, multiplies_assign<>(),
                                         num_values, num_reps);
  time_restrict_update<lmatrix, lvector>(true, multiplies_assign<>(),
                                         num_values, num_reps);

  return 0;
}
