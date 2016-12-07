#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/utility/traits.hpp>
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

// global options
uint_vector num_values;

const char* version(double) {
  return "probability";
}

const char* version(logd) {
  return "logarithmic";
}

template <typename Matrix, std::size_t N, typename Op>
void time_transform(Op op, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    auto f = tuple_rep<N>(Matrix(n, n));
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
void time_matmat_transpose(JoinOp join_op, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    Matrix f(n, n);
    Matrix g(n, n);
    Matrix h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g.transpose());
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}


template <typename Matrix, typename Vector, typename JoinOp>
void time_matvec_join(JoinOp join_op, std::size_t d, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    Matrix f(n, n);
    Vector g(n);
    Matrix h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f.dim(d), g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename JoinOp>
void time_vecmat_join(JoinOp join_op, std::size_t d, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    Vector f(n);
    Matrix g(n, n);
    Matrix h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g.dim(d));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename JoinOp>
void time_matvec_outer_join(JoinOp join_op, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    Matrix f(n, n);
    Vector gx(n);
    Vector gy(n);
    Matrix h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, outer_prod(gx, gy));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename AggOp>
void time_aggregate(AggOp agg_op, std::size_t retain, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    Matrix f(n, n);
    Vector g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = agg_op(f, retain);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename Op>
auto time_find(Op op, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  typename Matrix::result_type value;
  for (std::size_t n : num_values) {
    Matrix f(n, n);
    std::size_t row, col;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      value *= op(f, row, col);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
  return value;
}

template <typename Matrix, typename Vector, typename JoinOp, typename AggOp>
void time_join_aggregate(JoinOp join_op, AggOp agg_op, std::size_t d,
                         std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    Matrix f(n, n);
    Vector g(n);
    Vector h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = agg_op(join_op(f.dim(d), g), 1-d);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector>
void time_restrict(std::size_t d, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    Matrix f(n, n);
    Vector g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f.restrict(d, n-1);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename UpdateOp>
void time_restrict_update(std::size_t d, UpdateOp update_op,
                          std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (std::size_t n : num_values) {
    Matrix f(n, n);
    Vector g(n);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      update_op(g, f.restrict(d, n-1));
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

  for (std::size_t i = min_values; i <= max_values; i += step_size) {
    num_values.push_back(i);
  }

  typedef experimental::probability_vector<> pvector;
  typedef experimental::probability_matrix<> pmatrix;
  typedef experimental::logarithmic_vector<> lvector;
  typedef experimental::logarithmic_matrix<> lmatrix;
  std::multiplies<> mult;

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "matrix * constant" << std::endl;
  time_transform<pmatrix, 1>(multiplied_by<double>(2), num_reps);
  time_transform<lmatrix, 1>(multiplied_by<logd>(logd(2)), num_reps);

  std::cout << std::endl << "matrix + matrix" << std::endl;
  time_transform<pmatrix, 2>(std::plus<>(), num_reps);
  time_transform<lmatrix, 2>(std::plus<>(), num_reps / 10);

  std::cout << std::endl << "matrix * matrix -- direct" << std::endl;
  time_transform<pmatrix, 2>(mult, num_reps);
  time_transform<lmatrix, 2>(mult, num_reps);

  std::cout << std::endl << "matrix * matrix -- transposed" << std::endl;
  time_matmat_transpose<pmatrix>(mult, num_reps);
  time_matmat_transpose<lmatrix>(mult, num_reps);

  std::cout << std::endl << "matrix * vector -- direct" << std::endl;
  time_matvec_join<pmatrix, pvector>(mult, 0, num_reps);
  time_matvec_join<lmatrix, lvector>(mult, 0, num_reps);

  std::cout << std::endl << "matrix * vector -- transposed" << std::endl;
  time_matvec_join<pmatrix, pvector>(mult, 1, num_reps);
  time_matvec_join<lmatrix, lvector>(mult, 1, num_reps);

  std::cout << std::endl << "vector * matrix -- direct" << std::endl;
  time_vecmat_join<pmatrix, pvector>(mult, 0, num_reps);
  time_vecmat_join<lmatrix, lvector>(mult, 0, num_reps);

  std::cout << std::endl << "vector * matrix -- transposed" << std::endl;
  time_vecmat_join<pmatrix, pvector>(mult, 1, num_reps);
  time_vecmat_join<lmatrix, lvector>(mult, 1, num_reps);

  std::cout << std::endl << "matrix * outer(vector * vector)" << std::endl;
  time_matvec_outer_join<pmatrix, pvector>(mult, num_reps);
  time_matvec_outer_join<lmatrix, lvector>(mult, num_reps);

  std::cout << std::endl << "matrix.marginal(0)" << std::endl;
  time_aggregate<pmatrix, pvector>(member_marginal(), 0, num_reps);
  time_aggregate<lmatrix, lvector>(member_marginal(), 0, num_reps / 10);

  std::cout << std::endl << "matrix.marginal(1)" << std::endl;
  time_aggregate<pmatrix, pvector>(member_marginal(), 1, num_reps);
  time_aggregate<lmatrix, lvector>(member_marginal(), 1, num_reps / 10);

  std::cout << std::endl << "matrix.maximum(row, col)" << std::endl;
  time_find<pmatrix, pvector>(member_max(), num_reps);
  time_find<lmatrix, lvector>(member_max(), num_reps);

  std::cout << std::endl << "(matrix * vector).marginal(0)" << std::endl;
  time_join_aggregate<pmatrix, pvector>(mult, member_marginal(), 0, num_reps);
  time_join_aggregate<lmatrix, lvector>(mult, member_marginal(), 0, num_reps / 10);

  std::cout << std::endl << "(matrix * vector).marginal(1)" << std::endl;
  time_join_aggregate<pmatrix, pvector>(mult, member_marginal(), 1, num_reps);
  time_join_aggregate<lmatrix, lvector>(mult, member_marginal(), 1, num_reps / 10);

  std::cout << std::endl << "(matrix * vector).maximum(0)" << std::endl;
  time_join_aggregate<pmatrix, pvector>(mult, member_maximum(), 0, num_reps);
  time_join_aggregate<lmatrix, lvector>(mult, member_maximum(), 0, num_reps);

  std::cout << std::endl << "(matrix * vector).maximum(1)" << std::endl;
  time_join_aggregate<pmatrix, pvector>(mult, member_maximum(), 1, num_reps);
  time_join_aggregate<lmatrix, lvector>(mult, member_maximum(), 1, num_reps);

  std::cout << std::endl << "matrix.restrict(0, val)" << std::endl;
  time_restrict<pmatrix, pvector>(0, num_reps);
  time_restrict<lmatrix, lvector>(0, num_reps);

  std::cout << std::endl << "matrix.restrict(1, val)" << std::endl;
  time_restrict<pmatrix, pvector>(1, num_reps);
  time_restrict<lmatrix, lvector>(1, num_reps);

  std::cout << std::endl << "vector *= matrix.restrict(0, val)" << std::endl;
  time_restrict_update<pmatrix, pvector>(0, multiplies_assign<>(), num_reps);
  time_restrict_update<lmatrix, lvector>(0, multiplies_assign<>(), num_reps);

  std::cout << std::endl << "vector *= matrix.restrict(1, val)" << std::endl;
  time_restrict_update<pmatrix, pvector>(1, multiplies_assign<>(), num_reps);
  time_restrict_update<lmatrix, lvector>(1, multiplies_assign<>(), num_reps);

  return 0;
}
