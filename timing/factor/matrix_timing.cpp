#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/math/exp.hpp>
#include <timing/timer.hpp>

#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

std::vector<size_t> num_values;

const char* version(double) {
  return "probability";
}

const char* version(Exp<double>) {
  return "logarithmic";
}

template <typename Matrix, std::size_t N, typename Op>
void time_transform(Op op, size_t num_reps) {
  Timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (size_t n : num_values) {
    Matrix f(n, n);
    Matrix g;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      if constexpr (N == 1) {
        g = op(f);
      } else {
        g = op(f, f);
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename JoinOp>
void time_matmat_transpose(JoinOp join_op, size_t num_reps) {
  Timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (size_t n : num_values) {
    Matrix f(n, n);
    Matrix g(n, n);
    Matrix h;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g.transpose());
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename JoinOp>
void time_matvec_join(JoinOp join_op, size_t num_reps) {
  Timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (size_t n : num_values) {
    Matrix f(n, n);
    Vector g(n);
    Matrix h;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename JoinOp>
void time_vecmat_join(JoinOp join_op, size_t num_reps) {
  Timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (size_t n : num_values) {
    Vector f(n);
    Matrix g(n, n);
    Matrix h;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector, typename AggOp>
void time_aggregate(AggOp agg_op, size_t num_reps) {
  Timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (size_t n : num_values) {
    Matrix f(n, n);
    Vector g;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = agg_op(f);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Op>
void time_find(Op op, size_t num_reps) {
  Timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (size_t n : num_values) {
    Matrix f(n, n);
    typename Matrix::result_type value{};
    std::vector<size_t> pos;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      value = op(f, &pos);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
    (void)value;
  }
  std::cout << std::endl;
}

template <typename Matrix, typename Vector>
void time_restrict(size_t d, size_t num_reps) {
  Timer t;
  std::cout << version(typename Matrix::result_type()) << std::flush;
  for (size_t n : num_values) {
    Matrix f(n, n);
    Vector g;
    std::vector<size_t> vals = {n - 1};
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = d == 0 ? f.restrict_front(vals) : f.restrict_back(vals);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  size_t min_values;
  size_t step_size;
  size_t max_values = 0;
  size_t num_reps = 0;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("min-values", po::value<size_t>(&min_values)->default_value(2),
     "the initial number of values")
    ("max-values", po::value<size_t>(&max_values),
     "the maximum number of values")
    ("step-size", po::value<size_t>(&step_size)->default_value(1),
     "the increment in the number of values")
    ("num-reps", po::value<size_t>(&num_reps),
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

  for (size_t i = min_values; i <= max_values; i += step_size) {
    num_values.push_back(i);
  }

  using PVector = ProbabilityVector<double>;
  using PMatrix = ProbabilityMatrix<double>;
  using LVector = LogarithmicVector<double>;
  using LMatrix = LogarithmicMatrix<double>;

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "matrix * constant" << std::endl;
  time_transform<PMatrix, 1>([](const PMatrix& x) { return x * 2.0; }, num_reps);
  time_transform<LMatrix, 1>([](const LMatrix& x) { return x * Exp<double>(2.0); }, num_reps);

  std::cout << std::endl << "matrix * matrix -- direct" << std::endl;
  time_transform<PMatrix, 2>(std::multiplies<>(), num_reps);
  time_transform<LMatrix, 2>(std::multiplies<>(), num_reps);

  std::cout << std::endl << "matrix * matrix -- transposed" << std::endl;
  time_matmat_transpose<PMatrix>(std::multiplies<>(), num_reps);
  time_matmat_transpose<LMatrix>(std::multiplies<>(), num_reps);

  std::cout << std::endl << "matrix * vector -- direct" << std::endl;
  time_matvec_join<PMatrix, PVector>([](const PMatrix& x, const PVector& y) {
    return x.multiply_back(y);
  }, num_reps);
  time_matvec_join<LMatrix, LVector>([](const LMatrix& x, const LVector& y) {
    return x.multiply_back(y);
  }, num_reps);

  std::cout << std::endl << "matrix * vector -- transposed" << std::endl;
  time_matvec_join<PMatrix, PVector>([](const PMatrix& x, const PVector& y) {
    return x.multiply_back(y);
  }, num_reps);
  time_matvec_join<LMatrix, LVector>([](const LMatrix& x, const LVector& y) {
    return x.multiply_back(y);
  }, num_reps);

  std::cout << std::endl << "vector * matrix -- direct" << std::endl;
  time_vecmat_join<PMatrix, PVector>([](const PVector& x, const PMatrix& y) {
    return y.multiply_front(x);
  }, num_reps);
  time_vecmat_join<LMatrix, LVector>([](const LVector& x, const LMatrix& y) {
    return y.multiply_front(x);
  }, num_reps);

  std::cout << std::endl << "vector * matrix -- transposed" << std::endl;
  time_vecmat_join<PMatrix, PVector>([](const PVector& x, const PMatrix& y) {
    return y.multiply_front(x);
  }, num_reps);
  time_vecmat_join<LMatrix, LVector>([](const LVector& x, const LMatrix& y) {
    return y.multiply_front(x);
  }, num_reps);

  std::cout << std::endl << "matrix.marginal(0)" << std::endl;
  time_aggregate<PMatrix, PVector>([](const PMatrix& x) { return x.marginal_front(); }, num_reps);

  std::cout << std::endl << "matrix.marginal(1)" << std::endl;
  time_aggregate<PMatrix, PVector>([](const PMatrix& x) { return x.marginal_back(); }, num_reps);

  std::cout << std::endl << "matrix.maximum(row, col)" << std::endl;
  time_find<PMatrix>([](const PMatrix& x, std::vector<size_t>* pos) { return x.maximum(pos); }, num_reps);
  time_find<LMatrix>([](const LMatrix& x, std::vector<size_t>* pos) { return x.maximum(pos); }, num_reps);

  std::cout << std::endl << "matrix.restrict(0, val)" << std::endl;
  time_restrict<PMatrix, PVector>(0, num_reps);
  time_restrict<LMatrix, LVector>(0, num_reps);

  std::cout << std::endl << "matrix.restrict(1, val)" << std::endl;
  time_restrict<PMatrix, PVector>(1, num_reps);
  time_restrict<LMatrix, LVector>(1, num_reps);

  return 0;
}
