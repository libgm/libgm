#include <libgm/argument/dims.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
#include <timing/timer.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

using CGaussian = CanonicalGaussian<double>;
using Vec = Vector<double>;
using Mat = Matrix<double>;

std::vector<size_t> num_dims;
std::size_t num_reps = 0;

Shape unit_shape(size_t n) {
  return Shape(n, 1);
}

Dims first_last_dims(size_t n) {
  return make_dims({0, n - 1});
}

void time_unary_transform() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n));
    CGaussian g;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = f * Exp<double>(3.0);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_binary_transform() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n));
    CGaussian g(unit_shape(n));
    CGaussian h;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = f * g;
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply_front() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(Shape{1, n - 1});
    CGaussian g(Shape{1});
    CGaussian h;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = f.multiply_front(g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply_dims() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(Shape{n});
    CGaussian g(Shape{n});
    CGaussian h;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = multiply(f, g, make_dims({0}), make_dims({1}));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply_in_front() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n));
    CGaussian g(unit_shape(n - 1));
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      f.multiply_in_front(g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply_in_dims() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n));
    CGaussian g(unit_shape(n - 1));
    Dims dims;
    for (size_t i = 1; i < n; ++i) {
      dims.set(i);
    }
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      f.multiply_in(g, dims);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_marginal_front() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n), Vec::Zero(n), Mat::Identity(n, n));
    CGaussian g;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = f.marginal_front(2);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_marginal_dims() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n), Vec::Zero(n), Mat::Identity(n, n));
    CGaussian g;
    Dims dims = first_last_dims(n);
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = f.marginal_dims(dims);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_sum_front() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n), Vec::Zero(n), Mat::Identity(n, n));
    Exp<double> z;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      z = f.marginal_front(2).marginal();
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
    (void)z;
  }
  std::cout << std::endl;
}

void time_sum_dims() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n), Vec::Zero(n), Mat::Identity(n, n));
    Exp<double> z;
    Dims dims = first_last_dims(n);
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      z = f.marginal_dims(dims).marginal();
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
    (void)z;
  }
  std::cout << std::endl;
}

void time_restrict_front() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n));
    CGaussian g;
    Vector<double> vals = Vector<double>::Ones(2);
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = f.restrict_front(vals);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_restrict_dims() {
  Timer t;
  for (size_t n : num_dims) {
    CGaussian f(unit_shape(n));
    CGaussian g;
    Vector<double> vals = Vector<double>::Ones(2);
    Dims dims = first_last_dims(n);
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = f.restrict_dims(dims, vals);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  size_t min_dims, max_dims;
  size_t step_size;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("min-dims", po::value<size_t>(&min_dims)->default_value(2),
     "the initial number of dimensions")
    ("max-dims", po::value<size_t>(&max_dims),
     "the maximum number of dimensions")
    ("step-size", po::value<size_t>(&step_size)->default_value(1),
     "the increment in the number of dimensions")
    ("num-reps", po::value<size_t>(&num_reps),
     "the number of repetitions");

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  notify(vm);

  if (vm.count("help") || !max_dims || !num_reps) {
    std::cout << desc;
    return 0;
  }

  std::cout << "Measuring canonical Gaussian factors with "
            << " min_dims=" << min_dims
            << " max_dims=" << max_dims
            << " step_size=" << step_size
            << " num_reps=" << num_reps
            << std::endl;

  for (size_t i = min_dims; i <= max_dims; i += step_size) {
    num_dims.push_back(i);
  }

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "cg * constant" << std::endl;
  time_unary_transform();

  std::cout << std::endl << "cg * cg -- direct" << std::endl;
  time_binary_transform();

  std::cout << std::endl << "cg.multiply_front() -- contiguous" << std::endl;
  time_multiply_front();

  std::cout << std::endl << "cg.multiply(dims) -- indexed" << std::endl;
  time_multiply_dims();

  std::cout << std::endl << "cg.multiply_in_front() -- contiguous" << std::endl;
  time_multiply_in_front();

  std::cout << std::endl << "cg.multiply_in(dims) -- indexed" << std::endl;
  time_multiply_in_dims();

  std::cout << std::endl << "cg.marginal_front() -- contiguous" << std::endl;
  time_marginal_front();

  std::cout << std::endl << "cg.marginal_dims() -- indexed" << std::endl;
  time_marginal_dims();

  std::cout << std::endl << "cg.marginal_front().marginal()" << std::endl;
  time_sum_front();

  std::cout << std::endl << "cg.marginal_dims().marginal()" << std::endl;
  time_sum_dims();

  std::cout << std::endl << "cg.restrict_front()" << std::endl;
  time_restrict_front();

  std::cout << std::endl << "cg.restrict_dims()" << std::endl;
  time_restrict_dims();

  return 0;
}
