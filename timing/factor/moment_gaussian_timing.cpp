#include <libgm/argument/dims.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
#include <timing/timer.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

using MGaussian = MomentGaussian<double>;
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

void time_multiply_constant() {
  Timer t;
  for (size_t n : num_dims) {
    MGaussian f(unit_shape(n));
    MGaussian g;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = f * Exp<double>(2.0);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply_back() {
  Timer t;
  for (size_t n : num_dims) {
    size_t m = n - 1;
    Shape shape{m, 1};
    MGaussian f(shape, Vec::Zero(m), Mat::Identity(m, m), Mat::Zero(m, 1));
    MGaussian g(Shape{1});
    MGaussian h;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = f.multiply_back(g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply_dims() {
  Timer t;
  for (size_t n : num_dims) {
    MGaussian f(Shape{n});
    MGaussian g(Shape{n});
    MGaussian h;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = multiply(f, g, make_dims({0}), make_dims({1}));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_marginal_front() {
  Timer t;
  for (size_t n : num_dims) {
    MGaussian f(unit_shape(n));
    MGaussian g;
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
    MGaussian f(unit_shape(n));
    MGaussian g;
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
    MGaussian f(unit_shape(n));
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
    MGaussian f(unit_shape(n));
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
    MGaussian f(unit_shape(n));
    MGaussian g;
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
    MGaussian f(unit_shape(n));
    MGaussian g;
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

  std::cout << "Measuring moment Gaussian factors with "
            << " min_dims=" << min_dims
            << " max_dims=" << max_dims
            << " step_size=" << step_size
            << " num_reps=" << num_reps
            << std::endl;

  for (size_t i = min_dims; i <= max_dims; i += step_size) {
    num_dims.push_back(i);
  }

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "mg * constant" << std::endl;
  time_multiply_constant();

  std::cout << std::endl << "mg.multiply_back() -- contiguous" << std::endl;
  time_multiply_back();

  std::cout << std::endl << "mg.multiply(dims) -- indexed" << std::endl;
  time_multiply_dims();

  std::cout << std::endl << "mg.marginal_front() -- contiguous" << std::endl;
  time_marginal_front();

  std::cout << std::endl << "mg.marginal_dims() -- indexed" << std::endl;
  time_marginal_dims();

  std::cout << std::endl << "mg.marginal_front().marginal()" << std::endl;
  time_sum_front();

  std::cout << std::endl << "mg.marginal_dims().marginal()" << std::endl;
  time_sum_dims();

  std::cout << std::endl << "mg.restrict_front()" << std::endl;
  time_restrict_front();

  std::cout << std::endl << "mg.restrict_dims()" << std::endl;
  time_restrict_dims();

  return 0;
}
