#include <libgm/factor/experimental/moment_gaussian.hpp>

#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

using mgaussian = experimental::moment_gaussian<double>;

// global options
uint_vector num_dims;
std::size_t num_reps;

void time_multiply_constant() {
  boost::timer t;
  for (std::size_t n : num_dims) {
    mgaussian f(n);
    mgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f * logd(2);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply_head_tail(bool contiguous) {
  boost::timer t;
  for (std::size_t n : num_dims) {
    mgaussian f(n);
    mgaussian g(n, 2);
    mgaussian h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      if (contiguous) {
        h = f.head(0, 2) * g.tail();
      } else {
        h = f.head(uint_vector{0, n-1}) * g.tail();
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_marginal(bool contiguous) {
  boost::timer t;
  for (std::size_t n : num_dims) {
    mgaussian f(n);
    mgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      if (contiguous) {
        g = f.marginal(0, 2);
      } else {
        g = f.marginal(uint_vector{0, n-1});
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_sum(bool contiguous) {
  boost::timer t;
  for (std::size_t n : num_dims) {
    mgaussian f(n);
    mgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      if (contiguous) {
        g = f.head(0, 2).sum();
      } else {
        g = f.head(uint_vector{0, n-1}).sum();
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_restrict_head(bool contiguous) {
  boost::timer t;
  for (std::size_t n : num_dims) {
    mgaussian f(real_vector<>::Zero(n), real_matrix<>::Identity(n, n));
    mgaussian g;
    uint_vector dims = {0, n-1};
    real_vector<> vals = real_vector<>::Ones(2);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      if (contiguous) {
        g = f.restrict_head(0, 2, vals);
      } else {
        g = f.restrict_head(dims, vals);
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_restrict_tail(bool contiguous) {
  boost::timer t;
  for (std::size_t n : num_dims) {
    mgaussian f(real_vector<>::Zero(n), real_matrix<>::Identity(n, n),
                real_matrix<>::Identity(n, n));
    mgaussian g;
    uint_vector dims = {0, n-1};
    real_vector<> vals = real_vector<>::Ones(2);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      if (contiguous) {
        g = f.restrict_tail(0, 2, vals);
      } else {
        g = f.restrict_tail(dims, vals);
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  std::size_t min_dims, max_dims;
  std::size_t step_size;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("min-dims", po::value<std::size_t>(&min_dims)->default_value(2),
     "the initial number of dimensions")
    ("max-dims", po::value<std::size_t>(&max_dims)->default_value(0),
     "the maximum number of dimensions")
    ("step-size", po::value<std::size_t>(&step_size)->default_value(1),
     "the increment in the number of dimensions")
    ("num-reps", po::value<std::size_t>(&num_reps)->default_value(0),
     "the number of repetitions");

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  notify(vm);

  if (vm.count("help") || !max_dims || !num_reps) {
    std::cout << desc;
    return 0;
  }

  std::cout << "Measuring matrix factors with "
            << " min_dims=" << min_dims
            << " max_dims=" << max_dims
            << " step_size=" << step_size
            << " num_reps=" << num_reps
            << std::endl;

  for (std::size_t i = min_dims; i <= max_dims; i += step_size) {
    num_dims.push_back(i);
  }

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "mg * constant" << std::endl;
  time_multiply_constant();

  std::cout << std::endl << "mg.head() * mg.tail() -- span" << std::endl;
  time_multiply_head_tail(true);

  std::cout << std::endl << "mg.head() * mg.tail() -- iref" << std::endl;
  time_multiply_head_tail(false);

  std::cout << std::endl << "mg.marginal(dom) -- span" << std::endl;
  time_marginal(true);

  std::cout << std::endl << "mg.marginal(dom) -- iref" << std::endl;
  time_marginal(false);

  std::cout << std::endl << "mg.head().sum(dom) -- span" << std::endl;
  time_sum(true);

  std::cout << std::endl << "mg.head().sum(dom) -- iref" << std::endl;
  time_sum(false);

  std::cout << std::endl << "mg.restrict_head() -- span" << std::endl;
  time_restrict_head(true);

  std::cout << std::endl << "mg.restrict_head() -- iref" << std::endl;
  time_restrict_head(false);

  std::cout << std::endl << "mg.restrict_tail() -- span" << std::endl;
  time_restrict_tail(true);

  std::cout << std::endl << "mg.restrict_tail() -- iref" << std::endl;
  time_restrict_tail(false);

  return 0;
}
