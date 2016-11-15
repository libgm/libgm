#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/functional/arithmetic.hpp>

#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

using cgaussian = canonical_gaussian<var>;

// global options
uint_vector num_dims;
std::size_t num_reps;

domain<var> make_domain(universe& u, std::size_t n) {
  domain<var> result;
  for (std::size_t i = 0; i < n; ++i) {
    result.push_back(var::continuous(u, "x"));
  }
  return result;
}

domain<var> endpoints(const domain<var>& x) {
  return { x.front(), x.back() };
}

template <typename Op>
void time_unary_transform(Op op) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    cgaussian f(make_domain(u, n));
    cgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = op(f);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Op>
void time_binary_transform(Op op) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> args = make_domain(u, n);
    cgaussian f(args);
    cgaussian g(args);
    cgaussian h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = op(f, g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply(bool contiguous) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> x = make_domain(u, n);
    domain<var> y = make_domain(u, n);
    if (contiguous) {
      y[0] = x[0];
      y[1] = x[1];
    } else {
      y[0] = x[0];
      y[n-1] = x[n-1];
    }
    cgaussian f(x);
    cgaussian g(y);
    cgaussian h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = f * g;
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_multiply_in(bool contiguous) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> x = make_domain(u, n);
    domain<var> y;
    if (contiguous) {
      y = x.prefix(n-1);
    } else {
      y = x.suffix(n-1);
      y[0] = x[0];
    }
    cgaussian f(x);
    cgaussian g(y);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      f *= g;
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_marginal(bool contiguous) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> args = make_domain(u, n);
    cgaussian f(args, dense_vector<>::Ones(n), dense_matrix<>::Identity(n, n));
    cgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      if (contiguous) {
        g = f.marginal(args.prefix(2));
      } else {
        g = f.marginal(endpoints(args));
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_sum(bool contiguous) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> args = make_domain(u, n);
    cgaussian f(args, dense_vector<>::Ones(n), dense_matrix<>::Identity(n, n));
    cgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      if (contiguous) {
        g = f.marginal(args.suffix(n-2));
      } else {
        g = f.marginal(args - endpoints(args));
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_restrict(bool contiguous) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> args = make_domain(u, n);
    cgaussian f(args);
    cgaussian g;
    real_assignment<var> a;
    a[args[0]] = 2.0;
    a[contiguous ? args[1] : args.back()] = 3.0;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f.restrict(a);
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

  std::cout << std::endl << "cg * constant" << std::endl;
  time_unary_transform(multiplied_by<logd>(logd(3.0, log_tag())));

  std::cout << std::endl << "cg * cg -- direct" << std::endl;
  time_binary_transform(std::multiplies<>());

  std::cout << std::endl << "cg * cg -- span" << std::endl;
  time_multiply(true);

  std::cout << std::endl << "cg * cg -- iref" << std::endl;
  time_multiply(false);

  std::cout << std::endl << "cg *= cg -- span" << std::endl;
  time_multiply_in(true);

  std::cout << std::endl << "cg *= cg -- iref" << std::endl;
  time_multiply_in(false);

  std::cout << std::endl << "cg.marginal(dom) -- span" << std::endl;
  time_marginal(true);

  std::cout << std::endl << "cg.marginal(dom) -- iref" << std::endl;
  time_marginal(false);

  std::cout << std::endl << "cg.head().sum() -- span" << std::endl;
  time_sum(true);

  std::cout << std::endl << "cg.dims().sum() -- iref" << std::endl;
  time_sum(false);

  std::cout << std::endl << "cg.restrict() -- span" << std::endl;
  time_restrict(true);

  std::cout << std::endl << "cg.restrict() -- iref" << std::endl;
  time_restrict(false);

  return 0;
}
