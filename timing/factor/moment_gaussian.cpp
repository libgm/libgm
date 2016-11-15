#include <libgm/argument/var.hpp>
#include <libgm/factor/moment_gaussian.hpp>

#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

using mgaussian = moment_gaussian<var>;

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

void time_multiply_constant() {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    mgaussian f(make_domain(u, n));
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
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> head1 = make_domain(u, n);
    domain<var> head2 = make_domain(u, n);
    domain<var> tail2 = contiguous ? head1.prefix(2) : endpoints(head1);
    mgaussian f(head1);
    mgaussian g(head2, tail2);
    mgaussian h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = f * g;
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
    domain<var> retain = contiguous ? args.prefix(2) : endpoints(args);
    mgaussian f(args);
    mgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f.marginal(retain);
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
    domain<var> retain = contiguous ? args.prefix(2) : endpoints(args);
    mgaussian f(args);
    mgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f.marginal(args - retain);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_restrict_head(bool contiguous) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> head = make_domain(u, n);
    real_assignment<var> a;
    a[head[0]] = 2.0;
    a[contiguous ? head[1] : head.back()] = 3.0;
    mgaussian f(head, dense_vector<>::Zero(n), dense_matrix<>::Identity(n, n));
    mgaussian g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f.restrict(a);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

void time_restrict_tail(bool contiguous) {
  universe u;
  boost::timer t;
  for (std::size_t n : num_dims) {
    domain<var> head = make_domain(u, n);
    domain<var> tail = make_domain(u, n);
    real_assignment<var> a;
    a[tail[0]] = 2.0;
    a[contiguous ? tail[1] : tail.back()] = 3.0;
    mgaussian f(head, tail,
                dense_vector<>::Zero(n), dense_matrix<>::Identity(n, n),
                dense_matrix<>::Identity(n, n));
    mgaussian g;
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
