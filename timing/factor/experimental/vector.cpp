#include <libgm/factor/probability_vector.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/traits.hpp>
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
std::size_t num_reps = 0;

const char* version(double) {
  return "probability";
}

const char* version(logd) {
  return "logarithmic";
}

template <typename Factor, std::size_t N, typename Op>
void time_transform(Op op) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t len : num_values) {
    auto f = tuple_rep<N>(Factor(len));
    Factor g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = tuple_apply(op, f);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename Op>
auto time_accumulate(Op op) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  typename Factor::result_type value;
  for (std::size_t len : num_values) {
    Factor f(len);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      value *= op(f);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
  return value;
}

template <typename Factor, typename Op>
auto time_find(Op op) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  typename Factor::result_type value;
  for (std::size_t len : num_values) {
    Factor f(len);
    std::size_t pos;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      value *= op(f, pos);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
  return value;
}

struct ternary_multiplies {
  template <typename X, typename Y, typename Z>
  auto operator()(X&& x, Y&& y, Z&& z) {
    return std::forward<X>(x) * std::forward<Y>(y) * std::forward<Z>(z);
  }
};

int main(int argc, char** argv) {
  std::size_t min_values;
  std::size_t step_size;
  std::size_t max_values = 0;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("min-values", po::value<std::size_t>(&min_values)->default_value(1),
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

  std::cout << "Measuring vector factors with "
            << " min_values=" << min_values
            << " max_values=" << max_values
            << " step_size=" << step_size
            << " num_reps=" << num_reps
            << std::endl;

  for (std::size_t i = min_values; i <= max_values; i += step_size) {
    num_values.push_back(i);
  }

  typedef experimental::probability_vector<> pvector;
  typedef experimental::logarithmic_vector<> lvector;

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "vector * constant" << std::endl;
  time_transform<pvector, 1>(multiplied_by<double>(2));
  time_transform<lvector, 1>(multiplied_by<logd>(logd(2)));

  std::cout << std::endl << "vector + vector" << std::endl;
  time_transform<pvector, 2>(std::plus<>());
  time_transform<lvector, 2>(std::plus<>());

  std::cout << std::endl << "vector * vector * vector" << std::endl;
  time_transform<pvector, 3>(ternary_multiplies());
  time_transform<lvector, 3>(ternary_multiplies());

  std::cout << std::endl << "vector.marginal()" << std::endl;
  time_accumulate<pvector>(member_sum());
  time_accumulate<lvector>(member_sum());

  std::cout << std::endl << "vector.maximum()" << std::endl;
  time_accumulate<pvector>(member_max());
  time_accumulate<lvector>(member_max());

  std::cout << std::endl << "vector.minimum()" << std::endl;
  time_accumulate<pvector>(member_min());
  time_accumulate<lvector>(member_min());

  std::cout << std::endl << "vector.max(pos)" << std::endl;
  time_find<pvector>(member_max());
  time_find<lvector>(member_max());

  std::cout << std::endl << "vector.min(pos)" << std::endl;
  time_find<pvector>(member_min());
  time_find<lvector>(member_min());

  return 0;
}
