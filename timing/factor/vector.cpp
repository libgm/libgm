#include <libgm/argument/var.hpp>
#include <libgm/factor/probability_array.hpp>
#include <libgm/factor/canonical_array.hpp>
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

const char* version(double) {
  return "probability";
}

const char* version(logd) {
  return "canonical  ";
}

template <typename Factor, std::size_t N, typename Op>
void time_transform(Op op,
                    const uint_vector& num_values,
                    std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t n : num_values) {
    auto f = tuple_rep<N>(Factor({var::discrete(u, "x", n)}));
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
void time_assignment(Op op,
                     const uint_vector& num_values,
                     std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t n : num_values) {
    Factor f({var::discrete(u, "x", n)});
    uint_assignment<var> a;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      result_t<Factor> value = op(f, a);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
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
  std::size_t num_reps = 0;

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

  typedef probability_array<var, 1> pvector;
  typedef canonical_array<var, 1> lvector;
  uint_vector num_values;
  for (std::size_t i = min_values; i <= max_values; i += step_size) {
    num_values.push_back(i);
  }

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "vector * constant" << std::endl;
  time_transform<pvector, 1>(multiplied_by<double>(2), num_values, num_reps);
  time_transform<lvector, 1>(multiplied_by<logd>(logd(2)), num_values, num_reps);

  std::cout << std::endl << "vector + vector" << std::endl;
  time_transform<pvector, 2>(std::plus<>(), num_values, num_reps);
  time_transform<lvector, 2>(std::plus<>(), num_values, num_reps);

  std::cout << std::endl << "vector * vector * vector" << std::endl;
  time_transform<pvector, 3>(ternary_multiplies(), num_values, num_reps);
  time_transform<lvector, 3>(ternary_multiplies(), num_values, num_reps);

  std::cout << std::endl << "vector.maximum(a)" << std::endl;
  time_assignment<pvector>(member_maximum(), num_values, num_reps);

  return 0;
}
