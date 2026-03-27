#include <libgm/factor/logarithmic_vector.hpp>
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
std::size_t num_reps = 0;

const char* version(double) {
  return "probability";
}

const char* version(Exp<double>) {
  return "logarithmic";
}

template <typename Factor, std::size_t N, typename Op>
void time_transform(Op op) {
  Timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (size_t len : num_values) {
    Factor f(len);
    Factor g;
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

template <typename Factor, typename Op>
void time_accumulate(Op op) {
  Timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (size_t len : num_values) {
    Factor f(len);
    typename Factor::result_type value{};
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      value = op(f);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
    (void)value;
  }
  std::cout << std::endl;
}

template <typename Factor, typename Op>
void time_find(Op op) {
  Timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (size_t len : num_values) {
    Factor f(len);
    typename Factor::result_type value{};
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

int main(int argc, char** argv) {
  size_t min_values;
  size_t step_size;
  size_t max_values = 0;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("min-values", po::value<size_t>(&min_values)->default_value(1),
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

  std::cout << "Measuring vector factors with "
            << " min_values=" << min_values
            << " max_values=" << max_values
            << " step_size=" << step_size
            << " num_reps=" << num_reps
            << std::endl;

  for (size_t i = min_values; i <= max_values; i += step_size) {
    num_values.push_back(i);
  }

  using PVector = ProbabilityVector<double>;
  using LVector = LogarithmicVector<double>;

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "vector * constant" << std::endl;
  time_transform<PVector, 1>([](const PVector& x) { return x * 2.0; });
  time_transform<LVector, 1>([](const LVector& x) { return x * Exp<double>(2.0); });

  std::cout << std::endl << "vector.marginal()" << std::endl;
  time_accumulate<PVector>([](const PVector& x) { return x.marginal(); });

  std::cout << std::endl << "vector.maximum()" << std::endl;
  time_accumulate<PVector>([](const PVector& x) { return x.maximum(); });
  time_accumulate<LVector>([](const LVector& x) { return x.maximum(); });

  std::cout << std::endl << "vector.minimum()" << std::endl;
  time_accumulate<PVector>([](const PVector& x) { return x.minimum(); });
  time_accumulate<LVector>([](const LVector& x) { return x.minimum(); });

  std::cout << std::endl << "vector.max(pos)" << std::endl;
  time_find<PVector>([](const PVector& x, std::vector<size_t>* pos) { return x.maximum(pos); });
  time_find<LVector>([](const LVector& x, std::vector<size_t>* pos) { return x.maximum(pos); });

  std::cout << std::endl << "vector.min(pos)" << std::endl;
  time_find<PVector>([](const PVector& x, std::vector<size_t>* pos) { return x.minimum(pos); });
  time_find<LVector>([](const LVector& x, std::vector<size_t>* pos) { return x.minimum(pos); });

  return 0;
}
