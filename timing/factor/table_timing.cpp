#include <libgm/argument/dims.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/math/exp.hpp>
#include <timing/timer.hpp>

#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

std::size_t num_values = 0;
std::size_t max_arity = 0;
std::size_t num_reps = 0;

const char* version(double) {
  return "probability";
}

const char* version(Exp<double>) {
  return "logarithmic";
}

Dims make_dims(std::initializer_list<size_t> idx) {
  Dims d;
  for (size_t i : idx) {
    d.set(i);
  }
  return d;
}

Dims make_dims_range(size_t begin, size_t end) {
  Dims d;
  for (size_t i = begin; i < end; ++i) {
    d.set(i);
  }
  return d;
}

template <typename Factor, std::size_t N, typename Op>
void time_transform(Op op) {
  Timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (size_t arity = 0; arity <= max_arity; ++arity) {
    Factor f(Shape(arity, num_values));
    Factor g;
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      if constexpr (N == 1) {
        g = op(f);
      } else if constexpr (N == 2) {
        g = op(f, f);
      }
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename JoinOp>
void time_join(JoinOp join_op) {
  Timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(Shape(arity - 1, num_values));
    Factor g(Shape(arity - 1, num_values));
    Factor h;
    Dims x = make_dims_range(1, arity);
    Dims y = make_dims_range(0, arity - 1);
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g, x, y);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename AggOp>
void time_aggregate(AggOp agg_op) {
  Timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(Shape(arity, num_values));
    Factor g;
    Dims retain = make_dims_range(0, arity - 1);
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = agg_op(f, retain);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename RestrictOp>
void time_restrict_dims(RestrictOp restrict_op) {
  Timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(Shape(arity, num_values));
    Factor g;
    std::vector<size_t> vals = {1};
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      g = restrict_op(f, vals);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename UpdateOp>
void time_update(UpdateOp update_op) {
  Timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(Shape(arity, num_values));
    Factor g(Shape(arity - 1, num_values));
    Dims x = make_dims_range(1, arity);
    t.restart();
    for (size_t i = 0; i < num_reps; ++i) {
      update_op(f, g, x);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("num-values", po::value<size_t>(&num_values),
     "the number of values in the variables")
    ("max-arity", po::value<size_t>(&max_arity),
     "the maximum arity of the result")
    ("num-reps", po::value<size_t>(&num_reps),
     "the number of repetitions");

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  notify(vm);

  if (vm.count("help") || !num_values || !max_arity || !num_reps) {
    std::cout << desc;
    return 0;
  }

  std::cout << "Measuring table factors with"
            << " num_values=" << num_values
            << " max_arity=" << max_arity
            << " num_reps=" << num_reps
            << std::endl;

  using PTable = ProbabilityTable<double>;
  using LTable = LogarithmicTable<double>;

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "table * constant" << std::endl;
  time_transform<PTable, 1>([](const PTable& x) { return x * 2.0; });
  time_transform<LTable, 1>([](const LTable& x) { return x * Exp<double>(2.0); });

  std::cout << std::endl << "table * table -- direct" << std::endl;
  time_transform<PTable, 2>(std::multiplies<>());
  time_transform<LTable, 2>(std::multiplies<>());

  std::cout << std::endl << "table * table -- dims" << std::endl;
  time_join<PTable>([](const PTable& a, const PTable& b, const Dims& i, const Dims& j) {
    return multiply(a, b, i, j);
  });
  time_join<LTable>([](const LTable& a, const LTable& b, const Dims& i, const Dims& j) {
    return multiply(a, b, i, j);
  });

  std::cout << std::endl << "table / table -- dims" << std::endl;
  time_join<PTable>([](const PTable& a, const PTable& b, const Dims& i, const Dims& j) {
    return divide(a, b, i, j);
  });
  time_join<LTable>([](const LTable& a, const LTable& b, const Dims& i, const Dims& j) {
    return divide(a, b, i, j);
  });

  std::cout << std::endl << "table.marginal(dom)" << std::endl;
  time_aggregate<PTable>([](const PTable& x, const Dims& retain) { return x.marginal_dims(retain); });

  std::cout << std::endl << "table.maximum(dom)" << std::endl;
  time_aggregate<PTable>([](const PTable& x, const Dims& retain) { return x.maximum_dims(retain); });
  time_aggregate<LTable>([](const LTable& x, const Dims& retain) { return x.maximum_dims(retain); });

  std::cout << std::endl << "table.restrict(dims, vals)" << std::endl;
  time_restrict_dims<PTable>([](const PTable& x, const std::vector<size_t>& vals) {
    return x.restrict_dims(make_dims({0}), vals);
  });
  time_restrict_dims<LTable>([](const LTable& x, const std::vector<size_t>& vals) {
    return x.restrict_dims(make_dims({0}), vals);
  });

  std::cout << std::endl << "table.restrict_front(vals)" << std::endl;
  time_restrict_dims<PTable>([](const PTable& x, const std::vector<size_t>& vals) { return x.restrict_front(vals); });
  time_restrict_dims<LTable>([](const LTable& x, const std::vector<size_t>& vals) { return x.restrict_front(vals); });

  std::cout << std::endl << "table.restrict_back(vals)" << std::endl;
  time_restrict_dims<PTable>([](const PTable& x, const std::vector<size_t>& vals) { return x.restrict_back(vals); });
  time_restrict_dims<LTable>([](const LTable& x, const std::vector<size_t>& vals) { return x.restrict_back(vals); });

  std::cout << std::endl << "table *= table" << std::endl;
  time_update<PTable>([](PTable& x, const PTable& y, const Dims& dims) { x.multiply_in(y, dims); });
  time_update<LTable>([](LTable& x, const LTable& y, const Dims& dims) { x.multiply_in(y, dims); });

  std::cout << std::endl << "table /= table" << std::endl;
  time_update<PTable>([](PTable& x, const PTable& y, const Dims& dims) { x.divide_in(y, dims); });
  time_update<LTable>([](LTable& x, const LTable& y, const Dims& dims) { x.divide_in(y, dims); });

  return 0;
}
