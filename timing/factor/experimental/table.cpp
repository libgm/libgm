#include <libgm/factor/experimental/probability_table.hpp>
#include <libgm/factor/experimental/logarithmic_table.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>

#include <functional>
#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

// global options
std::size_t num_values = 0;
std::size_t max_arity = 0;
std::size_t num_reps = 0;

using namespace libgm;

const char* version(double) {
  return "probability";
}

const char* version(logd) {
  return "logarithmic";
}

template <typename Factor, std::size_t N, typename Op>
void time_transform(Op op) {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 0; arity <= max_arity; ++arity) {
    auto f = tuple_rep<N>(Factor(uint_vector(arity, num_values)));
    Factor g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = tuple_apply(op, f);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename JoinOp>
void time_join(JoinOp join_op) {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(uint_vector(arity, num_values));
    Factor g(uint_vector(arity, num_values));
    Factor h;
    uint_vector x = range(1, arity);
    uint_vector y = range(0, arity-1);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f.dims(x), g.dims(y));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename AggOp>
void time_aggregate(AggOp agg_op) {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(uint_vector(arity, num_values));
    Factor g;
    uint_vector retain = range(0, arity - 1);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = agg_op(f, retain);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename JoinOp, typename AggOp>
void time_join_aggregate(JoinOp join_op, AggOp agg_op) {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(uint_vector(arity, num_values));
    Factor g(uint_vector(arity, num_values));
    Factor h;
    uint_vector x = range(1, arity);
    uint_vector y = range(0, arity-1);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = agg_op(join_op(f.dims(x), g.dims(y)), 0);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}


template <typename Factor, typename JoinOp, typename AccuOp>
void time_join_accumulate(JoinOp join_op, AccuOp accu_op) {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(uint_vector(arity, num_values));
    Factor g(uint_vector(arity, num_values));
    uint_vector x = range(1, arity);
    uint_vector y = range(0, arity-1);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      result_t<Factor> e = accu_op(join_op(f.dims(x), g.dims(y)));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor>
void time_restrict() {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(uint_vector(arity, num_values));
    Factor g;
    uint_vector dims = {0};
    uint_vector vals = {1};
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f.restrict(dims, vals);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename RestrictOp>
void time_restrict_segment(RestrictOp restrict_op) {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(uint_vector(arity, num_values));
    Factor g;
    uint_vector vals = {1};
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = restrict_op(f, vals);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename UpdateOp>
void time_update(UpdateOp update_op) {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(uint_vector(arity, num_values));
    Factor g(uint_vector(arity-1, num_values));
    uint_vector x = range(1, arity);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      update_op(f.dims(x), g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename UpdateOp>
void time_restrict_update(UpdateOp update_op) {
  boost::timer t;
  std::cout << version(result_t<Factor>()) << std::flush;
  for (std::size_t arity = 1; arity < max_arity; ++arity) {
    Factor f(uint_vector(arity, num_values));
    Factor g(uint_vector(arity, num_values));
    uint_vector vals = {1};
    uint_vector x = range(0, arity-1);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      update_op(f.dims(x), g.tail(vals));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

struct ternary_plus {
  template <typename X, typename Y, typename Z>
  auto operator()(X&& x, Y&& y, Z&& z) {
    return std::forward<X>(x) + std::forward<Y>(y) + std::forward<Z>(z);
  }
};

int main(int argc, char** argv) {

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("num-values", po::value<std::size_t>(&num_values),
     "the number of values in the variables")
    ("max-arity", po::value<std::size_t>(&max_arity),
     "the maximum arity of the result")
    ("num-reps", po::value<std::size_t>(&num_reps),
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

  // construct the domains
  typedef experimental::probability_table<> ptable;
  typedef experimental::logarithmic_table<>ltable;

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "table * constant" << std::endl;
  time_transform<ptable, 1>(multiplied_by<double>(2));
  time_transform<ltable, 1>(multiplied_by<logd>(logd(2)));

  std::cout << std::endl << "table + table" << std::endl;
  time_transform<ptable, 2>(std::plus<>());
  time_transform<ltable, 2>(std::plus<>());

  std::cout << std::endl << "table + table + table" << std::endl;
  time_transform<ptable, 3>(ternary_plus());
  time_transform<ltable, 3>(ternary_plus());

  std::cout << std::endl << "table * table" << std::endl;
  time_join<ptable>(std::multiplies<>());
  time_join<ltable>(std::multiplies<>());

  std::cout << std::endl << "table / table" << std::endl;
  time_join<ptable>(std::divides<>());
  time_join<ltable>(std::divides<>());

  std::cout << std::endl << "table.marginal(dom)" << std::endl;
  time_aggregate<ptable>(member_marginal());
  time_aggregate<ltable>(member_marginal());

  std::cout << std::endl << "table.maximum(dom)" << std::endl;
  time_aggregate<ptable>(member_maximum());
  time_aggregate<ltable>(member_maximum());

  std::cout << std::endl << "(table * table).marginal(dom)" << std::endl;
  time_join_aggregate<ptable>(std::multiplies<>(), member_marginal());
  time_join_aggregate<ltable>(std::multiplies<>(), member_marginal());

  std::cout << std::endl << "(table * table).maximum(dom)" << std::endl;
  time_join_aggregate<ptable>(std::multiplies<>(), member_maximum());
  time_join_aggregate<ltable>(std::multiplies<>(), member_maximum());

  std::cout << std::endl << "(table * table).marginal()" << std::endl;
  time_join_accumulate<ptable>(std::multiplies<>(), member_marginal());
  time_join_accumulate<ltable>(std::multiplies<>(), member_marginal());

  std::cout << std::endl << "(table * table).maximum()" << std::endl;
  time_join_accumulate<ptable>(std::multiplies<>(), member_maximum());
  time_join_accumulate<ltable>(std::multiplies<>(), member_maximum());

  std::cout << std::endl << "table.restrict(dims, vals)" << std::endl;
  time_restrict<ptable>();
  time_restrict<ltable>();

  std::cout << std::endl << "table.head(vals)" << std::endl;
  time_restrict_segment<ptable>(member_head());
  time_restrict_segment<ltable>(member_head());

  std::cout << std::endl << "table.tail(vals)" << std::endl;
  time_restrict_segment<ptable>(member_tail());
  time_restrict_segment<ltable>(member_tail());

  std::cout << std::endl << "table *= table" << std::endl;
  time_update<ptable>(multiplies_assign<>());
  time_update<ltable>(multiplies_assign<>());

  std::cout << std::endl << "table /= table" << std::endl;
  time_update<ptable>(divides_assign<>());
  time_update<ltable>(divides_assign<>());

  std::cout << std::endl << "table *= table.restrict(a)" << std::endl;
  time_restrict_update<ptable>(multiplies_assign<>());
  time_restrict_update<ltable>(multiplies_assign<>());

  return 0;
}
