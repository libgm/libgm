#include <libgm/argument/var.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/canonical_table.hpp>
#include <libgm/functional/assign.hpp>
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
void time_transform(Op op, const domain<var>& args, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t arity = 0; arity < args.size(); ++arity) {
    auto f = tuple_rep<N>(Factor(args.prefix(arity)));
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
void time_join(JoinOp join_op,
               const domain<var>& x,
               const domain<var>& y,
               std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g(y.prefix(arity));
    Factor h;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = join_op(f, g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename AggOp>
void time_aggregate(AggOp agg_op,
                    const domain<var>& x,
                    std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g;
    domain<var> retain = x.prefix(arity - 1);
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = agg_op(f, retain);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename JoinOp, typename AggOp>
void time_join_aggregate(JoinOp join_op,
                         AggOp agg_op,
                         const domain<var>& x,
                         const domain<var>& y,
                         std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g(y.prefix(arity));
    Factor h;
    domain<var> z = { x[0] };
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = agg_op(join_op(f, g), z);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}


template <typename Factor, typename JoinOp, typename AccuOp>
void time_join_accumulate(JoinOp join_op,
                          AccuOp accu_op,
                          const domain<var>& x,
                          const domain<var>& y,
                          std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g(y.prefix(arity));
    domain<var> z = {x[0]};
    t.restart();
    typename Factor::result_type e;
    for (std::size_t i = 0; i < num_reps; ++i) {
      e += accu_op(join_op(f, g));
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor>
void time_restrict(bool prefix, const domain<var>& x, std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g;
    uint_assignment<var> a = { { x[prefix ? arity - 1 : 0], 1} };
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f.restrict(a);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename UpdateOp>
void time_update(UpdateOp update_op,
                 const domain<var>& x,
                 std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g(x.prefix(arity - 1));
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      update_op(f, g);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor, typename UpdateOp>
void time_restrict_update(UpdateOp update_op,
                          const domain<var>& x,
                          std::size_t num_reps) {
  boost::timer t;
  std::cout << version(typename Factor::result_type()) << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g(x.prefix(arity));
    uint_assignment<var> a = { { x[0], 1 } };
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      update_op(f, g.restrict(a));
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

  std::size_t num_values = 0;
  std::size_t max_arity = 0;
  std::size_t num_reps = 0;

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
  universe u;
  domain<var> x, y, z;
  for (std::size_t i = 0; i < max_arity; ++i) {
    var v = var::discrete(u, "v" + std::to_string(i), num_values);
    if (i < max_arity-1) { x.push_back(v); }
    if (i > 0) { y.push_back(v); }
    z.push_back(v);
  }

  typedef probability_table<var> ptable;
  typedef canonical_table<var> ltable;
  std::multiplies<> mult;
  std::divides<> div;

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "table * constant" << std::endl;
  time_transform<ptable, 1>(multiplied_by<double>(2), z, num_reps);
  time_transform<ltable, 1>(multiplied_by<logd>(logd(2)), z, num_reps);

  std::cout << std::endl << "table + table" << std::endl;
  time_transform<ptable, 2>(std::plus<>(), z, num_reps);
  time_transform<ltable, 2>(std::plus<>(), z, num_reps);

  std::cout << std::endl << "table + table + table" << std::endl;
  time_transform<ptable, 3>(ternary_plus(), z, num_reps);
  time_transform<ltable, 3>(ternary_plus(), z, num_reps);

  std::cout << std::endl << "table * table" << std::endl;
  time_join<ptable>(mult, x, y, num_reps);
  time_join<ltable>(mult, x, y, num_reps);

  std::cout << std::endl << "table / table" << std::endl;
  time_join<ptable>(div, x, y, num_reps);
  time_join<ltable>(div, x, y, num_reps);

  std::cout << std::endl << "table.marginal(dom)" << std::endl;
  time_aggregate<ptable>(member_marginal(), z, num_reps);
  time_aggregate<ltable>(member_marginal(), z, num_reps);

  std::cout << std::endl << "table.maximum(dom)" << std::endl;
  time_aggregate<ptable>(member_maximum(), z, num_reps);
  time_aggregate<ltable>(member_maximum(), z, num_reps);

  std::cout << std::endl << "(table * table).marginal(dom)" << std::endl;
  time_join_aggregate<ptable>(mult, member_marginal(), x, y, num_reps);
  time_join_aggregate<ltable>(mult, member_marginal(), x, y, num_reps);

  std::cout << std::endl << "(table * table).maximum(dom)" << std::endl;
  time_join_aggregate<ptable>(mult, member_maximum(), x, y, num_reps);
  time_join_aggregate<ltable>(mult, member_maximum(), x, y, num_reps);

  std::cout << std::endl << "(table * table).marginal()" << std::endl;
  time_join_accumulate<ptable>(mult, member_marginal(), x, y, num_reps);
  time_join_accumulate<ltable>(mult, member_marginal(), x, y, num_reps);

  std::cout << std::endl << "(table * table).maximum()" << std::endl;
  time_join_accumulate<ptable>(mult, member_maximum(), x, y, num_reps);
  time_join_accumulate<ltable>(mult, member_maximum(), x, y, num_reps);

  std::cout << std::endl << "table.restrict(a) -- general" << std::endl;
  time_restrict<ptable>(false, z, num_reps);
  time_restrict<ltable>(false, z, num_reps);

  std::cout << std::endl << "table.restrict(a) -- prefix" << std::endl;
  time_restrict<ptable>(true, z, num_reps);
  time_restrict<ltable>(true, z, num_reps);

  std::cout << std::endl << "table *= table" << std::endl;
  time_update<ptable>(multiplies_assign<>(), z, num_reps);
  time_update<ltable>(multiplies_assign<>(), z, num_reps);

  std::cout << std::endl << "table /= table" << std::endl;
  time_update<ptable>(divides_assign<>(), z, num_reps);
  time_update<ltable>(divides_assign<>(), z, num_reps);

  std::cout << std::endl << "table *= table.restrict(a)" << std::endl;
  time_restrict_update<ptable>(multiplies_assign<>(), z, num_reps);
  time_restrict_update<ltable>(multiplies_assign<>(), z, num_reps);

  return 0;
}
