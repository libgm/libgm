#include <libgm/argument/var.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/experimental/probability_table.hpp>

#include <boost/timer.hpp>

using namespace libgm;

template <typename Factor>
void test_unary_transform(const char* version,
                          const domain<var>& args,
                          std::size_t num_reps) {
  boost::timer t;
  std::cout << version << std::flush;
  for (std::size_t arity = 0; arity < args.size(); ++arity) {
    Factor f(args.prefix(arity));
    t.restart();
    Factor g;
    for (std::size_t i = 1; i < num_reps; ++i) {
      g = f * 2.0;
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor>
void test_binary_transform(const char* version,
                           const domain<var>& args,
                           std::size_t num_reps) {
  boost::timer t;
  std::cout << version << std::flush;
  for (std::size_t arity = 1; arity < args.size(); ++arity) {
    Factor f0(args.prefix(arity));
    Factor f1(args.prefix(arity));
    t.restart();
    Factor g;
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = f0 + f1;
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor>
void test_ternary_transform(const char* version,
                            const domain<var>& args,
                            std::size_t num_reps) {
  boost::timer t;
  std::cout << version << std::flush;
  for (std::size_t arity = 1; arity < args.size(); ++arity) {
    Factor f0(args.prefix(arity));
    Factor f1(args.prefix(arity));
    Factor f2(args.prefix(arity));
    t.restart();
    Factor h;
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = f0 + f1 + f2;
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename Factor>
void test_join_aggregate(const char* version,
                         const domain<var>& x,
                         const domain<var>& y,
                         std::size_t num_reps) {
  boost::timer t;
  std::cout << version << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g(y.prefix(arity));
    domain<var> z = {x[0]};
    t.restart();
    Factor h;
    for (std::size_t i = 0; i < num_reps; ++i) {
      h = (f*g).marginal(z);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}


template <typename Factor>
void test_join_accumulate(const char* version,
                          const domain<var>& x,
                          const domain<var>& y,
                          std::size_t num_reps) {
  boost::timer t;
  std::cout << version << std::flush;
  for (std::size_t arity = 1; arity < x.size(); ++arity) {
    Factor f(x.prefix(arity));
    Factor g(y.prefix(arity));
    domain<var> z = {x[0]};
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      double e = (f*g).marginal();
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  typedef libgm::probability_table<var> prod_table;
  typedef libgm::experimental::probability_table<var> exp_table;

  std::size_t num_values = argc > 1 ? atoi(argv[1]) : 5;
  std::size_t max_arity  = argc > 2 ? atoi(argv[2]) : 5;
  std::size_t num_reps   = argc > 3 ? atoi(argv[3]) : 10;

  universe u;
  domain<var> x, y, z;
  for (std::size_t i = 0; i < max_arity; ++i) {
    var v = var::discrete(u, "v" + std::to_string(i), num_values);
    if (i < max_arity-1) { x.push_back(v); }
    if (i > 0) { y.push_back(v); }
    z.push_back(v);
  }


  std::cout << "Multiplication by a constant" << std::endl;
  test_unary_transform<prod_table>("production", z, num_reps);
  test_unary_transform<exp_table>("experimental", z, num_reps);

  std::cout << "Adding two tables" << std::endl;
  test_binary_transform<prod_table>("production", z, num_reps);
  test_binary_transform<exp_table>("experimental", z, num_reps);

  std::cout << "Adding three tables" << std::endl;
  test_ternary_transform<prod_table>("production", z, num_reps);
  test_ternary_transform<exp_table>("experimental", z, num_reps);

  std::cout << "Join-aggregate" << std::endl;
  test_join_aggregate<prod_table>("production", x, y, num_reps);
  test_join_aggregate<exp_table>("experimental", x, y, num_reps);

  std::cout << "Join-accumulate" << std::endl;
  test_join_accumulate<prod_table>("production", x, y, num_reps);
  test_join_accumulate<exp_table>("experimental", x, y, num_reps);
}
