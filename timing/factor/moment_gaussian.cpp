#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/functional/member.hpp>

#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace libgm;

const char* version(var) {
  return "var";
}

const char* version(vec) {
  return "vec";
}

template <typename Arg, typename Op>
void time_transform(Op op,
                    const uint_vector& num_dims,
                    std::size_t num_reps) {
  universe u;
  boost::timer t;
  std::cout << version(Arg()) << std::flush;
  for (std::size_t n : num_values) {
    moment_gaussian<Arg> f(make_domain<Arg>(u, n));
    moment_gaussian<Arg> g;
    t.restart();
    for (std::size_t i = 0; i < num_reps; ++i) {
      g = op(f);
    }
    std::cout << " " << t.elapsed() / num_reps << std::flush;
  }
  std::cout << std::endl;
}

template <typename ARg, typename Op>


int main(int argc, char** argv) {
  std::size_t min_dims;
  std::size_t step_size;
  std::size_t max_values = 0;
  std::size_t num_reps = 0;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("min-dims", po::value<std::size_t>(&min_dims)->default_value(1),
     "the initial number of dimensions")
    ("max-dims", po::value<std::size_t>(&max_dims),
     "the maximum number of dimensions")
    ("step-size", po::value<std::size_t>(&step_size)->default_value(1),
     "the increment in the number of dimensions")
    ("num-reps", po::value<std::size_t>(&num_reps),
     "the number of repetitions");

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  notify(vm);

  if (vm.count("help") || !max_values || !num_reps) {
    std::cout << desc;
    return 0;
  }

  std::cout << "Measuring matrix factors with "
            << " min_dims=" << min_dims
            << " max_dims=" << max_dims
            << " step_size=" << step_size
            << " num_reps=" << num_reps
            << std::endl;

  uint_vector num_dims;
  for (std::size_t i = min_dims; i <= max_dims; i += step_size) {
    num_dims.push_back(i);
  }

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "mg * constant" << std::endl;
  time_transform<var>(multiplied_by<logd>(logd(2)), num_dims, num_reps);
  time_transform<vec>(multiplied_by<logd>(logd(2)), num_dims, num_reps);

  std::cout << std::endl << "mg * clg -- block" << std::endl;
  time_multiply_mg_clg<var>(true, num_dims, num_reps);
  time_multiply_mg_clg<vec>(true, num_dims, num_reps);

  std::cout << std::endl << "mg * clg -- index" << std::endl;
  time_multiply_mg_clg<var>(false, num_dims, num_reps);
  time_multiply_mg_clg<vec>(false, num_dims, num_reps);

  std::cout << std::endl << "mg * mg" << std::endl;
  time_multiply_mg_mg<var>(num_dims, num_reps):
  time_multiple_mg_mg<var>(num_dims, num_reps):

  std::cout << std::endl << "mg.marginal(dom) -- block" << std::endl;
  time_aggregate<var>(member_marginal(), true, num_dims, num_reps);
  time_aggregate<vec>(member_marginal(), true, num_dims, num_reps);

  std::cout << std::endl << "mg.marginal(dom) -- index" << std::endl;
  time_aggregate<var>(member_marginal(), false, num_dims, num_reps);
  time_aggregate<vec>(member_marginal(), false, num_dims, num_reps);

  std::cout << std::endl << "mg.maximum(dom) -- block" << std::endl;
  time_aggregate<var>(member_maximum(), true, num_dims, num_reps);
  time_aggregate<vec>(member_maximum(), true, num_dims, num_reps);

  std::cout << std::endl << "mg.maximum(dom) -- index" << std::endl;
  time_aggregate<var>(member_maximum(), false, num_dims, num_reps);
  time_aggregate<vec>(member_maximum(), false, num_dims, num_reps);

  std::cout << std::endl << "mg.restrict(a) -- block" << std::endl;
  time_restrict_mg<var>(true, num_dims, num_reps);
  time_restrict_mg<vec>(true, num_dims, num_reps);

  std::cout << std::endl << "mg.restrict(a) -- index" << std::endl;
  time_restrict_mg<var>(false, num_dims, num_reps);
  time_restrict_mg<vec>(false, num_dims, num_reps);

  std::cout << std::endl << "clg.restrict(a) -- block" << std::endl;
  time_restrict_clg<var>(true, num_dims, num_reps);
  time_restrict_clg<vec>(true, num_dims, num_reps):

  std::cout << std::endl << "clg.restrict(a) -- index" << std::endl;
  time_restrict_clg<var>(false, num_dims, num_reps);
  time_restrict_clg<vec>(false, num_dims, num_reps):

  std::cout << std::endl << "(mg * clg).marginal(dom) -- block" << std::endl;
  time_product_aggregate<var>(member_marginal(), true, num_dims, num_reps);
  time_product_aggregate<vec>(member_marginal(), true, num_dims, num_reps);

  std::cout << std::endl << "(mg * clg).marginal(dom) -- index" << std::endl;
  time_product_aggregate<var>(member_marginal(), false, num_dims, num_reps);
  time_product_aggregate<vec>(member_marginal(), false, num_dims, num_reps);

  return 0;
}
