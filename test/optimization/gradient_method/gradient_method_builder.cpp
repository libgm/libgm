#include <libgm/optimization/gradient_method/gradient_method_builder.hpp>

#include <armadillo>

#include <boost/shared_ptr.hpp>

int main(int argc, char** argv) {
  using namespace libgm;

  // Register the options
  namespace po = boost::program_options;
  po::options_description desc("gradient_method_builder test");
  desc.add_options()("help", "Print command options");
  gradient_method_builder<arma::vec> builder;
  builder.add_options(desc);

  // Parse the options
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Print help if requested
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  
  // Create the gradient_method object
  boost::shared_ptr<gradient_method<arma::vec> > gm(builder.get(NULL));
  std::cout << *gm << std::endl;
  
  return 0;
}
