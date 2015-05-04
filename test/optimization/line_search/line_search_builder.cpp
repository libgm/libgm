#include <libgm/optimization/line_search/line_search_builder.hpp>

#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/math/eigen/optimization.hpp>

#include <memory>

int main(int argc, char** argv) {
  using namespace libgm;

  // Register the options
  namespace po = boost::program_options;
  po::options_description desc("line_search_builder test");
  desc.add_options()("help", "Print command options");
  line_search_builder<dynamic_vector<double>> builder;
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
  
  // Create the line_search object
  std::unique_ptr<line_search<dynamic_vector<double>>> ls(builder.get());
  // TODO: change to std::unique_ptr
  std::cout << *ls << std::endl;
  
  return 0;
}
