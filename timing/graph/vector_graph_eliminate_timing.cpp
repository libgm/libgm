#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/model/markov_structure.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <timing/timer.hpp>

#include <boost/program_options.hpp>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace po = boost::program_options;

using namespace libgm;

std::vector<size_t> side_lengths;
std::size_t num_reps = 0;

template <typename Strategy>
void time_eliminate(const char* label, const Strategy& strategy) {
  Timer t;
  std::cout << label << std::flush;
  for (size_t side : side_lengths) {
    libgm::MarkovStructure<GridArg> base_graph = make_grid_graph<int, int>(side, side).structure();
    std::vector<libgm::MarkovStructure<GridArg>> graphs(num_reps, base_graph);
    std::size_t eliminated = 0;

    t.restart();
    for (libgm::MarkovStructure<GridArg>& graph : graphs) {
      graph.eliminate(strategy, [&](size_t) { ++eliminated; });
    }

    std::cout << " " << t.elapsed() / num_reps << std::flush;
    (void)eliminated;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  size_t min_side;
  size_t step_size;
  size_t max_side = 0;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "prints out the usage")
    ("min-side", po::value<size_t>(&min_side)->default_value(1),
     "the initial grid side length")
    ("max-side", po::value<size_t>(&max_side),
     "the maximum grid side length")
    ("step-size", po::value<size_t>(&step_size)->default_value(1),
     "the increment in the grid side length")
    ("num-reps", po::value<size_t>(&num_reps),
     "the number of repetitions");

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  notify(vm);

  if (vm.count("help") || !max_side || !num_reps) {
    std::cout << desc;
    return 0;
  }

  std::cout << "Measuring VectorGraph::eliminate on square grid graphs with"
            << " min_side=" << min_side
            << " max_side=" << max_side
            << " step_size=" << step_size
            << " num_reps=" << num_reps
            << std::endl;

  for (size_t side = min_side; side <= max_side; side += step_size) {
    side_lengths.push_back(side);
  }

  std::cout << std::scientific << std::setprecision(3);

  std::cout << std::endl << "vector_graph.eliminate(strategy)" << std::endl;
  time_eliminate("min-degree", MinDegreeStrategy());
  time_eliminate("min-fill", MinFillStrategy());

  return 0;
}
