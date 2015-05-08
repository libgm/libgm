#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <limits>

#include <boost/program_options.hpp>

#include <libgm/model/factor_graph_model.hpp>
#include <libgm/factor/table_factor.hpp>
#include <libgm/factor/canonical_table.hpp>
#include <libgm/parallel/timer.hpp>
#include <libgm/parser/protein.hpp>
#include <libgm/inference/loopy/blf_residual_splash_bp.hpp>
#include <libgm/inference/sampling/gibbs_engine.hpp>
#include <libgm/macros_def.hpp>


// Rename the program options namespace
namespace po = boost::program_options;

using namespace std;
using namespace libgm;


typedef factor_graph_model<canonical_table> factor_graph_model_type;
typedef factor_graph_model_type::factor_type factor_type;
typedef factor_graph_model_type::vertex_type vertex_type;
typedef factor_graph_model_type::variable_type variable_type;

typedef blf_residual_splash_bp<factor_type> engine_type;
//typedef residual_splash_bp<factor_type> engine_type;
//typedef round_robin_bp<factor_type> engine_type;


void save_beliefs(factor_graph_model_type& fg,
                  engine_type& engine,
                  string& output_filename) {
  // Create an ouptut filestream
  ofstream fout(output_filename.c_str());
  assert(fout.good());
  foreach(variable_type* v, fg.arguments()) {
    fout << v << ":->"  << engine.belief(v).v(1) << endl;
  }
  fout.flush();
  fout.close();
}


int main(int argc, char* argv[]) {
  string input_dir;
  string output_filename;
  double bound;
  double damping;
  std::size_t splashv;
  bool adjonly;
  std::size_t adjprop;
  // Parse the input
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("infn", po::value<string>(&input_dir),
     "MLN Factor graph")
    ("outfn", po::value<string>(&output_filename),
     "file to write the beliefs")
    ("help", "produce help message")
    ("bound", po::value<double>(&bound)->default_value(1E-5),
     "accuracy bound")
    ("splashv", po::value<std::size_t>(&splashv)->default_value(100),
     "volume of splash")
    ("damping", po::value<double>(&damping)->default_value(0.6),
     "volume of splash")
    ("adjonly", "Output adjacency file only")
    ("adjprop", po::value<std::size_t>(&adjprop)->default_value(0),
     "number of MK propagations");
  po::positional_options_description pos_opts;
  pos_opts.add("infn",1).add("outfn",1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);
  if(vm.count("help") || !vm.count("infn") || (!vm.count("outfn") && !vm.count("adjonly"))) {
    cout << "Usage: " << argv[0] << " [options] infn outfn" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }
	adjonly = vm.count("adjonly") > 0;
  cout << "==========================================================" << endl
       << "Program Settings: " << endl
       << "Input filename:  " << input_dir << endl
       << "Output filename: " << output_filename << endl
       << "bound:           " << bound << endl
       << "splash volume:   " << splashv << endl
       << "damping:         " << damping << endl
       << "==========================================================" << endl;

  universe universe;
  factor_graph_model_type fg;

  // Do the actual parsing.
  std::cout << "Loading..."; std::cout.flush();
  parse_protein(universe, fg, input_dir+"/network.bin");
  protein_truth_asg_type truth_asgs;
  protein_load_truth_data(input_dir, truth_asgs);

  fg.print_degree_distribution();
  std::cout << "Simplifying..."; std::cout.flush();
  fg.simplify_stable();
  fg.normalize();
  std::cout << "done!" << std::endl;

  ofstream fout("adjacency.txt");
  fout << fg;
  fout.close();

  cout << "Finished parsing: " << fg.arguments().size()
       << " variables and " << fg.size() << " factors."
       << endl;



  timer ti;
  ti.start();
  cout << "Performing Inference:" << endl;
  engine_type engine(&fg,
                              splashv,
                              bound,
                              damping);


  engine.run();

  fout.open("degcounts.txt");
  foreach(vertex_type v, fg.vertices()) {
    fout << fg.num_neighbors(v) << ", "
         << engine.update_count(v)  << std::endl;
  } // end of foreach vertex update count
  fout.close();

  std::map<vertex_type, factor_type> blfs;
  engine.belief(blfs);
  finite_assignment mapassg;
  engine.map_assignment(mapassg);

  std::cout.precision(10);
  std::cout << "Energy: " << fg.bethe(blfs) << std::endl;
  std::cout << "Log Likelihood: " << fg.log_likelihood(mapassg) << std::endl;

  cout << "Saving beliefs:" << endl;
  save_beliefs(fg, engine, output_filename);
  double error = protein_compute_error(fg, engine, truth_asgs);

  cout << "Finished!" << endl;
  cout << "Error: " << error << endl;
  cout << "Took " << ti.current_time() << "s" << endl;
  return EXIT_SUCCESS;
} // End of main

#include <libgm/macros_undef.hpp>
//End of file




