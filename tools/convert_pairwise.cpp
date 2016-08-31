#include <libgm/argument/var.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/model/factor_graph.hpp>
#include <libgm/model/pairwise_markov_network.hpp>
#include <libgm/parser/uai.hpp>

#include <iostream>

int main(int argc, char** argv) {
  using namespace libgm;

  if (argc != 3) {
    std::cout << "Syntax: " << argv[0] << "input-file output-file" << std::endl;
    return 1;
  }

  std::string input = argv[1];
  std::string output = argv[2];

  // load the original
  universe u;
  factor_graph<probability_table<var> > model;
  domain<var> variables;
  try {
    variables = load_uai(input, var::discrete_generator(u, "x"), 0.0, model);
  } catch(std::runtime_error& e) {
    std::cout << "Error loading the model. " << e.what() << std::endl;
    return 2;
  }

  // convert the factors
  pairwise_markov_network<probability_table<var> > mn;
  for (const auto& factor : model) {
    var new_var = mn.add_nary(factor, var::discrete_generator(u, "y"));
    if (new_var != var()) {
      variables.push_back(new_var);
    }
  }

  // save the model
  save_uai(output, "MARKOV", mn, variables);

  return 0;
}


