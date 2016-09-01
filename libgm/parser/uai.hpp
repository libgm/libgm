#ifndef LIBGM_UAI_HPP
#define LIBGM_UAI_HPP

#include <libgm/factor/probability_table.hpp>
#include <libgm/model/factor_graph.hpp>
#include <libgm/parser/string_functions.hpp>
#include <libgm/parser/range_io.hpp>
#include <libgm/range/reversed.hpp>

#include <fstream>

namespace libgm {

  /**
   * This function parses the file format described in
   * http://www.cs.huji.ac.il/project/PASCAL/fileFormat.php
   *
   * \return the variables in the order specified in the model
   */
  template <typename Arg, typename RealType>
  domain<Arg>
  load_uai(const std::string& filename,
           std::function<Arg(std::size_t)> arg_gen,
           RealType zero,
           factor_graph<probability_table<Arg, RealType> >& model) {
    std::string line;
    std::size_t line_num = 0;
    std::string whitespace = " \t\r\n";
    std::vector<const char*> tokens;

    // Open the file and read the preamble
    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }
    libgm::getline(in, line, line_num);
    std::string preamble = trim(line);
    if (preamble != "MARKOV" && preamble != "BAYES") {
      throw std::runtime_error("Line " + std::to_string(line_num) +
                               ": UAI format preamble not found!");
    }

    // The first section is:
    // # variables
    // [arity of all variables in the model]
    libgm::getline_nonempty(in, line, line_num, whitespace);
    domain<Arg> variables;
    try {
      variables.resize(parse_string<std::size_t>(line));
    } catch (std::invalid_argument&) {
      throw std::runtime_error("Line " + std::to_string(line_num) +
                               ": Cannot parse the number of variables!");
    }
    libgm::getline_nonempty(in, line, line_num, whitespace);
    string_split(line, whitespace, tokens);
    if (tokens.size() != variables.size()) {
      throw std::runtime_error("Line " + std::to_string(line_num) +
                               ": Invalid number of variable entries!");
    }
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      try {
        variables[i] = arg_gen(parse_string<std::size_t>(tokens[i]));
      } catch (std::invalid_argument&) {
        throw std::runtime_error("Line " + std::to_string(line_num) +
                                ": Cannot parse variable size " + tokens[i]);
      }
    }

    // The second section is
    // # factors
    // [domain of factor 1]
    // [domain of factor 2]
    // ....
    libgm::getline_nonempty(in, line, line_num, whitespace);
    std::vector<domain<Arg> > domains;
    try {
      domains.resize(parse_string<std::size_t>(line));
    } catch (std::invalid_argument&) {
      throw std::runtime_error("Line " + std::to_string(line_num) +
                               ": Cannot parse the number of factors!");
    }
    for (domain<Arg>& domain : domains) {
      libgm::getline_nonempty(in, line, line_num, whitespace);
      string_split(line, whitespace, tokens);
      if (tokens.empty()) {
        throw std::runtime_error("Line " + std::to_string(line_num) +
                                 ": Empty factor domain description!");
      }
      if (parse_string<std::size_t>(tokens[0]) != tokens.size() - 1) {
        throw std::runtime_error("Line " + std::to_string(line_num) +
                                 ": Invalid number of variables in domain!");
      }
      for (std::size_t i = 1; i < tokens.size(); ++i) {
        try {
          domain.push_back(variables.at(parse_string<std::size_t>(tokens[i])));
        } catch (std::invalid_argument&) {
          throw std::runtime_error("Line " + std::to_string(line_num) +
                                   ": Cannot parse variable id!");
        }
      }
      // match up with the order variables in the function entries (LSB last)
      std::reverse(domain.begin(), domain.end());
    }

    // Third section is
    // [#vals in factor 1]
    //  [list of values in factor 1]
    // [#vals in factor 2]
    //  [list of values in factor 2]
    for (std::size_t j = 0; j < domains.size(); ++j) {
      probability_table<Arg, RealType> factor(domains[j]);

      // first, check that the number of values matches the factor
      libgm::getline_nonempty(in, line, line_num, whitespace);
      try {
        std::size_t count = parse_string<std::size_t>(line);
        if (count != factor.size()) {
          throw std::runtime_error("Line " + std::to_string(line_num) +
                                   ": Invalid number of table values!");
        }
      } catch (std::invalid_argument&) {
        throw std::runtime_error("Line " + std::to_string(line_num) +
                                 ": Cannot parse the number of values!");
      }

      // next, parse the values
      RealType* dest = factor.begin();
      while (dest != factor.end()) {
        libgm::getline_nonempty(in, line, line_num, whitespace);
        string_split(line, whitespace, tokens);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
          if (dest == factor.end()) {
            throw std::runtime_error("Line " + std::to_string(line_num) +
                                     ": Too many entries for factor "+
                                     std::to_string(j) + "!");
          }
          *dest = parse_string<RealType>(tokens[i]);
          if (*dest == RealType(0)) { *dest = zero; }
          ++dest;
        }
      }

      // add the factor to the model
      model.add_factor(factor);
    }

    return variables;
  } // load_uai


  /**
   * Saves a graphical model in the UAI format described at
   * http://www.cs.huji.ac.il/project/PASCAL/fileFormat.php
   * using the specified ordering of variables. All variables must be present.
   */
  template <typename Model>
  void save_uai(const std::string& filename,
                const std::string& preamble,
                const Model& model,
                const domain<typename Model::argument_type>& arguments) {
    using argument_type = typename Model::argument_type;
    using hasher = typename argument_traits<argument_type>::hasher;

    std::ofstream out(filename);

    // output the preamble
    out << preamble << std::endl;

    // output the number of variables and number of values for each variable
    std::unordered_map<argument_type, std::size_t, hasher> arg_map;
    std::size_t index = 0;
    out << arguments.size() << std::endl;
    for (argument_type arg : arguments) {
      if (index > 0) {
        out << " ";
      }
      out << argument_traits<argument_type>::num_values(arg);
      arg_map[arg] = index++;
    }
    out << std::endl;

    // count the number of non-empty factors
    std::size_t count = 0;
    for (const auto& factor : model) {
      if (!factor.empty()) { ++count; }
    }
    out << count << std::endl;

    // output the number of factors and the domain for each factor
    for (const auto& factor : model) {
      if (!factor.empty()) {
        out << factor.arguments().size();
        for (argument_type arg : make_reversed(factor.arguments())) {
          out << " " << arg_map.at(arg);
        }
        out << std::endl;
      }
    }

    // output the factor values
    for (const auto& factor : model) {
      if (!factor.empty()) {
        out << std::endl << factor.size() << std::endl;
        print_range(out, factor.begin(), factor.end(), ' ', ' ', '\n');
      }
    }
  } // save_uai


  /**
   * Saves a graphical model in the UAI format described at
   * http://www.cs.huji.ac.il/project/PASCAL/fileFormat.php
   * using all the variables in the model.
   */
  template <typename Model>
  void save_uai(const std::string& filename,
                const std::string& preamble,
                const Model& model) {
    using domain_type = typename Model::domain_type;
    save_uai(filename, preamble, model, domain_type(model.arguments()));
  } // save_uai

} // namespace libgm

#endif
