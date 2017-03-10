#ifndef LIBGM_DATASET_FORMAT_HPP
#define LIBGM_DATASET_FORMAT_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/argument/sequence.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/parser/simple_config.hpp>
#include <libgm/parser/string_functions.hpp>

#include <algorithm>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <string>

namespace libgm {

  // Forward declarations
  class var;
  class vec;

  /**
   * A utility class for representing the formatting and the variables /
   * processes used in a file, such as the datasets from the UCI ML Repository.
   * The variables, processes, and formatting parameters can be loaded from
   * configuration files using the load_config() and saved using save_config().
   *
   * \tparam Arg an argument type, currently either libgm::var or libgm::vec
   *
   * \see dataset, sequence_dataset
   */
  template <typename Arg>
  struct text_dataset_format {
    static_assert(std::is_same<Arg,var>::value || std::is_same<Arg,vec>::value,
                  "Arg must be either libgm::var or libm::vec");

    //! Specifies the separator for the fields (default whitespace).
    std::string separator;

    //! Specifies the symbol for missing values.
    std::string missing;

    //! The number of lines at the beginning of the file to skip (default = 0).
    std::size_t skip_rows;

    //! The number of columns at the start of each line to skip (default = 0).
    std::size_t skip_cols;

    //! Indicates if the dataset is weighted (default = false).
    bool weighted;

    //! The variables / vectors in the given ordering.
    domain<Arg> variables;

    //! The discrete-time processes in the given ordering.
    domain<sequence<Arg> > sequences;

    /**
     * Constructs the symbolic format with default parameters.
     */
    text_dataset_format()
      : skip_rows(0), skip_cols(0), weighted(false) { }

    // Queries
    //--------------------------------------------------------------------------

    //! Returns true if all the variables in the format are discrete.
    bool all_variables_discrete() const {
      return std::all_of(variables.begin(), variables.end(),
                         [](Arg arg) { return arg.discrete(); });
    }

    //! Returns true if all the variables in teh format are continuous.
    bool all_variables_continuous() const {
      return std::all_of(variables.begin(), variables.end(),
                         [](Arg arg) { return arg.continuous(); });
    }

    //! Returns true if all the discrete-time processes are discrete-valued.
    bool all_sequences_discrete() const {
      return std::all_of(sequences.begin(), sequences.end(),
                         [](sequence<Arg> s) { return s.discrete(); });
    }

    //! Returns true if all the discrete-time processes are continuous-valued.
    bool all_sequences_continuous() const {
      return std::all_of(sequences.begin(), sequences.end(),
                         [](sequence<Arg> s) { return s.continuous(); });
    }

    /**
     * Finds a variable with the given name in the format.
     * \return the variable found or Arg() if no such variable exists.
     */
    Arg find_variable(const std::string& name) const {
      auto it = std::find_if(variables.begin(), variables.end(),
                             [&](Arg arg) { return arg.desc()->name == name; });
      return (it != variables.end()) ? *it : Arg();
    }

    /**
     * Finds a discrete-time process with the given name in the format.
     * \return the process found or dprocess() if no such process exists.
     */
    sequence<Arg> find_sequence(const std::string& name) const {
      auto it = std::find_if(sequences.begin(), sequences.end(),
                             [&](sequence<Arg> seq) {
                               return seq.desc()->name == name; });
      return (it != sequences.end()) ? *it : sequence<Arg>();
    }

    /**
     * Tokenizes a line according to the format and stores the result in an
     * array of C strings.
     *
     * \param num_values expected number of columns (ignoring the skipped ones)
     * \param input line (modified)
     * \param current line number (updated)
     * \param tokens output tokens (overwritten)
     * \throw runtime_error if the number of columns in the input data does not
     *        match the format.
     * \return false if the line should be ignored
     */
    bool tokenize(std::size_t num_values,
                  std::string& line,
                  std::size_t& line_number,
                  std::vector<const char*>& tokens) const {
      if (++line_number <= skip_rows) {
        return false;
      }
      string_split(line, separator.empty() ? "\t " : separator, tokens);
      if (tokens.empty()) {
        return false;
      }
      std::size_t expected_cols = skip_cols + num_values + weighted;
      if (tokens.size() == expected_cols) {
        return true;
      } else {
        std::ostringstream out;
        out << "Line " << line_number << ": invalid number of columns "
            << "(expected " << expected_cols
            << ", found " << tokens.size() << ")";
        throw std::runtime_error(out.str());
      }
    }

    // Functions for format I/O
    //--------------------------------------------------------------------------

    /**
     * Loads the text_dataset_format from a configuration file with the following
     * format:
     *
     * [variables]
     * variable_name=value0,value1,...,valuek-1 (where k >= 2) OR
     * variable_name=finite(k) (where k >= 2) OR
     * variable_name=vector(k) (where k >= 1)
     * other_variable_name=...
     *
     * [discrete_processes]
     * process_name=value0,value1,...,valuek-1 (where k >= 2) OR
     * process_name=finite(k) (where k >= 2) OR
     * process_name=vector(k) (where k >= 1)
     * other_process_name=...
     *
     * [options]
     * separator="\t" (optional)
     * skip_rows=1    (optional)
     * skip_cols=0    (optional)
     * weighted=1     (optional)
     *
     * Comments can be prepended with #. Whitespace is ignored. Sections order
     * can be swapped.
     */
    void load_config(const std::string& filename, universe& u) {
      simple_config config;
      config.load(filename);

      // load the components of a config
      domain<Arg> tmp;
      load_options(config);
      load_arguments(u, config["variables"], variables);
      load_arguments(u, config["sequences"], tmp);
      sequences = argument_cast<sequence<Arg>>(tmp);

      // empty formats are not allowed
      if (variables.empty() && sequences.empty()) {
        throw std::out_of_range(
          "Please specify at least one variable or process"
        );
      }
    }

    /**
     * Saves the text_dataset_format to a configuration file with the format
     * given in load().
     */
    void save_config(const std::string& filename) const {
      simple_config config;
      save_options(config);
      save_arguments(variables, config["variables"]);
      save_arguments(sequences(0), config["sequences"]);
      config.save(filename);
    }

  private:
    typedef std::pair<std::string, std::string> config_entry;

    //! Loads the options.
    void load_options(simple_config& config) {
      typedef std::pair<std::string, std::string> config_entry;
      for (const config_entry& entry : config["options"]) {
        if (entry.first == "separator") {
          separator = parse_escaped(entry.second);
        } else if (entry.first == "missing") {
          missing = parse_escaped(entry.second);
        } else if (entry.first == "skip_rows") {
          skip_rows = parse_string<std::size_t>(entry.second);
        } else if (entry.first == "skip_cols") {
          skip_cols = parse_string<std::size_t>(entry.second);
        } else if (entry.first == "weighted") {
          weighted = parse_string<bool>(entry.second);
        } else {
          std::cerr << "Unknown option \"" << entry.first
                    << "\", ignoring" << std::endl;
        }
      }
    }

    //! store the options
    void save_options(simple_config& config) const {
      config.add("options", "separator", escape_string(separator));
      config.add("options", "missing", escape_string(missing));
      config.add("options", "skip_rows", skip_rows);
      config.add("options", "skip_cols", skip_cols);
      config.add("options", "weighted", weighted);
    }

    //! A helper function that creates arguments from the config entries.
    void load_arguments(universe& u,
                        const std::vector<config_entry>& entries,
                        domain<Arg>& arguments) const {
      std::regex typed_regex("([a-z]+)(\\(([\\d\\s,]+)\\))?");
      std::regex level_regex("[^\\(\\)]+");
      std::smatch match;
      for (const config_entry& entry : entries) {
        // parse the type and the parameters / levels
        std::string type;
        std::vector<std::string> param;
        if (std::regex_match(entry.second, match, typed_regex)) {
          type = match[1].str();
          string_split(match[3].str(), ", \t\n\r\f\v", param);
        } else if (std::regex_match(entry.second, match, level_regex)) {
          string_split(match[0].str(), ", \t\n\r\f\v", param);
        } else {
          throw std::invalid_argument(
            "Invalid specification of argument \"" + entry.first + "\": " +
            entry.second
          );
        }

        // construct the argument
        if (type.empty()) { // discrete argument with named levels
          if (std::find(param.begin(), param.end(), missing) != param.end()) {
            throw std::invalid_argument(
              "The missing value symbol \"" + missing + "\" must not be " +
              "a level of discrete argument \"" + entry.first + "\""
            );
          }
          arguments.push_back(Arg::discrete(u, entry.first, param));
        } else if (type == "discrete") {
          std::vector<std::size_t> num_values(param.size());
          for (std::size_t i = 0; i < param.size(); ++i) {
            if (!parse_string(param[i], num_values[i])) {
              throw std::invalid_argument(
                "Invalid specification of discrete argument \"" + entry.first +
                "\": " + entry.second + "; cannot parse \"" + param[i] + "\""
              );
            }
          }
          arguments.push_back(Arg::discrete(u, entry.first, num_values));
        } else if (type == "continuous") {
          std::size_t dim;
          if (param.empty()) {
            arguments.push_back(Arg::continuous(u, entry.first));
          } else if (param.size() == 1 && parse_string(param[0], dim)) {
            arguments.push_back(Arg::continuous(u, entry.first, dim));
          } else {
            throw std::invalid_argument(
              "Invalid specification of continuous argument \"" + entry.first +
              "\": " + entry.second
            );
          }
        } else {
          throw std::invalid_argument("Invalid argument type \"" + type + "\"");
        }
      }
    }

    //! store the arguments
    void save_arguments(const domain<Arg>& arguments,
                        std::vector<config_entry>& entries) const {
      for (Arg arg : arguments) {
        const std::string& name = arg.desc()->name;
        if (arg.continuous() && arg.num_dimensions() == 1) {
          entries.emplace_back(name, "continuous");
        } else if (arg.continuous()) {
          std::string dimensions = to_string(arg.num_dimensions());
          entries.emplace_back(name, "continuous(" + dimensions + ")");
        } else if (arg.discrete() && arg.desc()->levels.empty()) {
          std::string cardinality = cardinality_string(arg.desc()->cardinality);
          entries.emplace_back(name, "discrete(" + cardinality + ")");
        } else if (arg.discrete()) {
          std::string levels = string_join(",", arg.desc()->levels);
          entries.emplace_back(name, levels);
        } else {
          throw std::logic_error("Unsupported argument \"" + name + '"');
        }
      }
    }

    //! converts a cardinality to string
    static std::string cardinality_string(std::size_t card) {
      return std::to_string(card);
    }

    //! converts a cardonality vector to string
    static std::string cardinality_string(const std::vector<std::size_t>& card) {
      std::ostringstream out;
      for (std::size_t i = 0; i < card.size(); ++i) {
        if (i > 0) { out << ','; }
        out << card[i];
      }
      return out.str();
    }

  }; // struct text_dataset_format

} // namespace libgm

#endif
