#ifndef LIBGM_DATASET_FORMAT_HPP
#define LIBGM_DATASET_FORMAT_HPP

#include <libgm/argument/process.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/variable.hpp>
#include <libgm/parser/simple_config.hpp>
#include <libgm/parser/string_functions.hpp>

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace libgm {

  /**
   * A utility class for representing the formatting and the variables /
   * processes used in a file, such as the datasets from the UCI ML Repository.
   * The variables, processes, and formatting parameters can be loaded from
   * configuration files using the load_config() and saved using save_config().
   *
   * \see uint_dataset, real_dataset, hybrid_dataset,
   *      uint_sequence_dataset, real_sequence_dataset, hybrid_sequence_dataset
   */
  struct text_dataset_format {

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

    //! The variables in the format in the given ordering.
    std::vector<variable> variables;

    //! The discrete-time processes in the format in the given ordering.
    std::vector<dprocess> dprocesses;

    /**
     * Constructs the symbolic format with default parameters.
     */
    text_dataset_format()
      : skip_rows(0), skip_cols(0), weighted(false) { }

    // Queries
    //========================================================================

    //! Returns true if all the variables in the format are discrete.
    bool all_variables_discrete() const {
      return std::all_of(variables.begin(), variables.end(),
                         [](variable v) { return v.is_discrete(); });
    }

    //! Returns true if all the variables in teh format are continuous.
    bool all_variables_continuous() const {
      return std::all_of(variables.begin(), variables.end(),
                         [](variable v) { return v.is_continuous(); });
    }

    //! Returns true if all the discrete-time processes are discrete-valued.
    bool all_dprocesses_discrete() const {
      return std::all_of(dprocesses.begin(), dprocesses.end(),
                         [](dprocess p) { return p.is_discrete(); });
    }

    //! Returns true if all the discrete-time processes are continuous-valued.
    bool all_dprocesses_continuous() const {
      return std::all_of(dprocesses.begin(), dprocesses.end(),
                         [](dprocess p) { return p.is_continuous(); });
    }

    /**
     * Finds a variable with the given name in the format.
     * \return the variable found or variable() if no such variable exists.
     */
    variable find_variable(const std::string& name) const {
      auto it = std::find_if(variables.begin(), variables.end(),
                             [&](variable v) { return v.name() == name; });
      return (it != variables.end()) ? *it : variable();
    }

    /**
     * Finds a discrete-time process with the given name in the format.
     * \return the process found or dprocess() if no such process exists.
     */
    dprocess find_dprocess(const std::string& name) const {
      auto it = std::find_if(dprocesses.begin(), dprocesses.end(),
                             [&](dprocess p) { return p.name() == name; });
      return (it != dprocesses.end()) ? *it : dprocess();
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
    //========================================================================

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
      load_options(config);
      load_variables(config, u);
      load_dprocesses(config, u);

      // empty formats are not allowed
      if (variables.empty() && dprocesses.empty()) {
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
      save_variables(config);
      save_dprocesses(config);
      config.save(filename);
    }

  private:
    //! load the options
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

    //! load the variables
    void load_variables(simple_config& config, universe& u) {
      typedef std::pair<std::string, std::string> config_entry;
      for (const config_entry& entry : config["variables"]) {
        if (entry.second.compare(0, 11, "continuous(") == 0) {
          std::string param = entry.second.substr(11, entry.second.size() - 12);
          std::size_t dim;
          if (!parse_string(param, dim) || dim == 0) {
            std::string msg =
              "Invalid specification of continuous variable \"" + entry.first +
              "\": " + entry.second;
            throw std::invalid_argument(msg);
          }
          variables.push_back(u.new_continuous_variable(entry.first, dim));
        } else if (entry.second.compare(0, 9, "discrete(") == 0) {
          std::string param = entry.second.substr(9, entry.second.size() - 10);
          std::size_t values;
          if (!parse_string(param, values) || values <= 1) {
            std::string msg =
              "Invalid specification of discrete variable \"" + entry.first +
              "\": " + entry.second;
            throw std::invalid_argument(msg);
          }
          variables.push_back(u.new_discrete_variable(entry.first, values));
        } else { // discrete variable with named levels
          std::vector<std::string> levels;
          string_split(entry.second, ", ", levels);
          if (levels.size() <= 1) {
            std::string msg =
              "Invalid specification of discrete variable \"" + entry.first +
              "\": " + entry.second + " (must have more than 1 level)";
            throw std::invalid_argument(msg);
          }
          if (std::find(levels.begin(), levels.end(), missing) != levels.end()){
            std::string msg =
              "The missing value symbol \"" + missing + "\" must not be " +
              "a level of discrete variable \"" + entry.first + "\"";
            throw std::invalid_argument(msg);
          }
          variables.push_back(u.new_discrete_variable(entry.first, levels));
        }
      }
    }

    //! load the discrete processes
    void load_dprocesses(simple_config& config, universe& u) {
      typedef std::pair<std::string, std::string> config_entry;
      for (const config_entry& entry : config["discrete_processes"]) {
        if (entry.second.compare(0, 11, "continuous(") == 0) {
          std::string name = entry.first;
          std::string param = entry.second.substr(11, entry.second.size() - 12);
          std::size_t dim;
          if (!parse_string(param, dim) || dim == 0) {
            std::string msg =
              "Invalid specification of discrete-time continuous-value process "
              "\"" + name + "\": " + entry.second;
            throw std::invalid_argument(msg);
          }
          dprocesses.push_back(u.new_continuous_dprocess(name, dim));
        } else if (entry.second.compare(0, 9, "discrete(") == 0) {
          std::string name = entry.first;
          std::string param = entry.second.substr(9, entry.second.size() - 10);
          std::size_t values;
          if (!parse_string(param, values) || values <= 1) {
            std::string msg =
              "Invalid specification of discrete-time discrete-value process \""
              + name + "\": " + entry.second;
            throw std::invalid_argument(msg);
          }
          dprocesses.push_back(u.new_discrete_dprocess(name, values));
        } else { // discrete-value process with named levels
          std::string name = entry.first;
          std::vector<std::string> levels;
          string_split(entry.second, ", ", levels);
          if (levels.size() <= 1) {
            std::string msg =
              "Invalid specification of discrete-time discrete-value process \""
              + name + "\": " + entry.second + " (must have > 1 levels)";
            throw std::invalid_argument(msg);
          }
          if (std::find(levels.begin(), levels.end(), missing) != levels.end()){
            std::string msg =
              "The missing value symbol \"" + missing + "\" must not be " +
              "a level of discrete-time discrete-value process \"" +
              entry.first + "\"";
            throw std::invalid_argument(msg);
          }
          dprocesses.push_back(u.new_discrete_dprocess(name, levels));
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

    //! store the variables
    void save_variables(simple_config& config) const {
      for (variable v : variables) {
        if (v.is_continuous()) {
          std::string dim = to_string(v.num_dimensions());
          config.add("variables", v.name(), "continuous(" + dim + ")");
        } else if (v.is_discrete() && v.levels().empty()) {
          std::string values = to_string(v.num_values());
          config.add("variables", v.name(), "discrete(" + values + ")");
        } else if (v.is_discrete()) {
          std::string levels = string_join(",", v.levels());
          config.add("variables", v.name(), levels);
        } else {
          throw std::logic_error("Unsupported variable \"" + v.name() + '"');
        }
      }
    }

    //! store the discrete processes
    void save_dprocesses(simple_config& config) const {
      for (dprocess p : dprocesses) {
        if (p.is_continuous()) {
          std::string dim = to_string(p.num_dimensions());
          config.add("discrete_processes", p.name(), "continuous(" + dim + ")");
        } else if (p.is_discrete() && p.levels().empty()) {
          std::string vals = to_string(p.num_values());
          config.add("discrete_processes", p.name(), "discrete(" + vals + ")");
        } else if (p.is_discrete()) {
          std::string levels = string_join(",", p.levels());
          config.add("discrete_processes", p.name(), levels);
        } else {
          throw std::logic_error("Unsupported discrete-time process \"" +
                                 p.name() + '"');
        }
      }
    }

  }; // struct text_dataset_format

} // namespace libgm

#endif
