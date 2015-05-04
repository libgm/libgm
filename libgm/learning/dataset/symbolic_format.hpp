#ifndef LIBGM_SYMBOLIC_FORMAT_HPP
#define LIBGM_SYMBOLIC_FORMAT_HPP

#include <libgm/global.hpp>
#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/argument/process.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/parser/simple_config.hpp>
#include <libgm/parser/string_functions.hpp>

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace libgm {

  /**
   * A utility class for representing the format and the columns of datasets
   * such as those in the UCI Machine Learning Repository. The user needs to
   * specify the attributes, values, and formatting parameters for the dataset;
   * these can be loaded from configuration files using load_config().
   * The datasets can be then loaded/saved using the load()/save() free functions
   * provided in *dataset_io.hpp headers.
   *
   * \see finite_memory_dataset, vector_memory_dataset, hybrid_memory_dataset,
   *      sequence_memory_dataset
   */
  struct symbolic_format {

    //! A class that holds formatting information for a single variable
    class variable_info {
      variable v;
      std::vector<std::string> labels;

    public:
      //! Default constructor with empty variable.
      variable_info() { }

      //! Constructor for a generic variable
      explicit variable_info(variable v)
        : v(v) { }

      //! Constructor for a finite variable with named values.
      variable_info(variable v, const std::vector<std::string>& values)
        : v(v), labels(values) {
        assert(v.finite());
        assert(v.size() == labels.size());
      }

      //! Returns the arity / dimensionality of the variable
      size_t size() const {
        return v.size();
      }

      //! Returns the name of the variable
      const std::string& name() const {
        return v.name();
      }

      //! Returns the values vector
      const std::vector<std::string>& values() const {
        return labels;
      }

      //! Returns true if the variable is finite
      bool is_finite() const {
        return v.finite();
      }

      //! Returns true if the variable is finite and uses unnamed values.
      bool is_plain_finite() const {
        return is_finite() && labels.empty();
      }

      //! Returns true if the variable is finite and uses named values.
      bool is_named_finite() const {
        return is_finite() && !labels.empty();
      }

      //! Returns true if the variable is vector
      bool is_vector() const {
        return v.vector();
      }

      //! Returns the underlying variable
      variable var() const {
        return v;
      }

      //! Returns the finite value corresponding to the given string.
      //! Requires that the variable is finite.
      size_t parse(const char* str) const {
        assert(is_finite());
        if (labels.empty()) {
          return parse_string<size_t>(str);
        }
        std::vector<std::string>::const_iterator it =
          std::find(labels.begin(), labels.end(), str);
        if (it == labels.end()) {
          std::ostringstream os;
          os << "Unknown value \"" << str
             << "\" for variable \"" << v.name()
             << "\"";
          throw std::invalid_argument(os.str());
        }
        return it - labels.begin();
      }

      //! Prints the finite value to a stream
      void print(std::ostream& out, size_t value) const {
        assert(is_finite());
        if (labels.empty()) {
          out << value;
        } else {
          assert(value < labels.size());
          out << labels[value];
        }
      }
    }; // class variable_info

    //! A class that holds formatting information for a single discrete process
    class discrete_process_info {
      dprocess p;
      std::vector<std::string> labels;

    public:
      //! Default constructor with an empty process
      discrete_process_info() { }

      //! Constructor for a plain finite discrete process
      explicit discrete_process_info(dprocess p)
        : p(p) { }

      //! Constructor for a finite discrete process with named values.
      discrete_process_info(dprocess p,
                            const std::vector<std::string>& values)
        : p(p), labels(values) {
        assert(p.finite());
        assert(p.size() == labels.size());
      }

      //! Returns the arity / dimensionality of the process
      size_t size() const {
        return p.size();
      }

      //! Returns the name of the process
      const std::string& name() const {
        return p.name();
      }

      //! Returns the values vector
      const std::vector<std::string>& values() const {
        return labels;
      }

      //! Returns true if the process is finite
      bool is_finite() const {
        return p.finite();
      }

      //! Returns true if the process is finite and uses unnamed values.
      bool is_plain_finite() const {
        return p.finite() && labels.empty();
      }

      //! Returns true if the process is finite and uses named values.
      bool is_named_finite() const {
        return p.finite() && !labels.empty();
      }

      //! Returns true if the process is vector
      bool is_vector() const {
        return p.vector();
      }

      //! Returns the underlying process
      dprocess proc() const {
        return p;
      }

      //! Returns the finite value corresponding to the given string.
      //! Requires that the process is finite.
      size_t parse(const char* str) const {
        assert(is_finite());
        if (labels.empty()) {
          return parse_string<size_t>(str);
        }
        std::vector<std::string>::const_iterator it =
          std::find(labels.begin(), labels.end(), str);
        if (it == labels.end()) {
          std::ostringstream os;
          os << "Unknown value \"" << str
             << "\" for process \"" << p.name()
             << "\"";
          throw std::invalid_argument(os.str());
        }
        return it - labels.begin();
      }

      //! Prints the finite value to a stream
      void print(std::ostream& out, size_t value) const {
        assert(is_finite());
        if (labels.empty()) {
          out << value;
        } else {
          assert(value < labels.size());
          out << labels[value];
        }
      }
    }; // class discrete_process_info

    //! Specifies the separator for the fields (default whitespace).
    std::string separator;

    //! Specifies the symbol for missing values.
    std::string missing;

    //! The number of lines at the beginning of the file to skip (default = 0).
    size_t skip_rows;

    //! The number of columns at the beginning of each line to skip (default = 0).
    size_t skip_cols;

    //! Indicates if the dataset is weighted (default = false).
    bool weighted;

    //! The information about variables in the dataset.
    std::vector<variable_info> var_infos;

    //! The information about discrete processes in the dataset.
    std::vector<discrete_process_info> discrete_infos;

    /**
     * Constructs the symbolic format with default parameters.
     */
    symbolic_format()
      : skip_rows(0), skip_cols(0), weighted(false) { }

    // Functions related to variables
    //========================================================================

    /**
     * Returns true if all the variables in the format are finite.
     */
    bool is_finite() const {
      for (const variable_info& info : var_infos) {
        if (!info.is_finite()) {
          return false;
        }
      }
      return true;
    }

    /**
     * Returns the domain of all finite variables in the format.
     */
    domain finite_vars() const {
      domain result;
      for (const variable_info& info : var_infos) {
        if (info.is_finite()) {
          result.push_back(info.var());
        }
      }
      return result;
    }

    /**
     * Returns a finite variable with the given name.
     * \throw std::out_of_range if the variable with the name is not present
     * \throw std::domain_error if the variable is present but is not finite
     */
    variable finite_var(const std::string& name) const {
      for (const variable_info& info : var_infos) {
        if (info.name() == name) {
          if (info.is_finite()) {
            return info.var();
          } else {
            throw std::domain_error("Variable \"" + name + "\" if not finite");
          }
        }
      }
      throw std::out_of_range("Could not find the variable \"" + name + "\"");
    }

    /**
     * Returns true if all the variables in the format are vector.
     */
    bool is_vector() const {
      for (const variable_info& info : var_infos) {
        if (!info.is_vector()) {
          return false;
        }
      }
      return true;
    }

    /**
     * Returns the domain of all vector variables in the format.
     */
    domain vector_vars() const {
      domain result;
      for (const variable_info& info : var_infos) {
        if (info.is_vector()) {
          result.push_back(info.var());
        }
      }
      return result;
    }

    /**
     * Returns a vector variable with the given name.
     * \throw std::out_of_range if the variable with the name is not present
     * \throw std::domain_error if the variable is present but is not vector
     */
    variable vector_var(const std::string& name) const {
      for (const variable_info& info : var_infos) {
        if (info.name() == name) {
          if (info.is_vector()) {
            return info.var();
          } else {
            throw std::domain_error("Variable \"" + name + "\" if not a vector");
          }
        }
      }
      throw std::out_of_range("Could not find the variable \"" + name + "\"");
    }

    /**
     * Returns the domain of variables in this format.
     */
    hybrid_domain<> vars() const {
      hybrid_domain<> result;
      for (const variable_info& info : var_infos) {
        if (info.is_finite()) {
          result.finite().push_back(info.var());
        } else {
          result.vector().push_back(info.var());
        }
      }
      return result;
    }

    /**
     * Returns a variable with the given name.
     * \throw std::out_of_range if the variable with the name is not present
     */
    variable var(const std::string& name) const {
      for (const variable_info& info : var_infos) {
        if (info.name() == name) {
          return info.var();
        }
      }
      throw std::out_of_range("Could not find the variable \"" + name + "\"");
    }

    // Functions related to discrete processes
    //========================================================================

    /**
     * Returns true if all the discrete processes in the format are finite.
     */
    bool is_finite_discrete() const {
      for (const discrete_process_info& info : discrete_infos) {
        if (!info.is_finite()) {
          return false;
        }
      }
      return true;
    }

    /**
     * Returns the domain of all finite discrete processes in the format.
     */
    dprocess_domain finite_discrete_procs() const {
      dprocess_domain result;
      for (const discrete_process_info& info : discrete_infos) {
        if (info.is_finite()) {
          result.push_back(info.proc());
        }
      }
      return result;
    }

    /**
     * Returns a finite discrete process with the given name.
     * \throw std::out_of_range if the process with the name is not present
     * \throw std::domain_error if the process is present but is not finite
     */
    dprocess finite_discrete_proc(const std::string& name) const {
      for (const discrete_process_info& info : discrete_infos) {
        if (info.name() == name) {
          if (info.is_finite()) {
            return info.proc();
          } else {
            throw std::domain_error("Process \"" + name + "\" if not finite");
          }
        }
      }
      throw std::out_of_range("Could not find the process \"" + name + "\"");
    }

    /**
     * Returns true if all the discrete processs in the format are vector.
     */
    bool is_vector_discrete() const {
      for (const discrete_process_info& info : discrete_infos) {
        if (!info.is_vector()) {
          return false;
        }
      }
      return true;
    }

    /**
     * Returns the domain of all vector discrete processes in the format.
     */
    dprocess_domain vector_discrete_procs() const {
      dprocess_domain result;
      for (const discrete_process_info& info : discrete_infos) {
        if (info.is_vector()) {
          result.push_back(info.proc());
        }
      }
      return result;
    }

    /**
     * Returns a vector variable with the given name.
     * \throw std::out_of_range if the variable with the name is not present
     * \throw std::domain_error if the variable is present but is not vector
     */
    dprocess vector_discrete_proc(const std::string& name) const {
      for (const discrete_process_info& info : discrete_infos) {
        if (info.name() == name) {
          if (info.is_vector()) {
            return info.proc();
          } else {
            throw std::domain_error("Process \"" + name + "\" if not a vector");
          }
        }
      }
      throw std::out_of_range("Could not find the process \"" + name + "\"");
    }

    /**
     * Returns the domain of discrete processes in this format.
     */
    hybrid_domain<dprocess> discrete_procs() const {
      hybrid_domain<dprocess> result;
      for (const discrete_process_info& info : discrete_infos) {
        result.push_back(info.proc());
      }
      return result;
    }

    /**
     * Returns a process with the given name.
     * \throw std::out_of_range if the variable with the name is not present
     */
    dprocess discrete_proc(const std::string& name) const {
      for (const discrete_process_info& info : discrete_infos) {
        if (info.name() == name) {
          return info.proc();
        }
      }
      throw std::out_of_range("Could not find the process \"" + name + "\"");
    }

    // Functions related to parsing
    //========================================================================

    /**
     * Parses a line according to the format and stores the result in an array
     * of C strings.
     * \param line input (modified)
     * \param current line number (updated)
     * \param tokens output tokens
     * \throw runtime_error if the number of columns in the input data does not
     *        match the format.
     * \return false if the line should be ignored
     */
    bool parse(size_t num_values,
               std::string& line,
               size_t& line_number,
               std::vector<const char*>& tokens) const {
      if (++line_number <= skip_rows) {
        return false;
      }
      string_split(line, separator.empty() ? "\t " : separator, tokens);
      if (tokens.empty()) {
        return false;
      }
      size_t expected_cols = skip_cols + num_values + weighted;
      if (tokens.size() != expected_cols) {
        std::ostringstream os;
        os << "Line " << line_number << ": invalid number of columns "
           << "(expected " << expected_cols << ", found " << tokens.size() << ")";
        throw std::runtime_error(os.str());
      }
      return true;
    }

    // Functions for format I/O
    //========================================================================

    /**
     * Loads the symbolic_format from a configuration file with the following
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
    void load(const std::string& filename, universe& u) {
      simple_config config;
      config.load(filename);

      // load the components of a config
      load_options(config);
      load_variables(config, u);
      load_discrete(config, u);

      // empty formats are not allowed
      if (var_infos.empty() && discrete_infos.empty()) {
        throw std::out_of_range("Please specify at least one variable or process");
      }
    }

    /**
     * Saves the symbolic_format to a configuration file with the format
     * given in load().
     */
    void save(const std::string& filename) const {
      simple_config config;
      save_variables(config);
      save_discrete(config);
      save_options(config);
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
          skip_rows = parse_string<size_t>(entry.second);
        } else if (entry.first == "skip_cols") {
          skip_cols = parse_string<size_t>(entry.second);
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
        if (entry.second.compare(0, 7, "vector(") == 0) {
          std::string param = entry.second.substr(7, entry.second.size() - 8);
          size_t dim;
          if (!parse_string(param, dim) || dim == 0) {
            std::string msg = 
              "Invalid specification of vector variable \"" + entry.first +
              "\": " + entry.second;
            throw std::invalid_argument(msg);
          }
          var_infos.emplace_back(u.new_vector_variable(entry.first, dim));
        } else if (entry.second.compare(0, 7, "finite(") == 0) {
          std::string param = entry.second.substr(7, entry.second.size() - 8);
          size_t arity;
          if (!parse_string(param, arity) || arity <= 1) {
            std::string msg = 
              "Invalid specification of finite variable \"" + entry.first + 
              "\": " + entry.second;
            throw std::invalid_argument(msg);
          }
          var_infos.emplace_back(u.new_finite_variable(entry.first, arity));
        } else { // finite variable with named values
          std::vector<std::string> values;
          string_split(entry.second, ", ", values);
          if (values.size() <= 1) {
            std::string msg =
              "Invalid specification of finite variable \"" + entry.first +
              "\": " + entry.second + " (must have arity > 1)";
            throw std::invalid_argument(msg);
          }
          if (std::find(values.begin(), values.end(), missing) != values.end()) {
            std::string msg =
              "The missing value symbol \"" + missing + "\" must not be " +
              "a value of finite variable \"" + entry.first + "\"";
            throw std::invalid_argument(msg);
          }
          variable v = u.new_finite_variable(entry.first, values.size());
          var_infos.emplace_back(v, values);
        }
      }
    }

    //! load the discrete processes
    void load_discrete(simple_config& config, universe& u) {
      typedef std::pair<std::string, std::string> config_entry;
      for (const config_entry& entry : config["discrete_processes"]) {
        if (entry.second.compare(0, 7, "vector(") == 0) {
          std::string name = entry.first;
          std::string param = entry.second.substr(7, entry.second.size() - 8);
          size_t dim;
          if (!parse_string(param, dim) || dim == 0) {
            std::string msg = 
              "Invalid specification of vector discrete process \"" + name +
              "\": " + entry.second;
            throw std::invalid_argument(msg);
          }
          discrete_infos.emplace_back(u.new_vector_dprocess(name, dim));
        } else if (entry.second.compare(0, 7, "finite(") == 0) {
          std::string name = entry.first;
          std::string param = entry.second.substr(7, entry.second.size() - 8);
          size_t arity;
          if (!parse_string(param, arity) || arity <= 1) {
            std::string msg = 
              "Invalid specification of finite discrete process \"" + name + 
              "\": " + entry.second;
            throw std::invalid_argument(msg);
          }
          discrete_infos.emplace_back(u.new_finite_dprocess(name, arity));
        } else { // finite discrete process with named values
          std::string name = entry.first;
          std::vector<std::string> values;
          string_split(entry.second, ", ", values);
          if (values.size() <= 1) {
            std::string msg =
              "Invalid specification of finite discrete process \"" + name +
              "\": " + entry.second + " (must have arity > 1)";
            throw std::invalid_argument(msg);
          }
          if (std::find(values.begin(), values.end(), missing) != values.end()) {
            std::string msg =
              "The missing value symbol \"" + missing + "\" must not be " +
              "a value of finite discrete proces \"" + entry.first + "\"";
            throw std::invalid_argument(msg);
          }
          dprocess p = u.new_finite_dprocess(name, values.size());
          discrete_infos.emplace_back(p, values);
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
      for (const variable_info& info : var_infos) {
        if (info.is_vector()) {
          std::string dim = to_string(info.size());
          config.add("variables", info.name(), "vector(" + dim + ")");
        } else if (info.is_plain_finite()) {
          std::string arity = to_string(info.size());
          config.add("variables", info.name(), "finite(" + arity + ")");
        } else if (info.is_named_finite()) {
          std::string values = string_join(",", info.values());
          config.add("variables", info.name(), values);
        } else {
          throw std::logic_error("Unsupported variable type " + info.name());
        }
      }
    }

    //! store the discrete processes
    void save_discrete(simple_config& config) const {
      for (const discrete_process_info& info : discrete_infos) {
        if (info.is_vector()) {
          std::string dim = to_string(info.size());
          config.add("discrete_processes", info.name(), "vector(" + dim + ")");
        } else if (info.is_plain_finite()) {
          std::string arity = to_string(info.size());
          config.add("discrete_processes", info.name(), "finite(" + arity + ")");
        } else if (info.is_named_finite()) {
          std::string values = string_join(",", info.values());
          config.add("discrete_processes", info.name(), values);
        } else {
          throw std::logic_error("Unsupported discrete process type " + info.name());
        }
      }
    }

  }; // struct symbolic_format

} // namespace libgm

#endif
