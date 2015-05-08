#ifndef LIBGM_SIMPLE_CONFIG_HPP
#define LIBGM_SIMPLE_CONFIG_HPP

#include <libgm/parser/string_functions.hpp>

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace libgm {

  /**
   * A class for loading and saving configurations in a simple INI format.
   * \see http://en.wikipedia.org/wiki/INI_file
   */
  class simple_config {
  public:
    //! The entries for a single section, in the order specified in the file
    typedef std::vector< std::pair<std::string, std::string> > config_entries;

    //! A single entry in the configuration
    typedef std::pair<std::string, std::string> config_entry;

    //! Default constructor
    simple_config() { }

    //! Returns the entries for a given section
    config_entries& operator[](const std::string& section) {
      return sections_[section];
    }

    //! Adds an entry to a section
    void add(const std::string& section,
             const std::string& key,
             const std::string& value) {
      sections_[section].push_back(std::make_pair(key, value));
    }

    //! Adds an entry to a section by casting the value to a std::string
    template <typename T>
    void add(const std::string& section,
             const std::string& key,
             const T& value) {
      sections_[section].push_back(std::make_pair(key, to_string(value)));
    }

    //! Loads the configuration from a file
    void load(const std::string& filename) {
      std::ifstream in(filename);
      if (!in) {
        throw std::runtime_error("Could not open the file " + filename);
      }

      std::size_t line_number = 0;
      std::string line;
      std::string section;
      while (getline(in, line)) {
        ++line_number;

        // trim and get rid of comments
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) {
          continue;
        }

        // parse the section or the value
        if (line.front() == '[' && line.back() == ']') {
          section = line.substr(1, line.size() - 2);
        } else {
          std::size_t pos = line.find('=');
          if (pos == std::string::npos) {
            throw std::runtime_error(
              "Line " + to_string(line_number) + ": missing '='"
            );
          } else if (pos == 0) {
            throw std::runtime_error(
              "Line " + to_string(line_number) + ": missing key"
            );
          } else {
            add(section, trim(line, 0, pos - 1), trim(line, pos + 1));
          }
        }
      }
    }

    //! Saves the configuration to a file
    void save(const std::string& filename) const {
      std::ofstream out(filename);
      if (!out) {
        throw std::runtime_error("Could not open the file " + filename);
      }

      typedef std::pair<const std::string, config_entries> value_type;
      for (const value_type& section : sections_) {
        out << '[' << section.first << ']' << std::endl;
        for (const config_entry& entry : section.second) {
          out << entry.first << '=' << entry.second << std::endl;
        }
        out << std::endl;
      }
    }

  private:
    std::map<std::string, config_entries> sections_;

  }; // class simple_config

} // namespace libgm

#endif
