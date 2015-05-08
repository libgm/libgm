#ifndef LIBGM_STRING_FUNCTIONS_HPP
#define LIBGM_STRING_FUNCTIONS_HPP

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace libgm {

  //! Removes leading and trailing white space from a string
  inline std::string trim(const std::string& str,
                          std::size_t start = 0,
                          std::size_t end = std::string::npos) {
    const char* whitespace = " \t\n\v\f\r";
    std::size_t pos1 = str.find_first_not_of(whitespace, start);
    if (pos1 == std::string::npos) {
      return std::string();
    } else {
      std::size_t pos2 = str.find_last_not_of(whitespace, end);
      assert(pos2 != std::string::npos);
      return str.substr(pos1, pos2 - pos1 + 1);
    }
  }

  //! Converts all letters in a string to lowercase in place.
  inline void tolower_inplace(std::string& s) {
    for (std::size_t i = 0; i < s.size(); ++i) {
      s[i] = std::tolower(s[i]);
    }
  }

  //! Converts all letters in a string to uppercase in place.
  inline void toupper_inplace(std::string& s) {
    for (std::size_t i = 0; i < s.size(); ++i) {
      s[i] = std::toupper(s[i]);
    }
  }

  //! Converts all letters in a string to lowercase.
  inline std::string tolower(std::string s) {
    tolower_inplace(s);
    return s;
  }

  //! Converts all letters in a string to uppercase.
  inline std::string toupper(std::string s) {
    toupper_inplace(s);
    return s;
  }

  //! Converts all instances of character a in string s to character b in place.
  inline void swap_characters_inplace(std::string& s, char a, char b) {
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (s[i] == a) { s[i] = b; }
    }
  }

  //! Convert all non-alphanumeric characters in string s to character c,
  //! in place.
  inline void nonalnum_to_char_inplace(std::string& s, char c) {
    for (std::size_t i(0); i < s.size(); ++i) {
      if (!isalnum(s[i])) { s[i] = c; }
    }
  }

  /**
   * The same as the std::getline, but this also increments a line
   * counter, which is useful for debugging purposes.
   */
  inline std::istream& getline(std::istream& in,
                               std::string& line,
                               std::size_t& line_number) {
    ++line_number;
    return std::getline(in, line);
  }

  /**
   * If the string is enclosed in parentheses, returns the string with
   * the C-style substitutions (\t => tab). Otherwise, returns the
   * input string.
   * \todo Make this more robust
   */
  inline std::string parse_escaped(const std::string& str) {
    if (str.empty() || str.front() != '"') {
      return str;
    }

    std::string result;
    for (std::size_t i = 1; i < str.size(); ++i) {
      switch (str[i]) {
      case '\\':
        ++i;
        if (i < str.size()) {
          result.push_back(str[i] == 't' ? '\t' : str[i]);
          break;
        } else {
          throw std::runtime_error("Unexpected end of the string: " + str);
        }
      case '"':
        if (i == str.size() - 1) {
          return result;
        } else {
          throw std::runtime_error(
            "Extra characters at the end of the string: " + str
          );
        }
      default:
        result.push_back(str[i]);
      }
    }

    throw std::runtime_error("Unexpcted end of the string: " + str);
  }

  /**
   * Escapes the string using C-style substitutions.
   * \todo Make this more robust
   */
  inline std::string escape_string(const std::string& str) {
    std::string result;
    result.push_back('"');
    for (char c : str) {
      switch (c) {
      case '\t': result.append("\\t"); break;
      case '"': result.append("\\\""); break;
      default: result.push_back(c); break;
      }
    }
    result.push_back('"');
    return result;
  }

  /**
   * Converts the argument to a string using string stream.
   */
  template <typename T>
  std::string to_string(const T& t) {
    std::ostringstream o;
    o << t;
    return o.str();
  }

  /**
   * Parses the string to an arbitrary type using string stream.
   * \return false if the parsing fails.
   */
  template <typename T>
  bool parse_string(const std::string& s, T& val) {
    std::istringstream in(s);
    in >> val;
    return bool(in);
  }

  /**
   * Parses the string to an arbitrary type using string stream.
   * \return false if the parsing fails
   */
  template <typename T>
  bool parse_string(const char* s, T& val) {
    std::istringstream in(s);
    in >> val;
    return bool(in);
  }

  /**
   * Parses a double using strtod. The string must not end with whitespace.
   * \return false if the parsing fails
   */
  inline bool parse_string(const char* s, double& val) {
    char* end;
    val = std::strtod(s, &end);
    return *s && *end == 0;
  }

  /**
   * Parses a float using strtod. The string must not end with whitespace.
   * \return false if the parsing fails
   * \todo fix this to use strtof once we move to C++11
   */
  inline bool parse_string(const char* s, float& val) {
    char* end;
    val = std::strtod(s, &end);
    return *s && *end == 0;
  }

  /**
   * Parses a long using strtol. The string must not end with whitespace.
   * \return false if the parsing fails
   */
  inline bool parse_string(const char* s, long& val) {
    char* end;
    val = std::strtol(s, &end, 10);
    return *s && *end == 0;
  }

  /**
   * Parses an unsigned long using strtoul. The string must not end with
   * whitespace.
   * \return false if the parsing fails
   */
  inline bool parse_string(const char* s, unsigned long& val) {
    char* end;
    val = std::strtoul(s, &end, 10);
    return *s && *end == 0;
  }

  /**
   * Parses a string as a value of type T.
   * \throw invalid_argument if the format is invalid
   */
  template <typename T>
  T parse_string(const std::string& s) {
    T val;
    if (!parse_string(s, val)) {
      throw std::invalid_argument("Could not parse the string \"" + s + "\"");
    }
    return val;
  }

  /**
   * Parses a string as a value of type T.
   * \throw invalid_argument if the format is invalid
   */
  template <typename T>
  T parse_string(const char* s) {
    T val;
    if (!parse_string(s, val)) {
      throw std::invalid_argument("Could not parse the string \"" +
                                  std::string(s) + "\"");
    }
    return val;
  }

  /**
   * Concatenate the given Range of values (using operator<< to print them),
   * with sep separating them, similar to Perl/Python's join.
   * \param  sep   String with which to separate the values.
   * \param  vals  Values to concatenate.
   */
  template <typename Range>
  std::string string_join(const std::string& sep, const Range& vals) {
    typename Range::const_iterator it  = vals.begin();
    typename Range::const_iterator end = vals.end();
    if (it == end) {
      return std::string();
    }

    std::ostringstream o;
    while (it != end) {
      o << *it;
      ++it;
      if (it != end) { o << sep; }
    }

    return o.str();
  }

  /**
   * Splits the string into tokens separated by the given separators.
   * \param str A string to be split into tokens. The strings is modified.
   * \param sep A string of delimiters.
   * \param tokens A vector of tokens. These are pointers to str.
   */
  inline void string_split(std::string& str, const std::string& sep,
                           std::vector<const char*>& tokens) {
    const char* begin = str.c_str();
    std::size_t pos1 = str.find_first_not_of(sep);
    while (pos1 != std::string::npos) {
      tokens.push_back(begin + pos1);
      std::size_t pos2 = str.find_first_of(sep, pos1);
      if (pos2 == std::string::npos) {
        pos1 = pos2;
      } else {
        str[pos2] = 0;
        pos1 = str.find_first_not_of(sep, pos2 + 1);
      }
    }
  }

  /**
   * Splits the string into tokens separated by the given separators.
   * \param str A string to be split.
   * \param sep A string of delimiters.
   * \param tokens A vector of tokens.
   */
  inline void string_split(const std::string& str, const std::string& sep,
                            std::vector<std::string>& tokens) {
    std::size_t pos1 = str.find_first_not_of(sep);
    while (pos1 != std::string::npos) {
      std::size_t pos2 = str.find_first_of(sep, pos1);
      if (pos2 == std::string::npos) {
        tokens.push_back(str.substr(pos1));
        pos1 = pos2;
      } else {
        tokens.push_back(str.substr(pos1, pos2 - pos1));
        pos1 = str.find_first_not_of(sep, pos2 + 1);
      }
    }
  }

  /**
   * Splits a file path into a directory and the file name.
   * E.g., "directory/file.txt" --> "directory" "file.txt"
   *       "file.txt"           --> "."         "file.txt"
   * @return <directory, filename>
   */
  inline std::pair<std::string, std::string>
  split_directory_file(const std::string& filepath) {
    std::size_t i = filepath.find_last_of('/');
    if (i == std::string::npos) {
      return std::make_pair(".", "filepath");
    } else {
      return std::make_pair(filepath.substr(0,i), filepath.substr(i+1));
    }
  }

} // namespace libgm

#endif
