#ifndef LIBGM_ARGUMENT_OBJECT_HPP
#define LIBGM_ARGUMENT_OBJECT_HPP

#include <libgm/parser/string_functions.hpp>
#include <libgm/serialization/string.hpp>
#include <libgm/serialization/vector.hpp>

#include <iostream>
#include <sstream>

namespace libgm {

  /**
   * A struct implementing a basic process or variable.
   */
  struct argument_object {
  public:
    //! An enum representing the category of the variable.
    enum category_enum { NONE = 0, DISCRETE = 1, CONTINUOUS = 2 };

    //! The name of the argument.
    std::string name;

    //! The cardinality / dimensionality of the argument.
    std::size_t size;

    //! The category of the variable.
    category_enum category;

    //! The levels / labels for a discrete argument (optional).
    std::vector<std::string> levels;

    //! Default constructor. Used only for deserializing.
    argument_object() { }

    //! Constructs an argument_object with given properties.
    argument_object(const std::string& name,
                    std::size_t size,
                    category_enum category)
      : name(name), size(size), category(category) { }

    //! Constructs an argumnet_object for a discrete argument with given levels.
    argument_object(const std::string& name,
                    const std::vector<std::string>& levels)
      : name(name), size(levels.size()), category(DISCRETE), levels(levels) { }

    //! Returns a singleton representing a deleted argument_object.
    static argument_object* deleted() {
      static argument_object obj("deleted", 0, NONE);
      return &obj;
    }

    //! Dynamically allocates a new finite argument_object.
    static argument_object*
    discrete(const std::string& name, std::size_t size) {
      return new argument_object(name, size, DISCRETE);
    }

    //! Dynamically allocates a new finite argument_object.
    static argument_object*
    discrete(const std::string& name, const std::vector<std::string>& levels) {
      return new argument_object(name, levels);
    }

    //! Dynamically allocates a new vector argument_object.
    static argument_object*
    continuous(const std::string& name, std::size_t size) {
      return new argument_object(name, size, CONTINUOUS);
    }

    //! Saves the object to an archive.
    void save(oarchive& ar) const {
      ar << name << size << char(category) << levels;
    }

    //! Loads the object from an archive.
    void load(iarchive& ar) {
      char cat;
      ar >> name >> size >> cat >> levels;
      category = category_enum(cat);
    }

    //! Parses the value of a discrete argument from to a string.
    std::size_t parse_discrete(const char* str) const {
      assert(category == DISCRETE);
      if (levels.empty()) {
        return parse_string<std::size_t>(str);
      } else {
        auto it = std::find(levels.begin(), levels.end(), str);
        if (it == levels.end()) {
          std::ostringstream out;
          out << "Unknown value \"" << str << "\" for \"" << *this << "\"";
          throw std::invalid_argument(out.str());
        } else {
          return it - levels.begin();
        }
      }
    }

    //! Prints the value of a discrete argument using the stored levels if any.
    void print_discrete(std::ostream& out, std::size_t value) const {
      assert(category == DISCRETE);
      if (levels.empty()) {
        out << value;
      } else {
        assert(value < levels.size());
        out << levels[value];
      }
    }

    //! Prints the object to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const argument_object& a) {
      switch (a.category) {
      case argument_object::NONE:
        out << 'N'; break;
      case argument_object::DISCRETE:
        out << 'D'; break;
      case argument_object::CONTINUOUS:
        out << 'C'; break;
      default:
        out << '?'; break;
      }
      out << '(' << a.name << '|' << a.size << ')';
      return out;
    }

  }; // struct argument_object

} // namespace libgm

#endif
