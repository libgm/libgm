#ifndef LIBGM_ARGUMENT_OBJECT_HPP
#define LIBGM_ARGUMENT_OBJECT_HPP

#include <libgm/serialization/serialize.hpp>
#include <libgm/serialization/string.hpp>

#include <iostream>
#include <string>

namespace libgm {

  /**
   * A struct implementing a basic process or variable.
   */
  struct argument_object {
  public:
    //! An enum representing the category of the variable.
    enum category_enum { FINITE = 1, VECTOR = 2 };

    //! The name of the argument.
    std::string name;

    //! The cardinality / dimensionality of the argument.
    std::size_t size;

    //! The category of the variable.
    category_enum category;

    //! Default constructor. Used only for deserializing.
    argument_object() { }

    //! Constructs an argument_object with given properties.
    argument_object(const std::string& name,
                    std::size_t size,
                    category_enum category)
      : name(name), size(size), category(category) { }

    //! Dynamically allocates a new finite argument_object.
    static argument_object* finite(const std::string& name, std::size_t size) {
      return new argument_object(name, size, FINITE);
    }

    //! Dynamically allocates a new vector argument_object.
    static argument_object* vector(const std::string& name, std::size_t size) {
      return new argument_object(name, size, VECTOR);
    }

    //! Saves the object to an archive.
    void save(oarchive& ar) const {
      ar << name << size << char(category);
    }

    //! Loads the object from an archive.
    void load(iarchive& ar) {
      char cat;
      ar >> name >> size >> cat;
      category = category_enum(cat);
    }

    //! Returns the string reprsentation of the argument object.
    std::string str() const {
      std::ostringstream out;
      out << *this;
      return out.str();
    }

    //! Prints the object to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const argument_object& a) {
      switch (a.category) {
      case argument_object::FINITE:
        out << 'F'; break;
      case argument_object::VECTOR:
        out << 'V'; break;
      default:
        out << '?'; break;
      }
      out << '(' << a.name << '|' << a.size << ')';
      return out;
    }

  }; // struct argument_object

} // namespace libgm

#endif
