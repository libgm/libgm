#ifndef LIBGM_VAR_HPP
#define LIBGM_VAR_HPP

#include <libgm/argument/indexed.hpp>
#include <libgm/argument/traits.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/serialization/string.hpp>
#include <libgm/serialization/vector.hpp>
#include <libgm/parser/string_functions.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace libgm {

  /**
   * A class that represents a basic variable (an univariate argument).
   * A var object models the MixedArgument concept, i.e., it can represent
   * either a discrete or a continuous argument. Internally, a variable
   * is represented by a pointer to an object describing the variable type
   * and values.
   *
   * The typical way of creating variable is via the static functions discrete()
   * and continuous(), whose first argument is a universe that will maintain the
   * ownership of the variable, and the second argument is the variable name.
   * For example, in order to create a discrete variable named "x" with 5
   * values, and a continuous variable named "y", you write:
   *
   *   universe u;
   *   var x = var::discrete(u, "x", 5);
   *   var y = var::continuous(u, "y");
   *
   * The lifetime of the universe u must extend past all the objects that
   * were created using it.
   *
   * This class models the MixedArgument, UnivariateArgument, and
   * IndexableArgument concepts.
   */
  class var {
  public:
    // Forward declaration
    struct description;

    //! The argument arity (univariate).
    using argument_arity = univariate_tag;

    //! The argument category (mixed).
    using argument_category = mixed_tag;

    //! The objects of class var can act as processes, i.e., are indexable.
    static const bool is_indexable = true;

    //! An enum representing the category of the variable.
    enum category_enum { NONE = 0, DISCRETE = 1, CONTINUOUS = 2 };

    /**
     * Default constructor; constructs a null variable.
     */
    var() : desc_(nullptr) { }

    /**
     * Constructs a discrete variable with the given name and number of values.
     */
    static var discrete(universe& u,
                        const std::string& name,
                        std::size_t num_values) {
      return var(u.acquire(new description(DISCRETE, name, num_values)));
    }

    /**
     * Constructs a discrete variable with the given name and named values
     * (levels). The number of values of the variable is the number of levels.
     */
    static var discrete(universe& u,
                        const std::string& name,
                        const std::vector<std::string>& levels) {
      return var(u.acquire(new description(DISCRETE, name, levels)));
    }

    /**
     * Constructs a discrete variable with the given name and number of values.
     * This is a convenience function for the compatibility with vec;
     * the provided vector num_values must have exactly one element.
     *
     * \throw std::invalid_argument if num_values.size() != 1
     */
    static var discrete(universe& u,
                        const std::string& name,
                        const std::vector<std::size_t>& num_values) {
      if (num_values.size() != 1) {
        throw std::invalid_argument("Attempt to construct multivariate var.");
      }
      return var(u.acquire(new description(DISCRETE, name, num_values[0])));
    }

    /**
     * Constructs a continuous variable with the given name.
     * This function accepts the dimensionality as an optional argument for
     * compatibility with vec; the dimensionality must always be 1.
     */
    static var continuous(universe& u,
                          const std::string& name,
                          std::size_t num_dimensions = 1) {
      if (num_dimensions != 1) {
        throw std::invalid_argument("Attempt to construct multivariate var");
      }
      return var(u.acquire(new description(CONTINUOUS, name)));
    }

    /**
     * Returns a functor that generates discrete variables with specified
     * base name and an increasing index.
     */
    static std::function<var(std::size_t)>
    discrete_generator(universe& u, const std::string& basename) {
      return [&u, basename, index = std::size_t(0)](std::size_t num_values)
        mutable {
        return var::discrete(u, basename + std::to_string(index++), num_values);
      };
    }

    /**
     * Returns a functor that generates continuous variables with specified
     * base name and an increasing index.
     */
    static std::function<var()>
    continuous_generator(universe& u, const std::string& basename) {
      return [&u, basename, index = std::size_t(0)]() mutable {
        return var::continuous(u, basename + std::to_string(index++));
      };
    }

    //! Returns the variable descriptor.
    const description* desc() const {
      return desc_;
    }

    //! Saves the variable to an archive.
    void save(oarchive& ar) const {
      ar.serialize_dynamic(desc_);
    }

    //! Loads the variable from an archive.
    void load(iarchive& ar) {
      desc_ = ar.deserialize_dynamic<description>();
    }

    //! Prints a variable to an output stream.
    friend std::ostream& operator<<(std::ostream& out, var x) {
      if (x.desc_) {
        out << x.desc_;
      } else {
        out << "null";
      }
      return out;
    }

    // Comparisons and hashing
    //--------------------------------------------------------------------------

    //! Compares two variables.
    friend bool operator==(var x, var y) {
      return x.desc_ == y.desc_;
    }

    //! Compares two variables.
    friend bool operator!=(var x, var y) {
      return x.desc_ != y.desc_;
    }

    //! Compares two variables.
    friend bool operator<(var x, var y) {
      return x.desc_ < y.desc_;
    }

    //! Compares two variables.
    friend bool operator>(var x, var y) {
      return x.desc_ > y.desc_;
    }

    //! Compares two variables.
    friend bool operator<=(var x, var y) {
      return x.desc_ <= y.desc_
    }

    //! Compares two variables.
    friend bool operator>=(var x, var y) {
      return x.desc_ >= y.desc_;
    }

    // Argument properties
    //--------------------------------------------------------------------------

    //! Returns the arity of a variable (always 1).
    std::size_t arity() const {
      return 1;
    }

    //! Returns the number of values for a discrete variable.
    std::size_t size() const {
      assert(desc_);
      if (desc_->category == DISCRETE) {
        return desc_->cardinality;
      } else {
        throw std::invalid_argument(
          "Attempt to call num_values() on a variable that is not discrete"
        );
      }
    }

    //! Returns true if the variable is discrete.
    bool discrete() const {
      assert(desc_);
      return desc_->category == DISCRETE;
    }

    //! Returns true if the variable is continuous.
    bool continuous() const {
      assert(desc_);
      return desc_->category == CONTINUOUS;
    }

    //! Returns an instance of the variable process for a single index.
    template <typename Index>
    indexed<var, Index> operator()(Index index) const {
      return { *this, index };
    }

    // Variable description
    //--------------------------------------------------------------------------

    //! A struct describing a variable.
    struct description : universe::managed {

      //! The category of the variable.
      category_enum category;

      //! The name of the variable.
      std::string name;

      //! The number of values for a discrete variable.
      std::size_t cardinality;

      //! The levels / labels for a discrete variable (optional).
      std::vector<std::string> levels;

      //! Constructs a description for an empty variable.
      description()
        : category(NONE), cardinality(0) { }

      //! Constructs a description.
      description(category_enum category,
                  const std::string& name,
                  std::size_t cardinality = 0)
        : category(category),
          name(name),
          cardinality(cardinality) { }

      //! Constructs a description.
      description(category_enum category,
                  const std::string& name,
                  const std::vector<std::string>& levels)
        : category(category),
          name(name),
          cardinality(levels.size()),
          levels(levels) { }

      //! Saves the object to an archive.
      void save(oarchive& ar) const {
        ar << name << char(category);
        if (category == DISCRETE) {
          ar << cardinality << levels;
        }
      }

      //! Loads the object from an archive.
      void load(iarchive& ar) {
        char cat;
        ar >> name >> cat;
        category = category_enum(cat);
        if (category == DISCRETE) {
          ar >> cardinality >> levels;
        } else {
          cardinality = 0;
          levels.clear();
        }
      }

      //! Parses the value of a discrete argument from to a string.
      std::size_t parse_discrete(const char* str, std::size_t pos = 0) const {
        assert(category == DISCRETE);
        assert(pos == 0);
        if (levels.empty()) {
          std::size_t value = parse_string<std::size_t>(str);
          if (value >= cardinality) {
            std::ostringstream out;
            out << "Value out of bounds: " << value << ">=" << cardinality;
            throw std::invalid_argument(out.str());
          }
          return value;
        } else {
          auto it = std::find(levels.begin(), levels.end(), str);
          if (it == levels.end()) {
            std::ostringstream out;
            out << "Unknown value \"" << str << "\" for \"" << this << "\"";
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
      operator<<(std::ostream& out, const description* d) {
        switch (d->category) {
        case NONE:
          out << "N(" << d->name << ")"; break;
        case DISCRETE:
          out << "D(" << d->name << "|" << d->cardinality << ")"; break;
        case CONTINUOUS:
          out << "C(" << d->name << ")"; break;
        default:
          out << '?'; break;
        }
        return out;
      }

    }; // struct description

  private:
    //! Constructs a variable with the given descriptor and index.
    explicit var(const description* desc)
      : desc_(desc) { }

    //! The descriptor of the variable.
    const description* desc_;

  }; // class var

} // namespace libgm

#endif
