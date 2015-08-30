#ifndef LIBGM_VAR_HPP
#define LIBGM_VAR_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/serialization/string.hpp>
#include <libgm/serialization/vector.hpp>
#include <libgm/parser/string_functions.hpp>

#include <iostream>
#include <stdexcept>
#include <sstream>

namespace libgm {

  /**
   * A class that represents a basic variable (an univariate argument).
   * A var object models the MixedArgument concept, i.e., it can represent
   * either a discrete or a continuous argument. Internally, a variable
   * is represented by a descriptor (a pointer to an object describing the
   * variable type and values) and an optional time index. Presently,
   * the time index is discrete, but this restriction will be relaxed later.
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
   * IndexedArgument concepts.
   */
  class var {
  public:
    // Forward declaration
    struct description;

    //! The argument arity (univariate).
    typedef univariate_tag argument_arity;

    //! The argument category (mixed).
    typedef mixed_tag argument_category;

    //! The descriptor of the variable and the fields based on it.
    typedef const description* descriptor;

    //! The index associated with a variable.
    typedef std::size_t index_type;

    //! The instance of a variable (void because variables are not indexable).
    typedef void instance_type;

    //! An enum representing the category of the variable.
    enum category_enum { NONE = 0, DISCRETE = 1, CONTINUOUS = 2 };

    /**
     * Constructs a variable with the given descriptor and index.
     */
    explicit var(const description* desc = nullptr, std::size_t index = -1)
      : desc_(desc), index_(index) { }

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

    //! Converts the variable to a pair of the descriptor and index.
    std::pair<descriptor, std::size_t> pair() const {
      return { desc_, index_ };
    }

    //! Saves the variable to an archive.
    void save(oarchive& ar) const {
      ar.serialize_dynamic(desc_);
      if (desc_) { ar << index_; }
    }

    //! Loads the variable from an archive.
    void load(iarchive& ar) {
      desc_ = ar.deserialize_dynamic<description>();
      if (desc_) { ar >> index_; } else { index_ = -1; }
    }

    //! Prints a variable to an output stream.
    friend std::ostream& operator<<(std::ostream& out, var x) {
      if (x.desc_) {
        out << x.desc_;
        if (x.indexed()) { out << '(' << x.index_ << ')'; }
      } else {
        out << "null";
      }
      return out;
    }

    // Comparisons and hashing
    //==========================================================================

    //! Compares two variables.
    friend bool operator==(var x, var y) {
      return x.desc_ == y.desc_ && x.index_ == y.index_;
    }

    //! Compares two variables.
    friend bool operator!=(var x, var y) {
      return x.desc_ != y.desc_ || x.index_ != y.index_;
    }

    //! Compares two variables.
    friend bool operator<(var x, var y) {
      return x.pair() < y.pair();
    }

    //! Compares two variables.
    friend bool operator>(var x, var y) {
      return x.pair() > y.pair();
    }

    //! Compares two variables.
    friend bool operator<=(var x, var y) {
      return x.pair() <= y.pair();
    }

    //! Compares two variables.
    friend bool operator>=(var x, var y) {
      return x.pair() >= y.pair();
    }

    //! Computes the hash of the variable.
    friend std::size_t hash_value(var x) {
      std::size_t seed = 0;
      libgm::hash_combine(seed, x.desc_);
      libgm::hash_combine(seed, x.index_);
      return seed;
    }

    // Implementation of argument traits
    //==========================================================================

    //! Returns true if two variables are compatible.
    static bool compatible(var x, var y) {
      assert(x.desc_ && y.desc_);
      return x.desc_->category == y.desc_->category
        && x.desc_->cardinality == y.desc_->cardinality;
    }

    //! Returns the number of dimensions of a variable (always 1).
    std::size_t num_dimensions() const {
      return 1;
    }

    //! Returns the number of values for a discrete variable.
    std::size_t num_values() const {
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

    //! Returns true if the variable is associated with a process.
    bool indexed() const {
      return index_ != std::size_t(-1);
    }

    //! Returns the variable descriptor.
    const description* desc() const {
      return desc_;
    }

    //! Returns the index of the variable.
    std::size_t index() const {
      return index_;
    }

    // Variable description
    //==========================================================================

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
    //! The descriptor of the variable.
    const description* desc_;

    //! The index associated with the variable.
    std::size_t index_;

  }; // class var

  // Traits
  //============================================================================

  /**
   * A specialization of vertex_traits for variable.
   */
  template <>
  struct vertex_traits<var> {
    //! Returns the default-constructed variable.
    static var null() { return var(); }

    //! Returns a special "deleted" variable.
    static var deleted() { static var::description desc; return var(&desc); }

    //! Prints the variable to an output stream.
    static void print(std::ostream& out, var v) { out << v; }

    //! Variables use the default hasher.
    typedef std::hash<var> hasher;
  };

} // namespace libgm


namespace std {

  template <>
  struct hash<libgm::var>
    : libgm::default_hash<libgm::var> { };

} // namespace std

#endif
