#ifndef LIBGM_VEC_HPP
#define LIBGM_VEC_HPP

#include <libgm/argument/indexed.hpp>
#include <libgm/argument/traits.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/serialization/string.hpp>
#include <libgm/serialization/vector.hpp>
#include <libgm/parser/string_functions.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

namespace libgm {

  /**
   * A class that represents a basic vector (a multivariate argument).
   * A vec object models the MixedArgument concept, i.e., it can represent
   * either a discrete or a continuous argument. Internally, a vector
   * is represented by a pointer to an object describing the vector type and
   * values.
   *
   * The typical way of creating vectors is via the static functions discrete()
   * and continuous(), whose first argument is a universe that will maintain the
   * ownership of the vector, and the second argument is the variable name.
   * For example, in order to create a discrete vector named "x" consisting of
   * two variables with 3 and 5 values, respectively, and a continuous vector
   * named "y" with 4 dimensions, you write:
   *
   *   universe u;
   *   vec x = vec::discrete(u, "x", {3, 5});
   *   vec y = vec::continuous(u, "y", 4);
   *
   * The lifetime of the universe u must extend past all the objects that
   * were created using it.
   *
   * This class models the MixedArgument, MultivariateArgumnet, and
   * IndexableArgument concepts.
   */
  class vec {
  public:
    // Forward declaration
    struct description;

    //! The argument arity (multivariate).
    using argument_arity = multivariate_tag;

    //! The argument category (mixed).
    using argument_category = mixed_tag;

    //! The objects of class vec can act as processes, i.e., are indexable.
    static const bool is_indexable = true;

    //! An enum representing the category of the vector.
    enum category_enum { NONE = 0, DISCRETE = 1, CONTINUOUS = 2 };

    /**
     * Default constructor; constructs a null vector.
     */
    vec() : desc_(nullptr) { }

    /**
     * Constructs a discrete vector with the given name and a single dimension
     * with the specified number of values.
     */
    static vec discrete(universe& u,
                        const std::string& name,
                        std::size_t num_values) {
      std::vector<std::size_t> num_values_vec = { num_values };
      return vec(u.acquire(new description(name, num_values_vec)));
    }

    /**
     * Constructs a discrete vector with the given name and a single dimension
     * with the specified named values (levels).
     */
    static vec discrete(universe& u,
                        const std::string& name,
                        const std::vector<std::string>& levels) {
      return vec(u.acquire(new description(name, levels)));
    }

    /**
     * Constructs a discrete vector with teh given name and numbers of values.
     * The specified vector must not be empty.
     */
    static vec discrete(universe& u,
                        const std::string& name,
                        const std::vector<std::size_t>& num_values) {
      if (num_values.empty()) {
        throw std::invalid_argument("Attempt to construct an empty vector");
      }
      return vec(u.acquire(new description(name, num_values)));
    }

    /**
     * Constructs a continuous vector with teh given name and dimensionality.
     * The number of dimensions must be > 0.
     */
    static vec continuous(universe& u,
                          const std::string& name,
                          std::size_t num_dimensions = 1) {
      if (num_dimensions == 0) {
        throw std::invalid_argument("Attempt to construct an empty vector");
      }
      return vec(u.acquire(new description(name, num_dimensions)));
    }

    //! Returns the descriptor of the vector.
    const description* desc() const {
      return desc_;
    }

    //! Saves the vector to an archive.
    void save(oarchive& ar) const {
      ar.serialize_dynamic(desc_);
    }

    //! Loads the vector from an archive.
    void load(iarchive& ar) {
      desc_ = ar.deserialize_dynamic<description>();
    }

    //! Prints a vector to an output stream.
    friend std::ostream& operator<<(std::ostream& out, vec x) {
      if (x.desc_) {
        out << x.desc_;
      } else {
        out << "null";
      }
      return out;
    }

    // Comparisons and hashing
    //--------------------------------------------------------------------------

    //! Compares two vectors.
    friend bool operator==(vec x, vec y) {
      return x.desc_ == y.desc_;
    }

    //! Compares two vectors.
    friend bool operator!=(vec x, vec y) {
      return x.desc_ != y.desc_;
    }

    //! Compares two vectors.
    friend bool operator<(vec x, vec y) {
      return x.desc_ < y.desc_;
    }

    //! Compares two vectors.
    friend bool operator>(vec x, vec y) {
      return x.desc_ > y.desc_;
    }

    //! Compares two vectors.
    friend bool operator<=(vec x, vec y) {
      return x.desc_ <= y.desc_;
    }

    //! Compares two vectors.
    friend bool operator>=(vec x, vec y) {
      return x.desc_ >= y.desc_;
    }

    // Argument properties
    //--------------------------------------------------------------------------

    //! Returns the number of dimensions of a vector.
    std::size_t arity() const {
      assert(desc_);
      return desc_->length;
    }

    //! Returns the total number of values for a discrete vector.
    std::size_t size() const {
      assert(desc_);
      if (desc_->category == DISCRETE) {
        return std::accumulate(desc_->cardinality.begin(),
                               desc_->cardinality.end(),
                               std::size_t(1),
                               std::multiplies<std::size_t>());
      } else {
        throw std::invalid_argument(
          "Attempt to call num_values() on a vector that is not discrete"
        );
      }
    }

    //! Returns the number of values for the given position of discrete vector.
    std::size_t size(std::size_t pos) const {
      assert(desc_);
      if (desc_->category == DISCRETE) {
        assert(pos < desc_->length);
        return desc_->cardinality[pos];
      } else {
        throw std::invalid_argument(
          "Attempt to call num_values() on a vector that is not discrete"
        );
      }
    }

    //! Returns true if the vector is discrete.
    bool discrete() const {
      assert(desc_);
      return desc_->category == DISCRETE;
    }

    //! Returns true if the vector is continuous.
    bool continuous() const {
      assert(desc_);
      return desc_->category == CONTINUOUS;
    }

    //! Returns an instance of a vector process for a single index.
    template <typename Index>
    indexed<vec, Index> operator()(Index index) const {
      return { *this, index };
    }

    // Vector description
    //--------------------------------------------------------------------------

    //! A struct describing a vector.
    struct description : universe::managed {

      //! The category of the vector.
      category_enum category;

      //! The name of the vector.
      std::string name;

      //! The length of the vector.
      std::size_t length;

      //! The number of values for each position of a discrete vector.
      std::vector<std::size_t> cardinality;

      //! The levels / labels for a discrete vector of size 1 (optional).
      std::vector<std::string> levels;

      //! Constructs an empty vector.
      description()
        : category(NONE), length(0) { }

      //! Constructs a description for a discrete vector.
      description(const std::string& name,
                  const std::vector<std::size_t>& cardinality)
        : category(DISCRETE),
          name(name),
          length(cardinality.size()),
          cardinality(cardinality) { }

      //! Constructs a description of a discrete vector.
      description(const std::string& name,
                  const std::vector<std::string>& levels)
        : category(DISCRETE),
          name(name),
          length(1),
          cardinality(1, levels.size()),
          levels(levels) { }

      //! Constructs a description of a continuous vector.
      description(const std::string& name,
                  std::size_t length)
        : category(CONTINUOUS),
          name(name),
          length(length) { }

      //! Saves the object to an archive.
      void save(oarchive& ar) const {
        ar << name << char(category);
        switch (category) {
        case NONE:
          break;
        case DISCRETE:
          ar << cardinality << levels;
          break;
        case CONTINUOUS:
          ar << length;
          break;
        }
      }

      //! Loads the object from an archive.
      void load(iarchive& ar) {
        char cat;
        ar >> name >> cat;
        category = category_enum(cat);
        cardinality.clear();
        levels.clear();
        switch (category) {
        case NONE:
          break;
        case DISCRETE:
          ar >> cardinality >> levels;
          length = cardinality.size();
          break;
        case CONTINUOUS:
          ar >> length;
          break;
        }
      }

      //! Parses the value of a discrete argument from to a string.
      std::size_t parse_discrete(const char* str, std::size_t pos) const {
        assert(category == DISCRETE);
        if (levels.empty()) {
          std::size_t value = parse_string<std::size_t>(str);
          if (value >= cardinality[pos]) {
            std::ostringstream out;
            out << "Value out of bounds: " << value << ">=" << cardinality[pos];
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
          out << "N(" << d->name << ")";
          break;
        case DISCRETE:
          out << "D(" << d->name << "|";
          for (std::size_t i = 0; i < d->length; ++i) {
            if (i > 0) { out << ','; }
            out << d->cardinality[i];
          }
          out << ")";
          break;
        case CONTINUOUS:
          out << "C(" << d->name << "|" << d->length << ")";
          break;
        default:
          out << '?'; break;
        }
        return out;
      }

    }; // struct description

  private:
    //! Constructs a vector with the given descriptor and index.
    explicit vec(const description* desc)
      : desc_(desc) { }

    //! The underlying description.
    const description* desc_;

  }; // class vec

} // namespace libgm

#endif
