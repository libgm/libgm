#ifndef LIBGM_VARIABLE_HPP
#define LIBGM_VARIABLE_HPP

#include <libgm/argument/argument_object.hpp>
#include <libgm/argument/basic_domain.hpp>
#include <libgm/functional/hash.hpp>

#include <unordered_map>

namespace libgm {

  // Forward declaration
  template <typename Index, typename Variable> class process;
  class universe;

  /**
   * A class that represents a basic variable. Internally, a variable
   * consists of a pointer to the argument object and time index
   * (which can be empty). The argument object is shared among the
   * variable copies and must persist past the lifetime of this one.
   * Variables are not created directly; instead, they are created
   * through the universe class.
   *
   * This class models the MixedArgument and ProcessVariable concepts.
   */
  class variable {
  public:
    typedef argument_object::category_enum category_enum;

    //! Constructs an empty variable.
    variable()
      : rep_(nullptr), index_(-1) { }

    //! Returns the category of the variable (discrete / continuous).
    category_enum category() const {
      return rep().category;
    }

    //! Returns the name of the variable.
    const std::string& name() const {
      return rep().name;
    }

    //! Returns the levels of the variable.
    const std::vector<std::string>& levels() const {
      return rep().levels;
    }

    //! Parses the value of a discrete variable from a string.
    std::size_t parse_discrete(const char* str) const {
      return rep().parse_discrete(str);
    }

    //! Prints the value of a discrete variable using the stored levels if any.
    void print_discrete(std::ostream& out, std::size_t value) const {
      return rep().print_discrete(out, value);
    }

    // Argument concept
    //==========================================================================

    //! Returns true if two variables are compatible.
    friend bool compatible(variable x, variable y) {
      return x.rep().category == y.rep().category
        && x.rep().size == y.rep().size;
    }

    //! Compares two variables.
    friend bool operator==(variable x, variable y) {
      return x.rep_ == y.rep_ && x.index_ == y.index_;
    }

    //! Compares two variables.
    friend bool operator!=(variable x, variable y) {
      return x.rep_ != y.rep_ || x.index_ != y.index_;
    }

    //! Compares two variables.
    friend bool operator<(variable x, variable y) {
      return x.pair() < y.pair();
    }

    //! Compares two variables.
    friend bool operator>(variable x, variable y) {
      return x.pair() > y.pair();
    }

    //! Computes the hash of the variable.
    friend std::size_t hash_value(variable x) {
      std::size_t seed = 0;
      libgm::hash_combine(seed, x.rep_);
      libgm::hash_combine(seed, x.index_);
      return seed;
    }

    //! Prints a variable to an output stream.
    friend std::ostream& operator<<(std::ostream& out, variable x) {
      out << x.rep();
      if (is_indexed(x)) { out << '(' << index(x) << ')'; }
      return out;
    }

    //! Saves the variable to an archive.
    void save(oarchive& ar) const {
      ar.serialize_dynamic(rep_);
      ar << index_;
    }

    //! Loads the variable from an archive.
    void load(iarchive& ar) {
      rep_ = ar.deserialize_dynamic<argument_object>();
      ar >> index_;
    }

    // DiscreteArgument concept
    //==========================================================================

    //! Returns the number of values for a discrete variable.
    friend std::size_t num_values(variable v) {
      if (v.rep().category == argument_object::DISCRETE) {
        return v.rep().size;
      } else {
        throw std::invalid_argument(
          "Attempt to call num_values() on a variable that is not discrete"
        );
      }
    }

    // ContinuousArgument concept
    //==========================================================================

    //! Returns the number of dimensions for continuous variable.
    friend std::size_t num_dimensions(variable v) {
      if (v.rep().category == argument_object::CONTINUOUS) {
        return v.rep().size;
      } else {
        throw std::invalid_argument(
        "Attempt to call num_dimensions() on a variable that is not continouous"
        );
      }
    }

    // HybridArgument concept
    //==========================================================================

    //! Returns true if the variable is discrete.
    friend bool is_discrete(variable v) {
      return v.rep().category == argument_object::DISCRETE;
    }

    //! Returns true if the variable is continuous.
    friend bool is_continuous(variable v) {
      return v.rep().category == argument_object::CONTINUOUS;
    }

    // ProcessVariable concept
    //==========================================================================

    //! Returns the index of the variable.
    friend std::size_t index(variable v) {
      return v.index_;
    }

    //! Returns true if the variable is associated with a process.
    friend bool is_indexed(variable v) {
      return v.index_ != std::size_t(-1);
    }

    // Private members
    //==========================================================================
  private:
    //! Converts the variable to a pair of hte argument object and index.
    std::pair<const argument_object*, std::size_t> pair() const {
      return { rep_, index_ };
    }

    //! Constructs a free variable with the given argument object.
    explicit variable(const argument_object* rep)
      : rep_(rep), index_(-1) { }

    //! Constructs a proces variable with the given argument object and index.
    variable(const argument_object* rep, std::size_t index)
      : rep_(rep), index_(index) { }

    //! Returns a reference to the underlying argument object.
    const argument_object& rep() const {
      assert(rep_ != nullptr);
      return *rep_;
    }

    // Representation
    //==========================================================================

    //! The underlying representation.
    const argument_object* rep_;

    //! The index associated with the variable.
    std::size_t index_;

    // Friends
    template <typename Index, typename Var> friend class process;
    friend class universe;

  }; // class variable

  //! A type that represents a domain of variables.
  typedef basic_domain<variable> domain;

  //! A type that maps one variable to another.
  typedef std::unordered_map<variable, variable> variable_map;

} // namespace libgm


namespace std {

  template <>
  struct hash<libgm::variable>
    : libgm::default_hash<libgm::variable> { };

} // namespace std

#endif
