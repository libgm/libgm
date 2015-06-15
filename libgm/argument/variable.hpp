#ifndef LIBGM_VARIABLE_HPP
#define LIBGM_VARIABLE_HPP

#include <libgm/argument/argument_object.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/basic_domain.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/graph/vertex_traits.hpp>

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
      : rep_(nullptr), index_(0) { }

    //! Returns a special "deleted" variable used in certain datastructures.
    static variable deleted() {
      return variable(argument_object::deleted(), 0);
    }

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

    //! The category of the argument.
    typedef mixed_argument_tag argument_category;

    //! The type of index associated with the variable.
    typedef std::size_t argument_index;

    //! Returns true if two variables are compatible.
    static bool compatible(variable x, variable y) {
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
      if (x.rep_) {
        out << x.rep();
        if (x.is_indexed()) { out << '(' << x.index() << ')'; }
      } else {
        out << "null";
      }
      return out;
    }

    //! Saves the variable to an archive.
    void save(oarchive& ar) const {
      ar.serialize_dynamic(rep_);
      if (rep_) { ar << index_; }
    }

    //! Loads the variable from an archive.
    void load(iarchive& ar) {
      rep_ = ar.deserialize_dynamic<argument_object>();
      if (rep_) { ar >> index_; }
    }

    // DiscreteArgument concept
    //==========================================================================

    //! Returns the number of values for a discrete variable.
    std::size_t num_values() {
      if (rep().category == argument_object::DISCRETE) {
        return rep().size;
      } else {
        throw std::invalid_argument(
          "Attempt to call num_values() on a variable that is not discrete"
        );
      }
    }

    // ContinuousArgument concept
    //==========================================================================

    //! Returns the number of dimensions for continuous variable.
    std::size_t num_dimensions() {
      if (rep().category == argument_object::CONTINUOUS) {
        return rep().size;
      } else {
        throw std::invalid_argument(
        "Attempt to call num_dimensions() on a variable that is not continouous"
        );
      }
    }

    // MixedArgument concept
    //==========================================================================

    //! Returns true if the variable is discrete.
    bool is_discrete() {
      return rep().category == argument_object::DISCRETE;
    }

    //! Returns true if the variable is continuous.
    bool is_continuous() {
      return rep().category == argument_object::CONTINUOUS;
    }

    // ProcessVariable concept
    //==========================================================================

    //! Returns the index of the variable.
    std::size_t index() {
      return index_;
    }

    //! Returns true if the variable is associated with a process.
    bool is_indexed() {
      return index_ != std::size_t(-1);
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

  // Traits
  //============================================================================

  /**
   * A specialization of vertex_traits for variable.
   */
  template <>
  struct vertex_traits<variable> {
    //! Returns the default-constructed variable.
    static variable null() { return variable(); }

    //! Returns a special "deleted" variable.
    static variable deleted() { return variable::deleted(); }

    //! Prints the variable to an output stream.
    static void print(std::ostream& out, variable v) { out << v; }

    //! Variables use the default hasher.
    typedef std::hash<variable> hasher;
  };

} // namespace libgm


namespace std {

  template <>
  struct hash<libgm::variable>
    : libgm::default_hash<libgm::variable> { };

} // namespace std

#endif
