#ifndef LIBGM_PROCESS_HPP
#define LIBGM_PROCESS_HPP

#include <libgm/argument/argument_object.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/variable.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/graph/vertex_traits.hpp>

#include <unordered_map>

namespace libgm {

  /**
   * A class that represents a process. A process is parameterized by
   * its index type (typically real_t or double), and the variable type.
   * By default, the implementation of process is empty; it needs to
   * be specialized for each variable type. We provide a default
   * implementation for the standard variable type.
   *
   * \tparam Index the index of the process
   * \tparam Variable the variable type associated with this process
   */
  template <typename Index, typename Variable>
  class process { };

  /**
   * Specialization of process for the standard variable type.
   * Each process is represented by apointer to the argument_object.
   */
  template <typename Index>
  class process<Index, variable> {
  public:
    typedef argument_object::category_enum category_enum;

    //! Constructs an empty process.
    process()
      : rep_(nullptr) { }

    //! Returns a special "deleted" process used in certain datastructures.
    static process deleted() {
      return process(argument_object::deleted());
    }

    //! Returns the category of the process (discrete / continuous).
    category_enum category() const {
      return rep().category;
    }

    //! Returns the name of the process.
    const std::string& name() const {
      return rep().name;
    }

    //! Returns the levels of the process.
    const std::vector<std::string>& levels() const {
      return rep().levels;
    }

    //! Parses the value of a discrete process from a string.
    std::size_t parse_discrete(const char* str) const {
      return rep().parse_discrete(str);
    }

    //! Prints the value of a discrete process using the stored levels if any.
    void print_discrete(std::ostream& out, std::size_t value) const {
      return rep().print_discrete(out, value);
    }

    // Argument concept
    //==========================================================================

    //! The category of the argument.
    typedef mixed_argument_tag argument_category;

    //! The type of index associated with the process.
    typedef Index argument_index;

    //! Returns true if two processes are compatible.
    static bool compatible(process x, process y) {
      return x.rep().category == y.rep().category
        && x.rep().size == y.rep().size;
    }

    //! Compares two processes.
    friend bool operator==(process x, process y) {
      return x.rep_ == y.rep_;
    }

    //! Copmares two processes.
    friend bool operator!=(process x, process y) {
      return x.rep_ != y.rep_;
    }

    //! Compares two proceses.
    friend bool operator<(process x, process y) {
      return x.rep_ < y.rep_;
    }

    //! Compares two processes.
    friend bool operator>(process x, process y) {
      return x.rep_ > y.rep_;
    }

    //! Computes the hash of the variable.
    friend std::size_t hash_value(process x) {
      return boost::hash_value(x.rep_);
    }

    //! Prints a process to an output stream.
    friend std::ostream& operator<<(std::ostream& out, process x) {
      if (x.rep_) {
        out << x.rep();
      } else {
        out << "null";
      }
      return out;
    }

    //! Saves the process to an archive.
    void save(oarchive& ar) const {
      ar.serialize_dynamic(rep_);
    }

    //! Loads the process from an archive.
    void load(iarchive& ar) {
      rep_ = ar.deserialize_dynamic<argument_object>();
    }

    // DiscreteArgument concept
    //==========================================================================

    //! Returns the number of values for a discrete process.
    std::size_t num_values() {
      if (rep().category == argument_object::DISCRETE) {
        return rep().size;
      } else {
        throw std::invalid_argument(
          "Attempt to call num_values() on a process that is not discrete"
        );
      }
    }

    // ContinuousArgument concept
    //==========================================================================

    //! Returns the number of dimensions for a continuous process.
    std::size_t num_dimensions() {
      if (rep().category == argument_object::CONTINUOUS) {
        return rep().size;
      } else {
        throw std::invalid_argument(
         "Attempt to call num_dimensions() on a process that is not continouous"
        );
      }
    }

    // HybridArgument concept
    //==========================================================================

    //! Returns true if the process is discrete.
    bool is_discrete() {
      return rep().category == argument_object::DISCRETE;
    }

    //! Returns true if the process is continuous.
    bool is_continuous() {
      return rep().category == argument_object::CONTINUOUS;
    }

    // ProcessVariable concept
    //==========================================================================

    //! The type of variables represented by this process.
    typedef variable variable_type;

    //! Returns the variable with the given index.
    variable operator()(Index index) const {
      return variable(rep_, index);
    }

    // Private members
    //==========================================================================
  private:
    //! Constructs process with the given argument object.
    explicit process(const argument_object* rep)
      : rep_(rep) { }

    //! Returns a reference to the underlying argument object.
    const argument_object& rep() const {
      assert(rep_ != nullptr);
      return *rep_;
    }

    //! The underlying representation.
    const argument_object* rep_;

    // Friends
    friend class universe;

  }; // class process

  //! A discrete-time process with the default variable type.
  typedef process<std::size_t, variable> dprocess;

  //! A continuous-time process with the default variable type.
  typedef process<double, variable> cprocess;

  //! A type that repesents a domain over discrete processes.
  typedef basic_domain<dprocess> dprocess_domain;

  //! A type that represents a domain over continuous processes.
  typedef basic_domain<cprocess> cprocess_domain;

  //! A type that maps one discrete process to another.
  typedef std::unordered_map<dprocess, dprocess> dprocess_map;

  //! A type that maps one continuous process to another.
  typedef std::unordered_map<cprocess, cprocess> cproces_map;

  // Traits
  //============================================================================

  /**
   * A specialization of vertex_traits for process.
   */
  template <typename Index>
  struct vertex_traits<process<Index, variable> > {
    typedef process<Index, variable> process_type;

    //! Returns the default-constructed process.
    static process_type null() { return process_type(); }

    //! Returns a special "deleted" process.
    static process_type deleted() { return process_type::deleted(); }

    //! Processes use the default hasher.
    typedef std::hash<process_type> hasher;
  };

} // namespace libgm


namespace std {

  template <typename Index>
  struct hash<libgm::process<Index, libgm::variable>>
    : libgm::default_hash<libgm::process<Index, libgm::variable>> { };

} // namespace std

#endif
