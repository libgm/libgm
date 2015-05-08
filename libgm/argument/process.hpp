#ifndef LIBGM_PROCESS_HPP
#define LIBGM_PROCESS_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/variable.hpp>
#include <libgm/functional/hash.hpp>

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

    //! The index type of the variables represented by this process.
    typedef Index index_type;

    //! The type of variables represented by this process.
    typedef variable variable_type;

    //! Constructs an empty process.
    process()
      : rep_(nullptr) { }

    //! Converts the process to bool indicating if the proces is empty.
    explicit operator bool() const {
      return rep_ != nullptr;
    }

    //! Saves the process to an archive.
    void save(oarchive& ar) const {
      ar.serialize_dynamic(rep_);
    }

    //! Loads the process from an archive.
    void load(iarchive& ar) {
      rep_ = ar.deserialize_dynamic<argument_object>();
    }

    //! Returns the cardinality / dimensionality of the process variables.
    std::size_t size() const {
      return rep().size;
    }

    //! Returns the categoru of the process (discrete / continuous).
    category_enum category() const {
      return rep().category;
    }

    //! Returns true if the variable is finite.
    bool finite() const {
      return rep().category == argument_object::FINITE;
    }

    //! Returns true if the variable is vector.
    bool vector() const {
      return rep().category == argument_object::VECTOR;
    }

    //! Returns the variable with the given index.
    variable operator()(Index index) const{
      return variable(rep_, index);
    }

    //! Returns the name of the variable.
    const std::string& name() const {
      return rep().name;
    }

    //! Conversion to human-readable representation.
    std::string str() const {
      return rep().str();
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

    //! Returns true if two processes are type-compatible.
    friend bool compatible(process x, process y) {
      return x.size() == y.size() && x.category() == y.category();
    }

    //! Computes the hash of the variable.
    friend std::size_t hash_value(process x) {
      return boost::hash_value(x.rep_);
    }

    //! Prints a process to an output stream.
    friend std::ostream& operator<<(std::ostream& out, process x) {
      out << x.rep();
      return out;
    }

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
    friend class std::hash<process>;

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

} // namespace libgm


namespace std {

  template <typename Index>
  struct hash<libgm::process<Index, libgm::variable>> {
    typedef libgm::process<Index, libgm::variable> argument_type;
    typedef std::size_t result_type;
    std::size_t operator()(argument_type x) const {
      return hash_value(x);
    }
  };

} // namespace std

#endif
