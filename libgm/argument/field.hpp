#ifndef LIBGM_FIELD_HPP
#define LIBGM_FIELD_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <utility>

namespace libgm {

  /**
   * A class that represents a random field. A field is parameterized by
   * the argument type (such as var or vec) and the index type (such as
   * std::size_t or double).
   *
   * \tparam Arg type representing an instantiation of the field for one index
   * \tparam Index the index of the field
   */
  template <typename Arg, typename Index>
  class field {
  public:

    //! The category of the field (the same as the category of the argument).
    typedef typename argument_traits<Arg>::argument_category argument_category;

    //! The arity of the field (the same as the arity of the argument).
    typedef typename argument_traits<Arg>::argument_arity argument_arity;

    //! The argument without its index (taken from the underlying argument).
    typedef typename argument_traits<Arg>::descriptor descriptor;

    //! The index associated with a field.
    typedef Index index_type;

    //! The instance for one index value.
    typedef Arg instance_type;

    // Constructors and accessors
    //==========================================================================

    /**
     * Constructs a null field. This is the field with the same descriptor
     * as the null argument Arg, as given by vertex_traits<Arg>:null().
     */
    field()
      : desc_(argument_traits<Arg>::desc(vertex_traits<Arg>::null())) { }

    /**
     * Constructs a field with the given descriptor. This constructor is
     * intentionally not explicit, in order to permit easy conversion from
     * user-defined argument types.
     */
    field(descriptor desc)
      : desc_(desc) { }

    //! Saves the field to an archive.
    void save(oarchive& ar) const {
      //ar << desc_;
    }

    //! Loads the field from an archive.
    void load(iarchive& ar) {
      //ar >> desc_;
    }

    //! Prints a field to an output stream.
    friend std::ostream& operator<<(std::ostream& out, field x) {
      if (x == field()) {
        out << "null";
      } else {
        out << x.desc_;
      }
      return out;
    }

    // Comparisons
    //==========================================================================

    //! Compares two fields.
    friend bool operator==(field f, field g) {
      return f.desc_ == g.desc_;
    }

    //! Copmares two fields.
    friend bool operator!=(field f, field g) {
      return f.desc_ != g.desc_;
    }

    //! Compares two fields.
    friend bool operator<(field f, field g) {
      return f.desc_ < g.desc_;
    }

    //! Compares two field.
    friend bool operator>(field f, field g) {
      return f.desc_ > g.desc_;
    }

    //! Compares two fields.
    friend bool operator<=(field f, field g) {
      return f.desc_ <= g.desc_;
    }

    //! Compares two field.
    friend bool operator>=(field f, field g) {
      return f.desc_ >= g.desc_;
    }

    // Traits
    //==========================================================================
    /**
     * Returns true if two fields are compatible.
     */
    static bool compatible(field f, field g) {
      return argument_traits<Arg>::compatible(f(Index()), g(Index()));
    }

    /**
     * Returns the dimensionality of the field.
     */
    std::size_t num_dimensions() const {
      return argument_traits<Arg>::num_dimensions(Arg(desc_, Index()));
    }

    /**
     * Returns the number of values of a field.
     * This function is only supported for discrete-valued fields.
     */
    template <bool B = is_discrete<Arg>::value>
    typename std::enable_if<B, std::size_t>::type num_values() const {
      return argument_traits<Arg>::num_values(Arg(desc_, Index()));
    }

    /**
     * Returns the number of values of a field at a particular position.
     * This function is only supported for multivariate discrete-valued fields.
     */
    template <bool B = is_discrete<Arg>::value && is_multivariate<Arg>::value>
    typename std::enable_if<B, std::size_t>::type
    num_values(std::size_t pos) const {
      return argument_traits<Arg>::num_values(Arg(desc_, Index()), pos);
    }

    /**
     * Returns true if the field is discrete-valued.
     * This function is only supported for mixed arguments.
     */
    template <bool B = is_mixed<Arg>::value>
    typename std::enable_if<B, bool>::type discrete() const {
      return argument_traits<Arg>::discrete(Arg(desc_, Index()));
    }

    /**
     * Returns true if the field is continuous-valued.
     * This function is only supported for mixed arguments.
     */
    template <bool B = is_mixed<Arg>::value>
    typename std::enable_if<B, bool>::type continuous() const {
      return argument_traits<Arg>::continuous(Arg(desc_, Index()));
    }

    //! Returns the descriptor of the field.
    descriptor desc() const {
      return desc_;
    }

    //! Returns the field instantiated to the given index.
    Arg operator()(Index index) const {
      return Arg(desc_, index);
    }

  private:
    //! The underlying representation.
    descriptor desc_;

  }; // class field

} // namespace libgm

namespace std {

  template <typename Arg, typename Index>
  struct hash<libgm::field<Arg, Index>> {
    typedef libgm::field<Arg, Index> argument_type;
    typedef std::size_t result_type;

    std::size_t operator()(libgm::field<Arg, Index> f) const {
      return arg_hash(f(Index()));
    }

  private:
    typename libgm::argument_traits<Arg>::hasher arg_hash;
  };

} // namespace std

#endif
