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
   * std::size_t or double). The field inherits its properties, including
   * the arity and size, from the underlying argument.
   *
   * \tparam Arg
   *         A type representing an instantiation of the field for one index.
   * \tparam Index
   *         The index type.
   */
  template <typename Arg, typename Index>
  class field {
  public:

    //! The category of the field (the same as the category of the argument).
    using argument_category = typename argument_traits<Arg>::argument_category;

    //! The arity of the field (the same as the arity of the argument).
    using argument_arity = typename argument_traits<Arg>::argument_arity;

    //! The argument without its index (taken from the underlying argument).
    using descriptor = typename argument_traits<Arg>::descriptor;

    //! The index associated with a field.
    using index_type = Index;

    //! The instance for one index value.
    using instance_type = Arg;

    // Constructors and accessors
    //--------------------------------------------------------------------------

    /**
     * Constructs a null field. This is the field with the same descriptor
     * as the null argument Arg, as given by vertex_traits<Arg>:null().
     */
    field()
      : desc_(argument_descriptor(vertex_traits<Arg>::null())) { }

    /**
     * Constructs a field with the given descriptor. This constructor is
     * intentionally not explicit, in order to permit easy conversion from
     * user-defined argument types.
     */
    field(descriptor desc)
      : desc_(desc) { }

    //! Saves the field to an archive.
    void save(oarchive& ar) const {
      ar << desc_;
    }

    //! Loads the field from an archive.
    void load(iarchive& ar) {
      ar >> desc_;
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
    //--------------------------------------------------------------------------

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

    // Properties
    //--------------------------------------------------------------------------

    //! Returns the arity of the field.
    std::size_t arity() const {
      using libgm::argument_arity;
      return argument_arity(Arg(desc_, Index()));
    }

    //! Returns the number of values of a discrete, univariate field.
    LIBGM_ENABLE_IF(is_discrete<Arg>::value && is_univariate<Arg>::value)
    std::size_t size() const {
      return argument_size(Arg(desc_, Index()));
    }

    //! Returns the number of values of a discrete field.
    LIBGM_ENABLE_IF(is_discrete<Arg>::value)
    std::size_t size(std::size_t pos) const {
      return argument_size(Arg(desc_, Index()), pos);
    }

    //! Returns true if a mixed field is discrete-valued.
    LIBGM_ENABLE_IF(is_mixed<Arg>::value)
    bool discrete() const {
      return argument_discrete(Arg(desc_, Index()));
    }

    //! Returns true if a mixed field is continuous-valued.
    LIBGM_ENABLE_IF(is_mixed<Arg>::value)
    bool continuous() const {
      return argument_continuous(Arg(desc_, Index()));
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
