#ifndef LIBGM_INDEXED_HPP
#define LIBGM_INDEXED_HPP

#include <libgm/argument/traits.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

namespace libgm {

  /**
   * An argument derived from another argument type by attaching an index.
   * Usually this is used to define variables that belong to a process
   * or a field.
   *
   * \tparam Arg
   *         The base argument type that represents a process.
   * \tparam Index
   *         A type that represents the index. Should be trivially copyable.
   */
  template <typename Arg, typename Index = std::ptrdiff_t>
  class indexed {
  public:
    //! The arity of the argument (same as the base argument type).
    using argument_arity = argument_arity_t<Arg>;

    //! The category of the argument (same as the base argument type).
    using argument_category = argument_category_t<Arg>;

    //! The underlying process type.
    using base_type = Arg;

    //! The underlying index type.
    using index_type = Index;

    //! The indexed arguments are not further indexable.
    static const bool is_indexable = false;

    //! Default constructor.
    indexed() {
      : arg_(argument_traits<Arg>::null()), index_() { }

    //! Constructs an argument with the given base argument and index.
    indexed(Arg arg, Index index)
      : arg_(arg), index_(index) { }

    //! Converts the variable to a pair of base argument and index.
    std::pair<Arg, Index> pair() const {
      return { arg_, index_ };
    }

    //! Swaps the contents of two indexed arguments.
    friend void swap(indexed& a, indexed& b) {
      using std::swap;
      swap(a.arg_, b.arg_);
      swap(a.index_, b.index_);
    }

    //! Saves the indexed argument to an archive.
    void save(oarchive& ar) const {
      ar << arg_ << index_;
    }

    //! Loads the indexed argument from an arhive.
    void load(iarchive& ar) const {
      ar >> arg_ >> index_;
    }

    //! Prints an indexed argument to an output stream.
    friend ostd::ostream& operator<<(std::ostream& out, indexed x) {
      out << x.arg_ << '(' << x.index_ << ')';
      return out;
    }

    // Comparison and hashing
    //--------------------------------------------------------------------------

    //! Compares two indexed aguments.
    friend bool operator==(indexed x, indexed y) {
      return x.pair() == y.pair();
    }

    //! Compares two indexed aguments.
    friend bool operator!=(indexed x, indexed y) {
      return x.pair() != y.pair();
    }

    //! Compares two indexed aguments.
    friend bool operator<(indexed x, indexed y) {
      return x.pair() < y.pair();
    }

    //! Compares two indexed aguments.
    friend bool operator>(indexed x, indexed y) {
      return x.pair() > y.pair();
    }

    //! Compares two indexed aguments.
    friend bool operator<=(indexed x, indexed y) {
      return x.pair() <= y.pair();
    }

    //! Compares two indexed aguments.
    friend bool operator>=(indexed x, indexed y) {
      return x.pair() >= y.pair();
    }

    //! Computes the hash of the variable.
    friend std::size_t hash_value(indexed x) {
      std::size_t seed = 0;
      libgm::hash_combine(seed, x.arg_);
      libgm::hash_combine(seed, x.index_);
      return seed;
    }

    // Argument properties
    //--------------------------------------------------------------------------

    //! Returns the arity of an indexed argument (delegates to Arg).
    std::size_t arity() const {
      using libgm::argument_arity;
      return argument_arity(arg_);
    }

    //! Returns the number of values for a discrete argument.
    LIBGM_ENABLE_IF(is_discrete<Arg>::value)
    std::size_t size() const {
      return argument_size(arg_);
    }

    //! Returns the number of values for a single idnex of a discrete argument.
    LIBGM_ENABLE_IF(is_discrete<Arg>::value)
    std::size_t size(std::size_t pos) const {
      return argument_size(arg_, pos);
    }

    //! Returns true if an indexed argument is discrete.
    bool discrete() const {
      return argument_discrete(arg_);
    }

    //! Returns true if an indexed argument is continuou.
    bool continuous() const {
      return argument_continuous(arg_);
    }

  }; // class indexed

} // namespace libgm


namespace std {

  template <typename Arg>
  struct hash<libgm::indexed<Arg> >
    : libgm::default_hash<indexed<Arg> > { };

} // namespace std

#endif
