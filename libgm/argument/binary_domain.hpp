#ifndef LIBGM_BINARY_DOMAIN_HPP
#define LIBGM_BINARY_DOMAIN_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/unary_domain.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <type_traits>

namespace libgm {

  /**
   * A domain that holds exactly two arguments.
   */
  template <typename Arg>
  class binary_domain {
  public:

    // Domain concept
    typedef Arg key_type;

    // Helper types
    typedef typename argument_traits<Arg>::instance_type instance_type;

    //! Default constructor. Creates a domain with uninitialized arguments.
    binary_domain() { }

    //! Constructs a domain with the given arguments.
    binary_domain(Arg x, Arg y) : x_(x), y_(y) { }

    //! Swaps the contents of two domains.
    friend void swap(binary_domain& a, binary_domain& b) {
      using std::swap;
      swap(a.x_, b.x_);
      swap(a.y_, b.y_);
    }

    //! Returns the first argument.
    const Arg& x() const {
      return x_;
    }

    //! Returns the first argument.
    Arg& x() {
      return x_;
    }

    //! Returns the second argument.
    const Arg& y() const {
      return y_;
    }

    //! Returns the second argument.
    Arg& y() {
      return y_;
    }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      ar << x_ << y_;
    }

    //! Loads the domain from an archive.
    void load(iarchive& ar) {
      ar >> x_ >> y_;
    }

    //! Returns true if two domains are equal.
    friend bool operator==(const binary_domain& a, const binary_domain& b) {
      return a.x_ == b.x_ && a.y_ == b.y_;
    }

    //! Returns true if two domains are not equal.
    friend bool operator!=(const binary_domain& a, const binary_domain& b) {
      return a.x_ != b.x_ || a.y_ != b.y_;
    }

    //! Returns the hash value of a domain.
    friend std::size_t hash_value(const binary_domain& dom) {
      std::size_t seed = 0;
      hash_combine(seed, dom.x_);
      hash_combine(seed, dom.y_);
      return seed;
    }

    //! Prints the domain to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const binary_domain& dom) {
      out << '[';
      argument_traits<Arg>::print(out, dom.x_);
      out << ',';
      argument_traits<Arg>::print(out, dom.y_);
      out << ']';
      return out;
    }

    // Sequence operations
    //==========================================================================

    //! Returns the number of elements of this domain (2).
    std::size_t size() const {
      return 2;
    }

    //! Returns true if the given domain is a prefix of this domain.
    bool prefix(const unary_domain<Arg>& dom) const {
      return x_ == dom.x();
    }

    //! Retursn true if the given domain is a suffix of this domain.
    bool suffix(const unary_domain<Arg>& dom) const {
      return y_ == dom.x();
    }

    // Set operations
    //==========================================================================

    /**
     * Returns the number of times an argument is present in the domain.
     */
    std::size_t count(Arg arg) const {
      return (arg == x_) + (arg == y_);
    }

    /**
     * Given an associative container (a set or a map), extracts an element
     * from this domain that is present in the container and an one that is
     * not.
     * \throw std::invalid_argument if neither or both arguments are present
     *                              in the input set / map.
     */
    template <typename Set>
    void partition(const Set& set,
                   unary_domain<Arg>& present, unary_domain<Arg>& absent) const {
      bool x_present = set.count(x_);
      bool y_present = set.count(y_);
      if (x_present && !y_present) {
        present = x_;
        absent  = y_;
      } else if (y_present && !x_present) {
        present = y_;
        absent  = x_;
      } else {
        throw std::invalid_argument(
          "Map / set does not contain precisely one argumnet"
        );
      }
    }

    /**
     * Returns the ordered union of two domains.
     * \throw std::invalid_argument
     *        if the union does not contain precisely two arguments
     */
    friend binary_domain
    operator+(const binary_domain& a, const unary_domain<Arg>& b) {
      if (!a.count(b.x())) {
        throw std::invalid_argument("The union contains more than 2 arguments");
      }
      return a;
    }

    /**
     * Returns the ordered union of two domains.
     * \throw std::invalid_argument
     *        if hte union does not contain precisely two arguments
     */
    friend binary_domain
    operator+(const unary_domain<Arg>& a, const binary_domain& b) {
      if (!b.count(a.x())) {
        throw std::invalid_argument("The union contains more than 2 arguments");
      }
      return { a.x(), b.x_ == a.x() ? b.y_ : b.x_ };
    }

    /**
     * Returns the ordered difference of two domains.
     * \throw std::invalid_argumnet
     *        if the difference does not contain precisely one argument
     */
    friend unary_domain<Arg>
    operator-(const binary_domain& a, const unary_domain<Arg>& b) {
      if (!a.count(b.x())) {
        throw std::invalid_argument("The difference contains more than 1 argument");
      }
      return a.x_ == b.x() ? a.y_ : a.x_;
    }

    /**
     * Returns true if two domains do not have any arguments in common.
     */
    friend bool disjoint(const binary_domain& a, const binary_domain& b) {
      return !(a.x_ == b.x_ || a.x_ == b.y_ || a.y_ == b.x_ || a.y_ == b.y_);
    }

    /**
     * Returns true if two domains do not have any arguments in common.
     */
    friend bool disjoint(const binary_domain& a, const unary_domain<Arg>& b) {
      return !(a.x_ == b.x() || a.y_ == b.x());
    }

    /**
     * Returns true if two domains do not have any arguments in common.
     */
    friend bool disjoint(const unary_domain<Arg>& a, const binary_domain& b) {
      return !(a.x() == b.x_ || a.x() == b.y_);
    }

    /**
     * Returns true if two domains contain the same set of arguments
     * (disregarding the order).
     */
    friend bool equivalent(const binary_domain& a, const binary_domain& b) {
      return std::minmax(a.x_, a.y_) == std::minmax(b.x_, b.y_);
    }

    /**
     * Returns true if the argument of the unary domain is present
     * in the binary domain.
     */
    friend bool subset(const unary_domain<Arg>& a, const binary_domain& b) {
      return b.count(a.x());
    }

    /**
     * Returns true if the argument of the unary domain is present
     * in the binary domain.
     */
    friend bool superset(const binary_domain& a, const unary_domain<Arg>& b) {
      return a.count(b.x());
    }

    // Argument operations
    //==========================================================================

    /**
     * Returns true if two domains are compatible. Two binary domains are
     * compatible if their arguments are compatible as specified by
     * argument_traits<Arg>.
     */
    friend bool compatible(const binary_domain& a, const binary_domain& b) {
      return argument_traits<Arg>::compatible(a.x_, b.x_)
          && argument_traits<Arg>::compatible(a.y_, b.y_);
    }

    /**
     * Returns the dimensionality for a collection of arguments.
     * This is the the sum of the number of dimensions for the two arguments.
     */
    std::size_t num_dimensions() const {
      return argument_traits<Arg>::num_dimensions(x_)
           + argument_traits<Arg>::num_dimensions(y_);
    }

    /**
     * Returns the number of values for a collection of discrete arguments.
     * This is only supported for discrete univariate arguments.
     */
    template <typename A = Arg,
              typename = std::enable_if_t<is_discrete<A>::value> >
    std::pair<std::size_t, std::size_t> num_values() const {
      return std::make_pair(argument_traits<Arg>::num_values(x_),
                            argument_traits<Arg>::num_values(y_));
    }

    /**
     * Returns the instances of an indexable argument for one index.
     */
    template <typename A = Arg,
              typename = std::enable_if_t<is_indexable<A>::value> >
    binary_domain<instance_type>
    operator()(typename argument_traits<Arg>::index_type index) const {
      return { x_(index), y_(index) };
    }

    /**
     * Substitutes arguments in-place according to a map. The keys of the map
     * must include all the arguments in this domain.
     *
     * \throw std::out_of_range if an argument is not present in the map.
     * \throw std::invalid_argument if the arguments are not compatible
     */
    template <typename Map>
    void substitute(const Map& map) {
      Arg new_x = map.at(x_);
      Arg new_y = map.at(y_);
      if (!argument_traits<Arg>::compatible(x_, new_x) ||
          !argument_traits<Arg>::compatible(y_, new_y)) {
        throw std::invalid_argument("Incompatible arguments");
      }
      x_ = new_x;
      y_ = new_y;
    }

  private:
    //! The first argument.
    Arg x_;

    //! The second argument.
    Arg y_;

  }; // class binary_domain

  /**
   * Concatenates two unary domains into a binary domain.
   * \relates binary_domain
   */
  template <typename Arg>
  binary_domain<Arg>
  concat(const unary_domain<Arg>& a, const unary_domain<Arg>& b) {
    return binary_domain<Arg>(a.x(), b.x());
  }

  /**
   * Converts one binary_domain to a binary_domain with another argument type.
   *
   * \tparam Target
   *         The target argument type. Must be convertible from Source using
   *         argument_cast.
   * \tparam Source
   *         The original argument type.
   * \relates binary_domain
   */
  template <typename Target, typename Source>
  binary_domain<Target> argument_cast(const binary_domain<Source>& dom) {
    static_assert(is_convertible_argument<Source, Target>::value,
                  "Source must be argument-convertible to Target");
    return { argument_cast<Target>(dom.x()), argument_cast<Target>(dom.y()) };
  }

} // namespace libgm

namespace std {

  template <typename Arg>
  struct hash<libgm::binary_domain<Arg>>
    : libgm::default_hash<libgm::binary_domain<Arg>> { };

} // namespace std

#endif
