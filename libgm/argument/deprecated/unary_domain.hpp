#ifndef LIBGM_UNARY_DOMAIN_HPP
#define LIBGM_UNARY_DOMAIN_HPP

#include <libgm/argument/traits.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <iostream>
#include <sstream>
#include <type_traits>

namespace libgm {

  /**
   * A domain that holds exactly one argument. This is mostly a convenience
   * wrapper around Arg and its traits, in order to model the Domain concept.
   */
  template <typename Arg>
  class unary_domain {
  public:
    // Domain concept
    typedef Arg key_type;

    // Helper types
    typedef typename argument_traits<Arg>::instance_type instance_type;

    //! Default constructor. Creates a domain with an uninitialized argument.
    unary_domain() { }

    //! Constructs a domain with the given argument.
    unary_domain(Arg x) : x_(x) { }

    //! Returns the argument of the domain.
    const Arg& x() const {
      return x_;
    }

    //! Returns the argument of the domain.
    Arg& x() {
      return x_;
    }

    //! Swaps the contents of two domains.
    friend void swap(unary_domain& a, unary_domain& b) {
      using std::swap;
      swap(a.x_, b.x_);
    }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      ar << x_;
    }

    //! Loads the domain from an archive.
    void load(iarchive& ar) {
      ar >> x_;
    }

    //! Returns true if the two domains are equal.
    friend bool operator==(const unary_domain& a, const unary_domain& b) {
      return a.x_ == b.x_;
    }

    //! Returns true if the two domains are not equal.
    friend bool operator!=(const unary_domain& a, const unary_domain& b) {
      return a.x_ != b.x_;
    }

    //! Returns the hash value of a domain.
    friend std::size_t hash_value(const unary_domain& dom) {
      typename argument_traits<Arg>::hasher hasher;
      return hasher(dom.x_);
    }

    //! Prints the domain to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const unary_domain& dom) {
      out << '[';
      argument_traits<Arg>::print(out, dom.x_);
      out << ']';
      return out;
    }

    // Container operations
    //==========================================================================

    //! Returns the number of elements of this domain (1).
    std::size_t size() const {
      return 1;
    }

    // Set operations
    //==========================================================================

    /**
     * Returns the number of times an argument is present in the domain.
     * This operation has a constant time complexity.
     */
    std::size_t count(Arg arg) const {
      return x_ == arg;
    }

    /**
     * Returns true if two domains do not have any arguments in common.
     * For unary_domain, this is the same as operator!=.
     */
    friend bool disjoint(const unary_domain& a, const unary_domain& b) {
      return a.x_ != b.x_;
    }

    /**
     * Returns true if two domains contain the same set of arguments.
     * For unary_domain, this is the same as operator==.
     */
    friend bool equivalent(const unary_domain& a, const unary_domain& b) {
      return a.x_ == b.x_;
    }

    // Argument operations
    //==========================================================================

    /**
     * Returns true if two domains have compatible arguments.
     */
    friend bool compatible(const unary_domain& a, const unary_domain& b) {
      return argument_traits<Arg>::compatible(a.x_, b.x_);
    }

    /**
     * Returns the dimensionality of the domain's argument.
     */
    std::size_t num_dimension() const {
      return argument_traits<Arg>::num_dimensions(x_);
    }

    /**
     * Returns the numebr of values of a discrete argument domain.
     */
    template <typename A = Arg,
              typename = std::enable_if_t<is_discrete<A>::value> >
    std::size_t num_values() const {
      return argument_traits<Arg>::num_values(x_);
    }

    /**
     * Returns the instance of an indexable argument for one index.
     */
    template <typename A = Arg,
              typename = std::enable_if_t<is_indexable<A>::value> >
    unary_domain<instance_type>
    operator()(typename argument_traits<Arg>::index_type index) const {
      return x_(index);
    }

    /**
     * Substitutes argumetns in-place according to a map. The map must include
     * the argument of this domain.
     * \throw std::out_of_range if an argumetn is not present in the map.
     * \throw std::invalid_argumetn if the arguments are not compatible.
     */
    template <typename Map>
    void substitute(const Map& map) {
      Arg new_x = map.at(x_);
      if (!argument_traits<Arg>::compatible(x_, new_x)) {
        std::ostringstream out;
        out << "Incompatible arguments ";
        argument_traits<Arg>::print(out, x_);
        out << " and ";
        argument_traits<Arg>::print(out, new_x);
        throw std::invalid_argument(out.str());
      }
      x_ = new_x;
    }

  private:
    //! The single argument.
    Arg x_;

  }; // class unary_domain

  /**
   * Converts a unary_domain to a unary_domain with another argument type.
   *
   * \tparam Target
   *         The target argument type. Must be convertible from Source using
   *         argument_cast.
   * \tparam Source
   *         The original argument type.
   * \relates unary_domain
   */
  template <typename Target, typename Source>
  unary_domain<Target> argument_cast(const unary_domain<Source>& dom) {
    static_assert(is_convertible_argument<Source, Target>::value,
                  "Source must be argument-convertible to Target");
    return argument_cast<Target>(dom.x());
  }

} // namespace libgm

#endif
