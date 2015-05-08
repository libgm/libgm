#ifndef LIBGM_HYBRID_DOMAIN_HPP
#define LIBGM_HYBRID_DOMAIN_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/variable.hpp>

namespace libgm {

  /**
   * A domain that consists of a finite and a vector component.
   */
  template <typename Arg = variable>
  class hybrid_domain {
  public:
    //! Default construct. Creates an empty domain.
    hybrid_domain() { }

    //! Constructs a hybrid domain with the given finite and vector components.
    hybrid_domain(const basic_domain<Arg>& finite,
                  const basic_domain<Arg>& vector)
      : finite_(finite), vector_(vector) { }

    //! Constructs a hybrid domain with the given finite and vector components.
    hybrid_domain(basic_domain<Arg>&& finite,
                  basic_domain<Arg>&& vector)
      : finite_(std::move(finite)), vector_(std::move(vector)) { }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      finite_.save(ar);
      vector_.save(ar);
    }

    //! Loads the domain from an archive.
    void load(iarchive& ar) {
      finite_.load(ar);
      vector_.load(ar);
    }

    //! Returns the finite component of the domain.
    basic_domain<Arg>& finite() {
      return finite_;
    }

    //! Returns the finite component of the domain.
    const basic_domain<Arg>& finite() const {
      return finite_;
    }

    //! Returns the vector component of the domain.
    basic_domain<Arg>& vector() {
      return vector_;
    }

    //! Returns the vector component of the domain.
    const basic_domain<Arg>& vector() const {
      return vector_;
    }

    //! Returns the number of variables in this domain.
    std::size_t size() const {
      return finite_.size() + vector_.size();
    }

    //! Returns the number of finite variables in this domain.
    std::size_t finite_size() const {
      return finite_.size();
    }

    //! Returns the total dimensionality of the vector variables.
    std::size_t vector_size() const {
      return libgm::vector_size(vector_);
    }

    //! Returns true if the domain contains no arguments.
    bool empty() const {
      return finite_.empty() && vector_.empty();
    }

    //! Returns the number of times a variable is present in the domain.
    std::size_t count(Arg v) const {
      if (v.finite()) {
        return finite_.count(v);
      } else if (v.vector()) {
        return vector_.count(v);
      } else {
        return 0;
      }
    }

    //! Returns true if two hybrid domains have the same variables.
    bool operator==(const hybrid_domain& other) const {
      return finite_ == other.finite_ && vector_ == other.vector_;
    }

    //! Returns true if two hybrid domian do not have the same variables.
    bool operator!=(const hybrid_domain& other) const {
      return !(*this == other);
    }

    /**
     * Partitions this domain into those elements that are present in
     * the given map and those that are not.
     */
    template <typename Map>
    void partition(const Map& map,
                   hybrid_domain& intersect, hybrid_domain& difference) const {
      finite_.partition(map, intersect.finite_, difference.finite_);
      vector_.partition(map, intersect.vector_, difference.vector_);
    }

    // Mutations
    //==========================================================================

    //! Adds a variable at the end.
    void push_back(Arg arg) {
      if (arg.finite()) {
        finite().push_back(arg);
      } else if (arg.vector()) {
        vector().push_back(arg);
      } else {
        throw std::invalid_argument("Invalid type for " + arg.str());
      }
    }

    //! Removes all variables from the domain.
    void clear() {
      finite_.clear();
      vector_.clear();
    }

    //! Substitutes arguments in-place according to a map.
    template <typename Map>
    void subst(const Map& map) {
      finite_.subst(map);
      vector_.subst(map);
    }

    //! Swaps the contents of two domains
    friend void swap(hybrid_domain& a, hybrid_domain& b) {
      swap(a.finite(), b.finite());
      swap(a.vector(), b.vector());
    }

  private:
    //! The finite component.
    basic_domain<Arg> finite_;

    //! The vector component.
    basic_domain<Arg> vector_;

  }; // struct hybrid_domain

  /**
   * Prints the hybrid domain to an output stream.
   * \relates hybrid_domain
   */
  template <typename Arg>
  std::ostream&
  operator<<(std::ostream& out, const hybrid_domain<Arg>& d) {
    out << '(' << d.finite() << ", " << d.vector() << ')';
    return out;
  }


  // Set operations
  //============================================================================

  /**
   * Returns the concatenation of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator+(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.finite() + b.finite(), a.vector() + b.vector());
  }

  /**
   * Returns the difference of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator-(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.finite() - b.finite(), a.vector() - b.vector());
  }

  /**
   * Returns the ordered union of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator|(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.finite() | b.finite(), a.vector() | b.vector());
  }

  /**
   * Returns the ordered intersection of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator&(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.finite() & b.finite(), a.vector() & b.vector());
  }

  /**
   * Returns true if two hybrid domains do not have nay elements in common.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool disjoint(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return disjoint(a.finite(), b.finite()) && disjoint(a.vector(), b.vector());
  }

  /**
   * Returns true if two hybrid domains are equivalent.
   * Two hybrid domains are equivalent if their respective finite and
   * vector components are equivalent.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool equivalent(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return equivalent(a.finite(), b.finite())
      && equivalent(a.vector(), b.vector());
  }

  /**
   * Returns true if all the elements of the first domain are also
   * present in the second domain.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool subset(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return subset(a.finite(), b.finite()) && subset(a.vector(), b.vector());
  }

  /**
   * Returns true if all the elements of the second domain are also
   * present in the first domain.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool superset(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return superset(a.finite(), b.finite())
      && superset(a.vector(), b.vector());
  }

  /**
   * Returns true if two domains are type-compatible.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool compatible(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return compatible(a.finite(), b.finite())
      && compatible(a.vector(), b.vector());
  }

} // namespace libgm

#endif
