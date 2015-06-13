#ifndef LIBGM_HYBRID_DOMAIN_HPP
#define LIBGM_HYBRID_DOMAIN_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/variable.hpp>

#include <sstream>

namespace libgm {

  /**
   * A domain that consists of a discrete and a continuous component.
   *
   * \tparam Arg a type that satisfies the MixedArgument concept
   */
  template <typename Arg = variable>
  class hybrid_domain {
  public:
    //! Default construct. Creates an empty domain.
    hybrid_domain() { }

    //! Constructs a hybrid domain with the given finite and vector components.
    hybrid_domain(const basic_domain<Arg>& discrete,
                  const basic_domain<Arg>& continuous)
      : discrete_(discrete), continuous_(continuous) { }

    //! Constructs a hybrid domain with the given finite and vector components.
    hybrid_domain(basic_domain<Arg>&& discrete,
                  basic_domain<Arg>&& continuous)
      : discrete_(std::move(discrete)), continuous_(std::move(continuous)) { }

    //! Conversion from a vector.
    hybrid_domain(const std::vector<Arg>& args) {
      for (Arg arg : args) {
        push_back(arg);
      }
    }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      discrete_.save(ar);
      continuous_.save(ar);
    }

    //! Loads the domain from an archive.
    void load(iarchive& ar) {
      discrete_.load(ar);
      continuous_.load(ar);
    }

    //! Returns the discrete component of the domain.
    basic_domain<Arg>& discrete() {
      return discrete_;
    }

    //! Returns the discrete component of the domain.
    const basic_domain<Arg>& discrete() const {
      return discrete_;
    }

    //! Returns the continuous component of the domain.
    basic_domain<Arg>& continuous() {
      return continuous_;
    }

    //! Returns the continuous component of the domain.
    const basic_domain<Arg>& continuous() const {
      return continuous_;
    }

    //! Returns the number of arguments in this domain.
    std::size_t size() const {
      return discrete_.size() + continuous_.size();
    }

    //! Returns the number of arguments in the discrete component.
    std::size_t discrete_size() const {
      return discrete_.size();
    }

    //! Returns the number of argumetnsin the continuous component.
    std::size_t continuous_size() const {
      return continuous_.size();
    }

    //! Returns true if the domain contains no arguments.
    bool empty() const {
      return discrete_.empty() && continuous_.empty();
    }

    //! Returns the number of times a variable is present in the domain.
    std::size_t count(Arg arg) const {
      if (is_discrete(arg)) {
        return discrete_.count(arg);
      } else if (is_continuous(arg)) {
        return continuous_.count(arg);
      } else {
        return 0;
      }
    }

    //! Returns true if two hybrid domains have the same variables.
    bool operator==(const hybrid_domain& other) const {
      return discrete_ == other.discrete_ && continuous_ == other.continuous_;
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
      discrete_.partition(map, intersect.discrete_, difference.discrete_);
      continuous_.partition(map, intersect.continuous_, difference.continuous_);
    }

    // Mutations
    //==========================================================================

    //! Adds a variable at the end.
    void push_back(Arg arg) {
      if (is_discrete(arg)) {
        discrete().push_back(arg);
      } else if (is_continuous(arg)) {
        continuous().push_back(arg);
      } else {
        std::ostringstream out;
        out << "hybrid_domain::push_back: unknown type of " << arg;
        throw std::invalid_argument(out.str());
      }
    }

    //! Removes all variables from the domain.
    void clear() {
      discrete_.clear();
      continuous_.clear();
    }

    //! Substitutes arguments in-place according to a map.
    template <typename Map>
    void subst(const Map& map) {
      discrete_.subst(map);
      continuous_.subst(map);
    }

    //! Swaps the contents of two domains
    friend void swap(hybrid_domain& a, hybrid_domain& b) {
      swap(a.discrete(), b.discrete());
      swap(a.continuous(), b.continuous());
    }

  private:
    //! The discrete component.
    basic_domain<Arg> discrete_;

    //! The continuous component.
    basic_domain<Arg> continuous_;

  }; // struct hybrid_domain

  /**
   * Prints the hybrid domain to an output stream.
   * \relates hybrid_domain
   */
  template <typename Arg>
  std::ostream&
  operator<<(std::ostream& out, const hybrid_domain<Arg>& dom) {
    out << '(' << dom.discrete() << ", " << dom.continuous() << ')';
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
    return hybrid_domain<Arg>(a.discrete() + b.discrete(),
                              a.continuous() + b.continuous());
  }

  /**
   * Returns the difference of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator-(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.discrete() - b.discrete(),
                              a.continuous() - b.continuous());
  }

  /**
   * Returns the ordered union of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator|(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.discrete() | b.discrete(),
                              a.continuous() | b.continuous());
  }

  /**
   * Returns the ordered intersection of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator&(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.discrete() & b.discrete(),
                              a.continuous() & b.continuous());
  }

  /**
   * Returns true if two hybrid domains do not have nay elements in common.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool disjoint(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return disjoint(a.discrete(), b.discrete())
      && disjoint(a.continuous(), b.continuous());
  }

  /**
   * Returns true if two hybrid domains are equivalent.
   * Two hybrid domains are equivalent if their respective discrete and
   * continuous components are equivalent.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool equivalent(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return equivalent(a.discrete(), b.discrete())
      && equivalent(a.continuous(), b.continuous());
  }

  /**
   * Returns true if all the elements of the first domain are also
   * present in the second domain.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool subset(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return subset(a.discrete(), b.discrete())
      && subset(a.continuous(), b.continuous());
  }

  /**
   * Returns true if all the elements of the second domain are also
   * present in the first domain.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool superset(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return superset(a.discrete(), b.discrete())
      && superset(a.continuous(), b.continuous());
  }

  // Argument operations
  //============================================================================

  /**
   * Returns true if two domains are compatible.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool compatible(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return compatible(a.discrete(), b.discrete())
      && compatible(a.continuous(), b.continuous());
  }

  /**
   * Returns the number of values for the discrete component of
   * a hybrid_domain.
   * \throws std::out_of_range in case of overflow
   * \relates hybrid_domain
   */
  template <typename Arg>
  std::size_t num_values(const hybrid_domain<Arg>& dom) {
    return num_values(dom.discrete());
  }

  /**
   * Returns the number of dimensions for the continuous component of
   * a hybrid_domain.
   * \relates hybrid_domain
   */
  template <typename Arg>
  std::size_t num_dimensions(const hybrid_domain<Arg>& dom) {
    return num_dimensions(dom.continuous());
  }

} // namespace libgm

#endif
