#ifndef LIBGM_HYBRID_DOMAIN_HPP
#define LIBGM_HYBRID_DOMAIN_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/argument_cast.hpp>
#include <libgm/argument/traits.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/hybrid_index.hpp>
#include <libgm/iterator/join_iterator.hpp>

#include <sstream>

namespace libgm {

  /**
   * A domain that consists of a discrete and a continuous component.
   *
   * \tparam Arg a type that models the MixedArgument concept
   */
  template <typename Arg>
  class hybrid_domain {
    static_assert(is_mixed<Arg>::value, "Arg must be a mixed argument type");

  public:
    // Container concept
    typedef Arg value_type;
    typedef Arg& reference;
    typedef const Arg& const_reference;
    typedef join_iterator<typename domain<Arg>::iterator> iterator;
    typedef join_iterator<typename domain<Arg>::const_iterator> const_iterator;
    typedef std::ptrdiff_t difference_type;
    typedef std::size_t size_t;

    // Domain concept
    typedef Arg key_type;
    typedef hybrid_index index_type;

    // Helper types
    typedef typename argument_traits<Arg>::instance_type instance_type;

    //! Default construct. Creates an empty domain.
    hybrid_domain() { }

    //! Constructs a hybrid domain with the given finite and vector components.
    hybrid_domain(const domain<Arg>& discrete,
                  const domain<Arg>& continuous)
      : discrete_(discrete), continuous_(continuous) { }

    //! Constructs a hybrid domain with the given finite and vector components.
    hybrid_domain(domain<Arg>&& discrete,
                  domain<Arg>&& continuous)
      : discrete_(std::move(discrete)), continuous_(std::move(continuous)) { }

    //! Conversion from a vector.
    hybrid_domain(const std::vector<Arg>& args) {
      for (Arg arg : args) {
        push_back(arg);
      }
    }

    //! Construction from an initializer list
    hybrid_domain(std::initializer_list<Arg> args) {
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

    //! Returns the hash value of a domain.
    friend std::size_t hash_value(const hybrid_domain& dom) {
      std::size_t seed = 0;
      hash_combine(seed, dom.discrete());
      hash_combine(seed, dom.continuous());
      return seed;
    }

    //! Prints the hybrid domain to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const hybrid_domain& dom) {
      out << '(' << dom.discrete() << ", " << dom.continuous() << ')';
      return out;
    }

    // Accessors
    //==========================================================================

    //! Returns the discrete component of the domain.
    domain<Arg>& discrete() {
      return discrete_;
    }

    //! Returns the discrete component of the domain.
    const domain<Arg>& discrete() const {
      return discrete_;
    }

    //! Returns the continuous component of the domain.
    domain<Arg>& continuous() {
      return continuous_;
    }

    //! Returns the continuous component of the domain.
    const domain<Arg>& continuous() const {
      return continuous_;
    }

    // Container operations
    //==========================================================================

    //! Returns an iterator to the beginning of the domain.
    iterator begin() {
      return { discrete().begin(), discrete().end(), continuous().begin() };
    }

    //! Returns an iterator to the beginning of the domain.
    const_iterator begin() const {
      return { discrete().begin(), discrete().end(), continuous().begin() };
    }

    //! Returns an iterator to the beginning of the domain.
    const_iterator cbegin() const {
      return { discrete().begin(), discrete().end(), continuous().begin() };
    }

    //! Returns an iterator to the end of the domain.
    iterator end() {
      return { discrete().end(), discrete().end(), continuous().end() };
    }

    //! Returns an iterator to the end of the domain.
    const_iterator end() const {
      return { discrete().end(), discrete().end(), continuous().end() };
    }

    //! Returns an iterator to the end of the domain.
    const_iterator cend() const {
      return { discrete().end(), discrete().end(), continuous().end() };
    }

    //! Returns the number of arguments in this domain.
    std::size_t size() const {
      return discrete_.size() + continuous_.size();
    }

    //! Returns the maximum number of arguments that can be in a domain.
    std::size_t max_size() const {
      return discrete_.max_size() / 2 + continuous_.max_size() / 2;
    }

    //! Returns true if the domain contains no arguments.
    bool empty() const {
      return discrete_.empty() && continuous_.empty();
    }

    //! Returns true if two domains have the same arguments.
    friend bool operator==(const hybrid_domain& a, const hybrid_domain& b) {
      return a.discrete_ == b.discrete_ && a.continuous_ == b.continuous_;
    }

    //! Returns true if two domains do not have the same arguments.
    friend bool operator!=(const hybrid_domain& a, const hybrid_domain& b) {
      return a.discrete_ != b.discrete_ || a.continuous_ != b.continuous_;
    }

    //! Swaps the contents of two domains
    friend void swap(hybrid_domain& a, hybrid_domain& b) {
      swap(a.discrete_, b.discrete_);
      swap(a.continuous_, b.continuous_);
    }

    // SequenceContainer operations
    //==========================================================================

    //! Removes all the arguments from the domain.
    void clear() {
      discrete_.clear();
      continuous_.clear();
    }

    //! Appends an argument at the end.
    void push_back(Arg arg) {
      if (argument_traits<Arg>::discrete(arg)) {
        discrete().push_back(arg);
      } else if (argument_traits<Arg>::continuous(arg)) {
        continuous().push_back(arg);
      } else {
        std::ostringstream out;
        out << "hybrid_domain::push_back: unknown type of argument ";
        argument_traits<Arg>::print(out, arg);
        throw std::invalid_argument(out.str());
      }
    }

    // Sequence operations
    //==========================================================================

    //! Returns true if the given domain is a prefix of this domain.
    bool prefix(const hybrid_domain& dom) const {
      return discrete_.prefix(dom.discrete_) && continuous_.prefix(continuous_);
    }

    //! Returns true if the given domain is a suffix of this domain.
    bool suffix(const hybrid_domain& dom) const {
      return discrete_.suffix(dom.discrete_) && continuous_.suffix(continuous_);
    }

    /**
     * Partitions this domain into those arguments that are present in the
     * given associative container (set or map) and those that are not.
     */
    template <typename Set>
    void partition(const Set& set,
                   hybrid_domain& present, hybrid_domain& absent) const {
      discrete_.partition(set, present.discrete_, absent.discrete_);
      continuous_.partition(set, present.continuous_, absent.continuous_);
    }

    /**
     * Returns the concatenation of two hybrid domains.
     * This operation has a liner time complexity.
     */
    friend hybrid_domain
    concat(const hybrid_domain& a, const hybrid_domain& b) {
      return hybrid_domain(concat(a.discrete(), b.discrete()),
                           concat(a.continuous(), b.continuous()));
    }

    /**
     * Removes the duplicate arguments from the domain in place.
     * Does not preserve the relative order of arguments in the domain.
     */
    hybrid_domain& unique() {
      discrete_.unique();
      continuous_.unique();
      return *this;
    }

    // Set operations
    //==========================================================================

    /**
     * Returns the number of times an argument is present in the domain.
     * This operation has a linear time complexity.
     */
    std::size_t count(Arg arg) const {
      return discrete_.count(arg) + continuous_.count(arg);
    }

    /**
     * Returns the ordered union of two hybrid domains.
     * This operation has a quadratic time complexity.
     */
    friend hybrid_domain
    operator+(const hybrid_domain& a, const hybrid_domain& b) {
      return hybrid_domain(a.discrete() + b.discrete(),
                           a.continuous() + b.continuous());
    }

    /**
     * Returns the ordered difference of two hybrid domains.
     * This operation has a quadratic time complexity.
     */
    friend hybrid_domain
    operator-(const hybrid_domain& a, const hybrid_domain& b) {
      return hybrid_domain(a.discrete() - b.discrete(),
                           a.continuous() - b.continuous());
    }

    /**
     * Returns the ordered intersection of two hybrid domains.
     * This operation has a quadratic time complexity.
     */
    friend hybrid_domain
    operator&(const hybrid_domain& a, const hybrid_domain& b) {
      return hybrid_domain(a.discrete() & b.discrete(),
                           a.continuous() & b.continuous());
    }

    /**
     * Returns true if two domains do not have any arguments in common.
     * This operation has a quadratic time complexity.
     */
    friend bool disjoint(const hybrid_domain& a, const hybrid_domain& b) {
      return disjoint(a.discrete(), b.discrete())
        && disjoint(a.continuous(), b.continuous());
    }

    /**
     * Returns true if two domains contain the same set of arguments
     * (disregarding the order).
     * This operation has a quadratic time complexity.
     */
    friend bool equivalent(const hybrid_domain& a, const hybrid_domain& b) {
      return equivalent(a.discrete(), b.discrete())
        && equivalent(a.continuous(), b.continuous());
    }

    /**
     * Returns true if all the arguments of the first domain are also
     * present in the second domain.
     * This operation has a quadratic time complexity.
     */
    friend bool subset(const hybrid_domain& a, const hybrid_domain& b) {
      return subset(a.discrete(), b.discrete())
        && subset(a.continuous(), b.continuous());
    }

    /**
     * Returns true if all the arguments of the second domain are also
     * present in the first domain.
     * This operation has a quadratic time complexity.
     */
    friend bool superset(const hybrid_domain& a, const hybrid_domain& b) {
      return superset(a.discrete(), b.discrete())
        && superset(a.continuous(), b.continuous());
    }

    // Argument operations
    //==========================================================================

    /**
     * Returns true if two domains are compatible. Two domains are compatible
     * if their corresponding discrete and contiuous components are compatible.
     */
    friend bool compatible(const hybrid_domain& a, const hybrid_domain& b) {
      return compatible(a.discrete(), b.discrete())
        && compatible(a.continuous(), b.continuous());
    }

    /**
     * Returns the overall dimensionality of discrete and continuous arguments
     * combined. For univariate arguments, this is simply the total cardinality
     * of the domain. For multivariate arguments, this is equal to the sum of
     * argument sizes.
     */
    std::size_t num_dimensions() const {
      return discrete_.num_dimensions() + continuous_.num_dimensions();
    }

    /**
     * Returns the number of values for the discrete arguments of this domain.
     */
    std::vector<std::size_t> num_values() const {
      return discrete_.num_values();
    }

    /**
     * Returns the instances (arguments) of a field for one index.
     */
    LIBGM_ENABLE_IF_D(is_indexable<A>::value, typename A = Arg)
    hybrid_domain<instance_type>
    operator()(typename argument_traits<A>::index_type index) const {
      return hybrid_domain<instance_type>(discrete()(index),
                                          continuous()(index));
    }

    /**
     * Returns the instance of an indexable argument for a vector of indices.
     * The instances are returned in the order given by all the instance for
     * the first index first, then all the instances for the second index, etc.
     */
    LIBGM_ENABLE_IF_D(is_indexable<A>::value, typename A = Arg)
    hybrid_domain<instance_type>
    operator()(const std::vector<typename argument_traits<A>::index_type>&
                 indices) const {
      return hybrid_domain<instance_type>(discrete()(indices),
                                          continuous()(indices));
    }

    /**
     * Substitutes arguments in-place according to a map. The keys of the map
     * must include all the arguments in the domain.
     *
     * \throw std::out_of_range if an argument is not present in the map
     * \throw std::invalid_argument if the arguments are not compatible
     */
    template <typename Map>
    void substitute(const Map& map) {
      discrete_.substitute(map);
      continuous_.substitute(map);
    }

    // Indexing operations
    //==========================================================================

    /**
     * Computes the start indexes of this domain in a linear ordering
     * of discrete and continuous arguments.
     *
     * \tparam Map A map object with keys Arg and values std::size_t.
     * \return the number of dimensions of this domain.
     */
    template <typename Map>
    std::pair<std::size_t, std::size_t> insert_start(Map& start) const {
      return { discrete().insert_start(start),
               continuous().insert_start(start) };
    }

    /**
     * Computes the the linear indices of the arguments in this domain
     * given the starting position in the given map.
     *
     * \tparam Map A map object with keys Arg and values std::size_t
     */
    template <typename Map>
    index_type index(const Map& start) const {
      return { discrete().index(start), continuous().index(start) };
    }

  private:
    //! The discrete component.
    domain<Arg> discrete_;

    //! The continuous component.
    domain<Arg> continuous_;

  }; // struct hybrid_domain

  /**
   * Converts one domain to a domain with another argument type.
   *
   * \tparam Target
   *         The target argument type. Must be convertible from Source using
   *         argument_cast.
   * \tparam Source
   *         The original argument type.
   * \relates hybrid_domain
   */
  template <typename Target, typename Source>
  hybrid_domain<Target> argument_cast(const hybrid_domain<Source>& dom) {
    static_assert(is_convertible_argument<Source, Target>::value,
                  "Source must be argument-convertible to Target");

    return hybrid_domain<Target>(argument_cast<Target>(dom.discrete()),
                                 argument_cast<Target>(dom.continuous()));
  }

} // namespace libgm

#endif
