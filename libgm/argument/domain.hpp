#ifndef LIBGM_DOMAIN_HPP
#define LIBGM_DOMAIN_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/argument_cast.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/functional/utility.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/vector.hpp>
#include <libgm/traits/missing.hpp>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace libgm {

  // Forward declarations
  template <typename Arg> class domain;

  namespace detail {

    //! Implements domain::sizes for univariate argument types.
    template <typename Arg>
    inline uint_vector
    argument_sizes(const domain<Arg>& dom, univariate_tag) {
      uint_vector result(dom.size());
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = argument_size(dom[i]);
      }
      return result;
    }

    //! Implements domain::sizes for multivariate argument types.
    template <typename Arg>
    inline uint_vector
    argument_sizes(const domain<Arg>& dom, multivariate_tag) {
      uint_vector result(dom.arity());
      auto dest = result.begin();
      for (Arg arg : dom) {
        std::size_t n = argument_arity(arg);
        for (std::size_t pos = 0; pos < n; ++pos) {
          *dest++ = argument_traits<Arg>::size(arg, pos);
        }
      }
      return result;
    }

    //! Implements domain::index for univariate argument types.
    template <typename Arg>
    inline uint_vector
    index(const domain<Arg>& dom, const domain<Arg>& args, univariate_tag) {
      uint_vector index(args.size());
      for(std::size_t i = 0; i < index.size(); i++) {
        auto it = std::find(dom.begin(), dom.end(), args[i]);
        if (it != dom.end()) {
          index[i] = it - dom.begin();
        } else {
          std::ostringstream out;
          print_argument(out << "domain::index cannot find argument ", args[i]);
          throw std::invalid_argument(out.str());
        }
      }
      return index;
    }

    //! Implements domain::index for multivariate argument types.
    template <typename Arg>
    inline uint_vector
    index(const domain<Arg>& dom, const domain<Arg>& args, multivariate_tag) {
      // compute the first dimension of each argument in dom
      uint_vector dim(dom.size());
      for (std::size_t i = 1; i < dom.size(); ++i) {
        dim[i] = dim[i-1] + argument_arity(dom[i-1]);
      }

      // extract the dimensions for the arguments
      uint_vector index(args.arity());
      auto dest = index.begin();
      for (Arg arg : arg) {
        auto it = std::find(dom.begin(), dom.end(), arg);
        if (it != dom.end()) {
          std::iota(dest, dest + argument_arity(arg), dim[it - dom.begin()]);
        } else {
          std::ostringstream out;
          print_argument(cout << "domain::index cannot find argument ", arg);
          throw std::invalid_argument(out.str());
        }
        dest += argument_arity(arg);
      }
      return index;
    }

  } // namespace detail

  /**
   * A domain that holds the arguments in an std::vector.
   *
   * \tparam Arg a type that satisfies the Argument concept
   */
  template <typename Arg>
  class domain : public std::vector<Arg> {
  public:

    // Domain concept
    typedef Arg key_type;

    // Helper types
    typedef typename argument_traits<Arg>::instance_type instance_type;

    using std::vector<Arg>::begin;
    using std::vector<Arg>::end;

    //! Default constructor. Creates an empty domain.
    domain() { }

    //! Constructs a domain with given number of empty arguments.
    explicit domain(std::size_t n)
      : std::vector<Arg>(n) { }

    //! Creates a domain with the given arguments.
    domain(std::initializer_list<Arg> init)
      : std::vector<Arg>(init) { }

    //! Creates a domain from the given iterator range.
    template <typename Iterator>
    domain(Iterator begin, Iterator end)
      : std::vector<Arg>(begin, end) { }

    //! Creates a domain from the given iterator range.
    template <typename Iterator>
    domain(const iterator_range<Iterator>& range)
      : std::vector<Arg>(range.begin(), range.end()) { }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      ar.serialize_range(begin(), end());
    }

    //! Loads the domain from an archive.
    void load(iarchive& ar) {
      this->clear();
      ar.deserialize_range<Arg>(std::back_inserter(*this));
    }

    //! Returns the hash value of a domain.
    friend std::size_t hash_value(const domain& dom) {
      return hash_range(dom.begin(), dom.end());
    }

    //! Prints the domain to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const domain& dom) {
      out << '[';
      for (std::size_t i = 0; i < dom.size(); ++i) {
        if (i > 0) { out << ','; }
        print_argument(out, dom[i]);
      }
      out << ']';
      return out;
    }

    // Sequence operations
    //--------------------------------------------------------------------------

    //! Returns a prefix of this domain.
    domain prefix(std::size_t n) const {
      assert(n <= this->size());
      return domain(begin(), begin() + n);
    }

    //! Returns a suffix of this domain.
    domain suffix(std::size_t n) const {
      assert(n <= this->size());
      return domain(end() - n, end());
    }

    //! Returns true if the given domain is a prefix of this domain.
    bool prefix(const domain& dom) const {
      return dom.size() <= this->size()
        && std::equal(dom.begin(), dom.end(), begin());
    }

    //! Returns true if the given domain is a suffix of this domain.
    bool suffix(const domain& dom) const {
      return dom.size() <= this->size()
        && std::equal(dom.begin(), dom.end(), end() - dom.size());
    }

    /**
     * Removes the duplicate arguments from the domain in place.
     * Does not preserve the relative order of arguments in the domain.
     */
    domain& unique() {
      std::sort(begin(), end());
      auto new_end = std::unique(begin(), end());
      this->erase(new_end, end());
      return *this;
    }

    /**
     * Returns the concatenation of two domains.
     * This operation has a linear time complexity, O(|a| + |b|).
     */
    friend domain concat(const domain& a, const domain& b) {
      domain result;
      result.reserve(a.size() + b.size());
      std::copy(a.begin(), a.end(), std::back_inserter(result));
      std::copy(b.begin(), b.end(), std::back_inserter(result));
      return result;
    }

    // Set operations
    //--------------------------------------------------------------------------

    /**
     * Returns the number of times an argument is present in the domain.
     * This operation has a linear time complexity.
     */
    std::size_t count(Arg x) const {
      return std::count(begin(), end(), x);
    }

    /**
     * Partitions this domain into those arguments that are present in the
     * given associative container (set or map) and those that are not.
     */
    template <typename Set>
    void partition(const Set& set, domain& present, domain& absent) const {
      for (Arg arg : *this) {
        if (set.count(arg)) {
          present.push_back(arg);
        } else {
          absent.push_back(arg);
        }
      }
    }

    /**
     * Returns the ordered union of two domains.
     * This operation has a quadratic time complexity, O(|a| * |b|).
     */
    friend domain operator+(const domain& a, const domain& b) {
      domain result = a;
      std::remove_copy_if(b.begin(), b.end(), std::back_inserter(result),
                          count_in(a));
      return result;
    }

    /**
     * Returns the ordered difference of two domains.
     * This operation has a quadratic time complexity, O(|a| * |b|).
     */
    friend domain operator-(const domain& a, const domain& b) {
      domain result;
      std::remove_copy_if(a.begin(), a.end(), std::back_inserter(result),
                          count_in(b));
      return result;
    }

    /**
     * Returns the ordered intersection of two domains.
     * This operation has a quadratic time complexity, O(|a| * |b|).
     */
    friend domain operator&(const domain& a, const domain& b) {
      domain result;
      std::copy_if(a.begin(), a.end(), std::back_inserter(result), count_in(b));
      return result;
    }

    /**
     * Returns true if two domains do not have any arguments in common.
     * This operation has a quadratic time complexity, O(|a| * |b|).
     */
    friend bool disjoint(const domain& a, const domain& b) {
      return std::none_of(a.begin(), a.end(), count_in(b));
    }

    /**
     * Returns true if two domains contain the same set of arguments
     * (disregarding the order).
     * This operation has a quadratic time complexity, O(|a| * |b|).
     */
    friend bool equivalent(const domain& a, const domain& b) {
      return a.size() == b.size()
        && std::all_of(a.begin(), a.end(), count_in(b));
    }

    /**
     * Returns true if all the arguments of the first domain are also
     * present in the second domain.
     * This operation has a quadratic time complexity, O(|a| * |b|).
     */
    friend bool subset(const domain& a, const domain& b) {
      return a.size() <= b.size()
        && std::all_of(a.begin(), a.end(), count_in(b));
    }

    /**
     * Returns true if all the arguments of the second domain are also
     * present in the first domain.
     * This operation has a quadratic time complexity, O(|a| * |b|).
     */
    friend bool superset(const domain& a, const domain& b) {
      return subset(b, a);
    }

    // Argument operations
    //--------------------------------------------------------------------------

    /**
     * Returns the overall dimensionality for a collection of arguments.
     * This is simply the cardinality of the domain for univariate arguments
     * and the the sum of argument dimensionalities for multivariate arguments.
     */
    std::size_t arity(std::size_t start = 0) const {
      assert(start <= this->size());
      if (is_univariate<Arg>::value) {
        return this->size() - start;
      } else {
        std::size_t size = 0;
        for (std::size_t i = start; i < this->size(); ++i) {
          size += argument_arity((*this)[i]);
        }
        return size;
      }
    }

    /**
     * Returns the total arity of the discrete and continuous arguments
     * in this domain.
     */
    std::pair<std::size_t, std::size_t> mixed_arity() const {
      std::size_t m = 0, n = 0;
      for (Arg arg : *this) {
        if (argument_discrete(arg)) {
          m += argument_arity(arg);
        } else if (argument_continous(arg)) {
          n += argument_arity(arg);
        } else {
          // throw
        }
      }
      return std::make_pair(m, n);
    }

    /**
     * Returns the vector specifying the number of values for a collection of
     * discrete arguments. The result first contains the number of values for
     * the first argument in this domain, then the number of values for the
     * second argument, etc. The resulting vector is guaranteed to have exactly
     * arity() elements.
     *
     * This function is supported only when Arg is discrete.
     */
    LIBGM_ENABLE_IF(is_discrete<Arg>::value)
    uint_vector sizes() const {
      return detail::argument_sizes(*this, argument_arity_t<Arg>());
    }

    /**
     * Returns true if all the arguments in the domain are discrete.
     */
    bool discrete() const {
      return is_discrete<Arg>::value ||
        std::all_of(begin(), end(), [](Arg arg) {
            return argument_discrete(arg); });
    }

    /**
     * Returns true if all the arguments in the domain are continuous.
     */
    bool continuous() const {
      return is_continuous<Arg>::value ||
        std::all_of(begin(), end(), [](Arg arg) {
            return argument_continuous(arg); });
    }

    /**
     * Computes the indices of the specified arguments in this domain.
     * For univariate arguments, this function returns an index vector
     * v s.t. v[i] is the index of args[i] in this domain.
     */
    uint_vector index(const domain& args) const {
      return detail::index(*this, args, argument_arity_t<Arg>());
    }

    /**
     * Computes the start indexes of this domain in a linear ordering
     * of arguments.
     *
     * \tparam Map A map object with keys Arg and values std::size_t.
     * \return the number of dimensions of this domain.
     */
    template <typename Map>
    std::size_t insert_start(Map& start) const {
      std::size_t pos = 0;
      for (Arg arg : *this) {
        start.emplace(arg, pos);
        pos += argument_arity(arg);
      }
      return pos;
    }

    //????
    /**
     * Returns the instances of a field for one index.???
     */
    LIBGM_ENABLE_IF_D(is_same<Arg, field<indexable<Arg>::value, typename A = Arg)
    domain<instance_type>
    operator()(typename argument_traits<A>::index_type index) const {
      domain<instance_type> result;
      result.reserve(this->size());
      for (Arg arg : *this) {
        result.push_back(arg(index));
      }
      return result;
    }

  }; // class domain

} // namespace libgm

namespace std {

  template <typename Arg>
  struct hash<libgm::domain<Arg>>
    : libgm::default_hash<libgm::domain<Arg>> { };

} // namespace std

#endif
