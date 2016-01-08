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
  template <typename Arg> class unary_domain;
  template <typename Arg> class binary_domain;

  namespace detail {

    //! Implements domain::num_values for univariate argument types.
    template <typename Arg>
    inline uint_vector num_values(const domain<Arg>& dom, univariate_tag) {
      uint_vector result(dom.size());
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = argument_traits<Arg>::num_values(dom[i]);
      }
      return result;
    }

    //! Implements domain::num_values for multivariate argument types.
    template <typename Arg>
    inline uint_vector num_values(const domain<Arg>& dom, multivariate_tag) {
      uint_vector result(dom.num_dimensions());
      auto dest = result.begin();
      for (Arg arg : dom) {
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        for (std::size_t pos = 0; pos < n; ++pos) {
          *dest++ = argument_traits<Arg>::num_values(arg, pos);
        }
      }
      return result;
    }

    //! Implements domain::index for univariate argument types.
    template <typename Arg>
    uint_vector index(const domain<Arg>& dom, const domain<Arg>& args,
                      bool strict,
                      univariate_tag) {
      uint_vector index(dom.size(), missing<std::size_t>::value);
      for(std::size_t i = 0; i < index.size(); i++) {
        auto it = std::find(args.begin(), args.end(), dom[i]);
        if (it != args.end()) {
          index[i] = it - args.begin();
        } else if (strict) {
          std::ostringstream out;
          out << "domain::index: cannot find argument ";
          argument_traits<Arg>::print(out, dom[i]);
          throw std::invalid_argument(out.str());
        }
      }
      return index;
    }

    //! Implements domain::index for multivariate argument types.
    template <typename Arg>
    uint_vector index(const domain<Arg>& dom, const domain<Arg>& args,
                      bool strict,
                      multivariate_tag) {
      // compute the first dimension of each argument in args
      uint_vector dim(args.size());
      for (std::size_t i = 1; i < args.size(); ++i) {
        dim[i] = dim[i-1] + argument_traits<Arg>::num_dimensions(args[i-1]);
      }

      // extract the dimensions for the arguments in this factor
      uint_vector index(dom.num_dimensions(), missing<std::size_t>::value);
      auto dest = index.begin();
      for (Arg arg : dom) {
        auto it = std::find(args.begin(), args.end(), arg);
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        if (it != args.end()) {
          std::iota(dest, dest + n, dim[it - args.begin()]);
        } else if (strict) {
          std::ostringstream out;
          out << "domain::index: cannot find argument ";
          argument_traits<Arg>::print(out, arg);
          throw std::invalid_argument(out.str());
        }
        dest += n;
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
    typedef uint_vector index_type;

    // Helper types
    typedef typename argument_traits<Arg>::instance_type instance_type;
    typedef typename argument_traits<Arg>::argument_arity argument_arity;

    //! Default constructor. Creates an empty domain.
    domain() { }

    //! Constructs a domain with given number of empty arguments.
    explicit domain(std::size_t n)
      : std::vector<Arg>(n) { }

    //! Creates a domain with the given arguments.
    domain(std::initializer_list<Arg> init)
      : std::vector<Arg>(init) { }

    //! Creates a domain from the given argument vector.
    domain(const std::vector<Arg>& elems)
      : std::vector<Arg>(elems) { }

    //! Creates a domain from the given argument array.
    template <std::size_t N>
    domain(const std::array<Arg, N>& elems)
      : std::vector<Arg>(elems.begin(), elems.end()) { }

    //! Creates a domain from the given iterator range.
    template <typename Iterator>
    domain(Iterator begin, Iterator end)
      : std::vector<Arg>(begin, end) { }

    //! Creates a domain from the given iterator range.
    template <typename Iterator>
    explicit domain(const iterator_range<Iterator>& range)
      : std::vector<Arg>(range.begin(), range.end()) { }

    //! Conversion from a unary domain.
    domain(const unary_domain<Arg>& dom)
      : std::vector<Arg>({dom.x()}) { }

    //! Conversion from a binary domain.
    domain(const binary_domain<Arg>& dom)
      : std::vector<Arg>({dom.x(), dom.y()}) { }

    //! Conversion to a unary domain. Throws std::invalid_argument if not unary.
    unary_domain<Arg> unary() const {
      if (this->size() != 1) {
        throw std::invalid_argument("The domain is not unary");
      }
      return { this->front() };
    }

    //! Conversion to a binary domain. Throws std::invalid_argument if not binary.
    binary_domain<Arg> binary() const {
      if (this->size() != 2) {
        throw std::invalid_argument("The domain is not binary");
      }
      return { this->front(), this->back() };
    }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      ar.serialize_range(this->begin(), this->end());
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
        argument_traits<Arg>::print(out, dom[i]);
      }
      out << ']';
      return out;
    }

    // Sequence operations
    //==========================================================================

    //! Returns a prefix of this domain.
    domain prefix(std::size_t n) const {
      assert(n <= this->size());
      return domain(this->begin(), this->begin() + n);
    }

    //! Returns a suffix of this domain.
    domain suffix(std::size_t n) const {
      assert(n <= this->size());
      return domain(this->end() - n, this->end());
    }

    //! Returns true if the given domain is a prefix of this domain.
    bool prefix(const domain& dom) const {
      return dom.size() <= this->size()
        && std::equal(dom.begin(), dom.end(), this->begin());
    }

    //! Returns true if the given domain is a suffix of this domain.
    bool suffix(const domain& dom) const {
      return dom.size() <= this->size()
        && std::equal(dom.begin(), dom.end(), this->end() - dom.size());
    }

    /**
     * Removes the duplicate arguments from the domain in place.
     * Does not preserve the relative order of arguments in the domain.
     */
    domain& unique() {
      std::sort(this->begin(), this->end());
      auto new_end = std::unique(this->begin(), this->end());
      this->erase(new_end, this->end());
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
    //==========================================================================

    /**
     * Returns the number of times an argument is present in the domain.
     * This operation has a linear time complexity.
     */
    std::size_t count(Arg x) const {
      return std::count(this->begin(), this->end(), x);
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
    //==========================================================================

    /**
     * Returns true if two domains are compatible. Two domains are compatible
     * if they have the same cardinality and the corresponding arguments are
     * compatible as specified by argument_traits<Arg>.
     */
    friend bool compatible(const domain& a, const domain& b) {
      if (a.size() != b.size()) {
        return false;
      }
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (!argument_traits<Arg>::compatible(a[i], b[i])) {
          return false;
        }
      }
      return true;
    }

    /**
     * Returns the overall dimensionality for a collection of arguments.
     * This is simply the cardinality of the domain for univariate arguments
     * and the the sum of argument dimensionalities for multivariate arguments.
     */
    std::size_t num_dimensions(std::size_t start = 0) const {
      assert(start <= this->size());
      if (is_univariate<Arg>::value) {
        return this->size() - start;
      } else {
        std::size_t size = 0;
        for (std::size_t i = start; i < this->size(); ++i) {
          size += argument_traits<Arg>::num_dimensions((*this)[i]);
        }
        return size;
      }
    }

    /**
     * Returns the vector specifying the number of values for a collection of
     * discrete arguments. The result first contains the number of values for
     * the first argument in this domain, then the numebr of values for the
     * second argument, etc. The resulting vector is guaranteed to have exactly
     * num_dimensions() elements.
     *
     * This function is supported only when Arg is discrete.
     */
    LIBGM_ENABLE_IF(is_discrete<Arg>::value)
    index_type num_values() const {
      return detail::num_values(*this, argument_arity());
    }

    /**
     * Returns the instances of an indexable argument for one index.
     */
    LIBGM_ENABLE_IF_D(is_indexable<A>::value, typename A = Arg)
    domain<instance_type>
    operator()(typename argument_traits<A>::index_type index) const {
      domain<instance_type> result;
      result.reserve(this->size());
      for (Arg arg : *this) {
        result.push_back(arg(index));
      }
      return result;
    }

    /**
     * Returns the instance of an indexable argument for a vector of indices.
     * The instances are returned in the order given by all the instance for
     * the first index first, then all the instances for the second index, etc.
     */
    LIBGM_ENABLE_IF_D(is_indexable<A>::value, typename A = Arg)
    domain<instance_type>
    operator()(const std::vector<typename argument_traits<A>::index_type>&
                 indices) const {
      domain<instance_type> result;
      result.reserve(this->size() * indices.size());
      for (auto index : indices) {
        for (Arg arg : *this) {
          result.push_back(arg(index));
        }
      }
      return result;
    }

    /**
     * Substitutes arguments in-place according to a map. The keys of the map
     * must include all the arguments in this domain.
     *
     * \throw std::out_of_range if an argument is not present in the map
     * \throw std::invalid_argument if the arguments are not compatible
     */
    template <typename Map>
    void substitute(const Map& map) {
      for (Arg& arg : *this) {
        Arg new_arg = map.at(arg);
        if (!argument_traits<Arg>::compatible(arg, new_arg)) {
          std::ostringstream out;
          out << "Incompatible arguments ";
          argument_traits<Arg>::print(out, arg);
          out << " and ";
          argument_traits<Arg>::print(out, new_arg);
          throw std::invalid_argument(out.str());
        }
        arg = new_arg;
      }
    }

    // Indexing operations
    //==========================================================================

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
        pos += argument_traits<Arg>::num_dimensions(arg);
      }
      return pos;
    }

    /**
     * Computes the the indices of the arguments in this domain
     * given the starting position in the given map.
     *
     * \tparam Map A map object with keys Arg and values std::size_t
     */
    template <typename Map>
    index_type index(const Map& start) const {
      index_type result(num_dimensions());
      auto dest = result.begin();
      for (Arg arg : *this) {
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        try {
          std::iota(dest, dest + n, start.at(arg));
        } catch(std::out_of_range&) {
          std::ostringstream out;
          out << "domain::index: cannnot find argument ";
          argument_traits<Arg>::print(out, arg);
          throw std::invalid_argument(out.str());
        }
        dest += n;
      }
      return result;
    }

    /**
     * Computes the indices of the arguments in this domain in the specified
     * domain. More precisely, it returns an index vector v s.t. v[i] is the
     * index of (*this)[i] in args. If strict is true, all arguments of this
     * domain must be present in the specified domain. If strict is false,
     * the missing arguments will be assigned a missing<std::size_t> value.
     *
     * When using this function in table factor operations, always call the
     * index() on the domain of the factor whose elements will be iterated
     * over in a non-linear fashion. The specified args are the arguments
     * of the table that is iterated over in a linear fashion.
     */
    index_type index(const domain& args, bool strict = true) const {
      return detail::index(*this, args, strict, argument_arity());
    }

  }; // class domain

  /**
   * Converts one domain to a domain with another argument type.
   *
   * \tparam Target
   *         The target argument type. Must be convertible from Source using
   *         argument_cast.
   * \tparam Source
   *         The original argument type.
   * \relates domain
   */
  template <typename Target, typename Source>
  domain<Target> argument_cast(const domain<Source>& dom) {
    static_assert(is_convertible_argument<Source, Target>::value,
                  "Source must be argument-convertible to Target");

    domain<Target> result;
    result.reserve(dom.size());
    for (Source arg : dom) {
      result.push_back(argument_cast<Target>(arg));
    }
    return result;
  }

} // namespace libgm

namespace std {

  template <typename Arg>
  struct hash<libgm::domain<Arg>>
    : libgm::default_hash<libgm::domain<Arg>> { };

} // namespace std

#endif
