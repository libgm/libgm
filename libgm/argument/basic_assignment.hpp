#ifndef LIBGM_BASIC_ASSIGNMENT_HPP
#define LIBGM_BASIC_ASSIGNMENT_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/functional/utility.hpp>
#include <libgm/parser/range_io.hpp>
#include <libgm/traits/vector_value.hpp>

#include <algorithm>
#include <unordered_map>

namespace libgm {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment arguments of a single type.
   * basic_assignment is guaranteed to be an UnorderedAssociativeContainer
   * (presently std::unordered_map), with key_type being Arg, and
   * mapped_type determined by the arity of Arg (as specified by its
   * argument_traits). If Arg is univariate, then arguments are mapped
   * to a scalar, and this map is effectively std::unordered_map<Arg, Scalar>.
   * If Arg is multivariate, then arguments are mapped to a vector,
   * and this map is effectively std::unordered_map<Arg, Vector>.
   * The hasher is taken from Arg's argument_traits.
   *
   * \tparam Arg A type that models the Argument concept.
   * \tparam Vector A type that represents a vector of values
   * \tparam Arity the arity of Arg, as specified by its argument_traits
   */
  template <typename Arg,
            typename Vector,
            typename Arity = typename argument_traits<Arg>::argument_arity>
  class basic_assignment;

  /**
   * Specialization of basic_assignment for univariate arguments.
   */
  template <typename Arg, typename Vector>
  class basic_assignment<Arg, Vector, univariate_tag>
    : public std::unordered_map<Arg,
                                typename vector_value<Vector>::type,
                                typename argument_traits<Arg>::hasher> {

    typedef std::unordered_map<Arg,
                               typename vector_value<Vector>::type,
                               typename argument_traits<Arg>::hasher> base;
  public:
    //! The type of values stored in the map.
    typedef typename base::value_type value_type;

    // Bring the overloads from the base class
    using base::insert;

    /**
     * Constructs an empty assignment.
     * \param nbuckets the minimum number buckets to create (if 0, use the
     *        STL implementation-defined default value).
     */
    explicit basic_assignment(std::size_t nbuckets = 0) {
      if (nbuckets) { this->reserve(nbuckets); }
    };

    /**
     * Constructs an assignment with the contents of the range [first; last).
     * \param nbuckets the minimum number buckets to create (if 0, use the
     *        STL implementation-defined default value).
     */
    template <typename InputIt>
    basic_assignment(InputIt first, InputIt last, std::size_t nbuckets = 0) {
      if (nbuckets) { this->reserve(nbuckets); }
      this->insert(first, last);
    }

    /**
     * Constructs an assignment with the contents of the given list.
     * \param nbuckets the minimum number buckets to create (if 0, use the
     *        STL implementation-defined default value).
     */
    basic_assignment(std::initializer_list<value_type> init,
                     std::size_t nbuckets = 0) {
      if (nbuckets) { this->reserve(nbuckets); }
      this->insert(init);
    }

    /**
     * Constructs an assignment with keys drawn from a domain and the
     * corresponding values drawn from a dense vector.
     */
    basic_assignment(const domain<Arg>& args, const Vector& vals) {
      this->reserve(args.size());
      insert(args, vals);
    }

    /**
     * Returns the values in this assignment for a subset of arguments
     * in the order specified by the given domain.
     */
    Vector values(const domain<Arg>& args, std::size_t start = 0) const {
      assert(start <= args.size());
      Vector result(args.size() - start);
      for (std::size_t i = start; i < args.size(); ++i) {
        result[i - start] = this->at(args[i]);
      }
      return result;
    }

    /**
     * Inserts the keys drawn from a domain and the corresponding values
     * drawn from a dense vector. If a key already exists, its original
     * value is preserved, similarly to std::unordered_map::insert.
     *
     * \return the number of values inserted
     */
    std::size_t insert(const domain<Arg>& args, const Vector& values) {
      assert(args.size() == values.size());
      std::size_t ninserted = 0;
      for (std::size_t i = 0; i < args.size(); ++i) {
        ninserted += this->emplace(args[i], values[i]).second;
      }
      return ninserted;
    }

    /**
     * Inserts the keys drawn from a domain and the corresponding values
     * drawn from a dense vector. If a key already exists, its value is
     * overwritten, similarly to std::unordered_map::insert_or_assign.
     *
     * \return the number of values inserted
     */
    std::size_t insert_or_assign(const domain<Arg>& args, const Vector& values) {
      assert(args.size() == values.size());
      std::size_t ninserted = 0;
      for (std::size_t i = 0; i < args.size(); ++i) {
        ninserted += 1 - this->count(args[i]);
        (*this)[args[i]] = values[i];
      }
      return ninserted;
    }

    /**
     * Inserts the keys drawn from a domain and the corresponding values
     * stored as a linear index. If a key already exists, its value is
     * overwritten, similarly to std::unordered_map::insert_or_assign.
     * This function is only supported for discrete arguments.
     *
     * \return the number of values inserted
     */
    template <bool B = is_discrete<Arg>::value, typename = std::enable_if_t<B> >
    std::size_t insert_or_assign(const domain<Arg>& args, std::size_t index) {
      std::size_t ninserted = 0;
      for (std::size_t i = 0; i < args.size(); ++i) {
        ninserted += 1 - this->count(args[i]);
        std::size_t cardinality = argument_traits<Arg>::num_values(args[i]);
        (*this)[args[i]] = index % cardinality;
        index /= cardinality;
      }
      return ninserted;
    }

    /**
     * Returns true if all the arguments in the given domain are present in
     * the given assignment.
     */
    friend bool subset(const domain<Arg>& args, const basic_assignment& a) {
      return std::all_of(args.begin(), args.end(), count_in(a));
    }

    /**
     * Returns true if none of the arguments in the given domain are present in
     * the given assignment.
     */
    friend bool disjoint(const domain<Arg>& args, const basic_assignment& a) {
      return std::none_of(args.begin(), args.end(), count_in(a));
    }

    /**
     * Returns the linear index corresponding to this assignment in a table
     * with the specified arguments. If strict is true, each argument* must be
     * present in this assignment. If strict is false, the missing arguments
     * are assumed to be 0.
     *
     * Only supported when Arg is discrete.
     */
    LIBGM_ENABLE_IF(is_discrete<Arg>::value)
    std::size_t linear_index(const domain<Arg>& args, bool strict = true) const{
      std::size_t result = 0;
      std::size_t multiplier = 1;
      for (Arg arg : args) {
        auto it = this->find(arg);
        if (it != this->end()) {
          result += multiplier * it->second;
        } else if (strict) {
          std::ostringstream out;
          out << "basic_assignment::linear_index: missing argument ";
          argument_traits<Arg>::print(out, arg);
          throw std::invalid_argument(out.str());
        }
        multiplier *= argument_traits<Arg>::num_values(arg);
      }
      return result;
    }

    /**
     * Prints a human-readable representaiton of the assignment to an
     * output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const basic_assignment& a) {
      out << '{';
      bool first = true;
      for (const auto& p : a) {
        if (first) { first = false; } else { out << ','; }
        argument_traits<Arg>::print(out, p.first);
        out << ':' << p.second;
      }
      out << '}';
      return out;
    }

  }; // class basic_assignment<Arg, Vector, univariate_tag>


  /**
   * Specialization of basic_assignment for multivariate arguments.
   */
  template <typename Arg, typename Vector>
  class basic_assignment<Arg, Vector, multivariate_tag>
    : public std::unordered_map<Arg,
                                Vector,
                                typename argument_traits<Arg>::hasher> {

      typedef std::unordered_map<Arg,
                                 Vector,
                                 typename argument_traits<Arg>::hasher> base;

  public:
    //! The type of values stored in the map.
    typedef typename base::value_type value_type;

    // Bring the overloads from the base class
    using base::insert;

    /**
     * Constructs an empty assignment.
     * \param nbuckets the minimum number buckets to create (if 0, use the
     *        STL implementation-defined default value).
     */
    explicit basic_assignment(std::size_t nbuckets = 0) {
      if (nbuckets) { this->reserve(nbuckets); }
    };

    /**
     * Constructs an assignment with the contents of the range [first; last).
     * \param nbuckets the minimum number buckets to create (if 0, use the
     *        STL implementation-defined default value).
     */
    template <typename InputIt>
    basic_assignment(InputIt first, InputIt last, std::size_t nbuckets = 0) {
      if (nbuckets) { this->reserve(nbuckets); }
      this->insert(first, last);
    }

    /**
     * Constructs an assignment with the contents of the given list.
     * \param nbuckets the minimum number buckets to create (if 0, use the
     *        STL implementation-defined default value).
     */
    basic_assignment(std::initializer_list<value_type> init,
                    std::size_t nbuckets = 0) {
      if (nbuckets) { this->reserve(nbuckets); }
      this->insert(init);
    }

    /**
     * Constructs an assignment with keys drawn from a domain and the
     * corresponding values drawn from a dense vector.
     */
    basic_assignment(const domain<Arg>& args, const Vector& vals) {
      this->reserve(args.size());
      insert(args, vals);
    }

    /**
     * Returns the values in this assignment for a subset of arguments
     * in the order specified by the given domain.
     */
    Vector values(const domain<Arg>& args, std::size_t start = 0) const {
      assert(start <= args.size());
      Vector result(args.num_dimensions(start));
      auto dest = result.data();
      for (std::size_t i = start; i < args.size(); ++i) {
        const Vector& vals = this->at(args[i]);
        assert(vals.size() == argument_traits<Arg>::num_dimensions(args[i]));
        dest = std::copy(vals.data(), vals.data() + vals.size(), dest);
      }
      return result;
    }

    /**
     * Inserts the keys drawn from a domain and the corresponding values
     * concatenated in a dense vector. If a key already exists, its original
     * value is preserved, similarly to std::unordered_map::insert.
     *
     * \return the number of values inserted
     */
    std::size_t insert(const domain<Arg>& args, const Vector& values) {
      assert(args.num_dimensions() == values.size());
      std::size_t ninserted = 0;
      auto src = values.data();
      for (Arg arg : args) {
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        auto result = this->emplace(arg, Vector());
        if (result.second) { // insertion took place
          ++ninserted;
          result.first->second.resize(n);
          std::copy(src, src + n, result.first->second.data());
        }
        src += n;
      }
      return ninserted;
    }

    /**
     * Inserts the keys drawn from a domain and the corresponding values
     * concatenated in a dense vector. If a key already exists, its value is
     * overwritten, similarly to std::unordereed_map::insert_or_assign.
     *
     * \return the number of values inserted
     */
    std::size_t insert_or_assign(const domain<Arg>& args, const Vector& values) {
      assert(args.num_dimensions() == values.size());
      std::size_t ninserted = 0;
      auto src = values.data();
      for (Arg arg : args) {
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        auto result = this->emplace(arg, Vector());
        ninserted += result.second;
        result.first->second.resize(n);
        std::copy(src, src + n, result.first->second.data());
        src += n;
      }
      return ninserted;
    }

    /**
     * Inserts the keys drawn from a domain and the corresponding values
     * stored as a linear index. If a key already exists, its value is
     * overwritten, similarly to std::unordered_map::insert_or_assign.
     * This function is only supported for discrete arguments.
     *
     * \return the number of values inserted
     */
    template <bool B = is_discrete<Arg>::value, typename = std::enable_if_t<B> >
    std::size_t insert_or_assign(const domain<Arg>& args, std::size_t index) {
      std::size_t ninserted = 0;
      for (Arg arg : args) {
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        auto result = this->emplace(arg, Vector());
        ninserted += result.second;
        result.first->second.resize(n);
        for (std::size_t i =0 ; i < n; ++i) {
          std::size_t cardinality = argument_traits<Arg>::num_values(arg, i);
          result.first->second[i] = index % cardinality;
          index /= cardinality;
        }
      }
      return ninserted;
    }

    /**
     * Returns true if all the arguments in the given domain are present in
     * the given assignment.
     */
    friend bool subset(const domain<Arg>& args, const basic_assignment& a) {
      return std::all_of(args.begin(), args.end(), count_in(a));
    }

    /**
     * Returns true if none of the arguments in the given domain are present in
     * the given assignment.
     */
    friend bool disjoint(const domain<Arg>& args, const basic_assignment& a) {
      return std::none_of(args.begin(), args.end(), count_in(a));
    }

    /**
     * Returns the linear index corresponding to this assignment in a table
     * with the specified arguments. If strict is true, each argument* must be
     * present in this assignment. If strict is false, the missing arguments
     * are assumed to be 0.
     *
     * Only supported when Arg is discrete.
     */
    LIBGM_ENABLE_IF(is_discrete<Arg>::value)
    std::size_t linear_index(const domain<Arg>& args, bool strict = true) const{
      std::size_t result = 0;
      std::size_t multiplier = 1;
      for (Arg arg : args) {
        auto it = this->find(arg);
        if (it != this->end()) {
          std::size_t n = argument_traits<Arg>::num_dimensions(arg);
          assert(it->second.size() == n);
          for (std::size_t pos = 0; pos < n; ++pos) {
            result += multiplier * it->second[pos];
            multiplier *= argument_traits<Arg>::num_values(arg, pos);
          }
        } else if (strict) {
          std::ostringstream out;
          out << "basic_assignment::linear_index: missing argument ";
          argument_traits<Arg>::print(out, arg);
          throw std::invalid_argument(out.str());
        } else {
          multiplier *= argument_traits<Arg>::num_values(arg);
        }
      }
      return result;
    }

    /**
     * Prints a human-readable representation of the assignment to
     * an output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const basic_assignment& a) {
      out << '{';
      bool first = true;
      for (const auto& p : a) {
        if (first) { first = false; } else { out << ','; }
        argument_traits<Arg>::print(out, p.first);
        out << ':';
        print_range(out, p.second.data(), p.second.data() + p.second.size(),
                    '[', ' ', ']');
      }
      out << '}';
      return out;
    }

  }; // class basic_assignment<Arg, Vector, multivariate_tag>

} // namespace libgm

#endif
