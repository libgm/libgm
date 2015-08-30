#ifndef LIBGM_ARGUMENT_CONCEPTS_HPP
#define LIBGM_ARGUMENT_CONCEPTS_HPP

#include <cstddef>

namespace libgm {

  /**
   * A concept that represents an argument.
   *
   * An argument must be CopyAssignable, CopyConstructible, Destructible,
   * EqualityComparable, LessThanComparable, and Swappable.
   *
   * \ingroup argument_concepts
   */
  struct Argument {

    /**
     * Returns true if two arguments are compatible. Two arguments are
     * compatible if one can be substituted for another in the model,
     * while keeping the model well-defined. Typically, this means that
     * the number of values or the dimensionality of the arguments must
     * be the same, but Arg can place additional constraints that are
     * meaningful in the application.
     *
     * Compatibility must satisfy two properties:
     * 1) symmetry: if (x, y) are compatible, so are (y, x);
     * 2) transitivity: if (x, y) are compatible, and (y, z) are
     *    compatible, so are (x, z).
     *
     * This function is accessed via the argument_traits class.
     */
    static bool compatible(Argument x, Argument y);

    /**
     * Prints the human-readable representation of the argument to an
     * output stream.
     *
     * This function is accessed via the argument_traits class.
     */
    static void print(std::ostream& out, Argument arg);

  }; // concept Argument


  /**
   * A concept that represents a discrete variable taking on a fixed
   * number of values.
   *
   * A type that models this concept must have the
   * argument_traits::argument_category trait convertible to discrete_tag.
   *
   * \see argument_traits
   */
  struct DiscreteArgument : Argument {

    /**
     * Returns the number of values a discrete variable can take on.
     * If this is a multivariate argument, returns the total numnber
     * values across all positions.
     *
     * This function is accessed via the argument_traits class.
     */
    static std::size_t num_values(DiscreteArgument arg);

    /**
     * Returns the number of values a multivariate discrete variable can
     * take on at a specific position.
     *
     * This function is accessed via the argument_traits class.
     */
    static std::size_t num_values(DiscreteArgument arg, std::size_t pos);

  }; // concept DiscreteArgument


  /**
   * A concept that represents a continuous variable taking on
   * real values.
   *
   * A type that models this concept must have the
   * argument_traits::argument_category trait convertible to continous_tag.
   *
   * \see argument_traits
   */
  struct ContinuousArgument : Argument { };


  /**
   * A concept that represents an argument that can be either discrete or
   * continuous. A call to num_values is only valid if the argument is discrete.
   *
   * A type that models this concept must have the
   * argument_traits::argument_category trait convertible to mixed_tag.
   *
   * \see argument_traits
   */
  struct MixedArgument : DiscreteArgument, ContinuousArgument {

    /**
     * Returns true if the given argument is discrete.
     *
     * This function is accessed via the argument_traits class.
     */
    static bool discrete(MixedArgument arg);

    /**
     * Returns true if the given argument is continuous.
     *
     * This function is accessed via the argument_traits class.
     */
    static bool continuous(MixedArgument arg);

  }; // concept MixedArgument


  /**
   * A concept that represents a univariate argument.
   *
   * A type that models this concept must have the
   * argument_traits::argument_category trait convertible to univariate_tag.
   *
   * \see argument_traits
   */
  struct UnivariateArgument : Argument { };


  /**
   * A concept that represents a multivariate argument.
   *
   * A type that models this concept must have the
   * argument_traits::argument_category trait convertible to multivariate_tag.
   *
   * \see argument_traits
   */
  struct MultivariateArgument : Argument {

    /**
     * Returns the length of the argument vector.
     *
     * This function is accessed via the argument_traits class.
     */
    static std::size_t size(MultivariateArgument arg);

  }; // concept MultivariateArgument


  /**
   * A concept that represents an indexed argument. An indexed argument consists
   * of two parts: a base, and an index. The argument_traits::base_type trait
   * specifies the base type, while the argument_traits::index_type trait
   * specifies the index type. The most common use of indexed arguments is in
   * random fields (including discrete-time random processes). For types that
   * do not model the IndexedArgument concept, the index_type must be void.
   *
   * \see argument_traits, field
   */
  struct IndexedArgument : Argument {
    /**
     * Returns true if the specified argument is indexed. If the argument
     * is not indexed, its index is undefined.
     *
     * This function is accessed via the argument_traits class.
     */
    static bool indexed(IndexedArgument arg);

    /**
     * Returns the base of the argument.
     *
     * This function is accessed via the argument_traits class.
     */
    static typename argument_traits<IndexedArgument>::base_type
    base(IndexedArgument arg);

    /**
     * Returns the index of the argument, provided that the argument is indexed.
     *
     * This function is accessed via the argument_traits class.
     */
    static typename argument_traits<IndexedArgument>::index_type
    index(IndexedArgument arg);

  }; // concept IndexedArgument


  /**
   * A concept that represents a domain.
   * A domain is a SequenceContainer that stores arguments.
   */
  struct Domain {

    /**
     * The underlying argument type. Must be the same as value_type.
     */
    typedef typename Domain::argument_type argument_type;

    /**
     * Returns true if the given domain is a prefix of this domain.
     */
    bool prefix(const Domain& other) const;

    /**
     * Returns true if the given domain is a suffic of this domain.
     */
    bool suffix(const Domain& other) const;

    /**
     * Partitions this domain into those arguments that are present in the
     * given associative container (set or map) and those that are not.
     *
     * \tparam Set a type that provides count(x) where x is of type Arg
     * \param present the ordered intersection of this domain with set
     * \param absent the ordered difference of this domain and the set
     */
    template <typename Set>
    void partition(const Set& set, Domain& present, Domain& absent) const;

    /**
     * Returns the concatenation of two hybrid domains.
     * This operation has a liner time complexity.
     */
    friend Domain concat(const Domain& a, const Domain& b);

    /**
     * Removes the duplicate arguments from the domain in place.
     * Does not preserve the relative order of arguments in the domain.
     */
    void unique();

    /**
     * Returns the number of times an argument is present in the domain.
     * This operation has at most a linear time complexity.
     */
    std::size_t count(argument_type x) const;

    /**
     * Returns the ordered union of two domains.
     * This operation has at most a quadratic time complexity, O(|a| * |b|).
     */
    friend Domain operator+(const Domain& a, const Domain& b);

    /**
     * Returns the ordered difference of two domains.
     * This operation has at most a quadratic time complexity, O(|a| * |b|).
     */
    friend Domain operator-(const Domain& a, const Domain& b);

    /**
     * Returns the ordered intersection of two domains.
     * This operation has at most a quadratic time complexity, O(|a| * |b|).
     */
    friend Domain intersect(const Domain& a, const Domain& b);

    /**
     * Returns true if two domains do not have any arguments in common.
     * This operation has at most a quadratic time complexity, O(|a| * |b|).
     */
    friend bool disjoint(const Domain& a, const Domain& b);

    /**
     * Returns true if two domains contain the same set of arguments
     * (disregarding the order).
     * This operation has at most a quadratic time complexity, O(|a| * |b|).
     */
    friend bool equivalent(const Domain& a, const Domain& b);

    /**
     * Returns true if all the arguments of the first domain are also
     * present in the second domain.
     * This operation has at most a quadratic time complexity, O(|a| * |b|).
     */
    friend bool subset(const Domain& a, const Domain& b);

    /**
     * Returns true if all the arguments of the second domain are also
     * present in the first domain.
     * This operation has at most a quadratic time complexity, O(|a| * |b|).
     */
    friend bool superset(const Domain& a, const Domain& b);

    /**
     * Returns true if two domains are compatible. Two domains are compatible
     * if they have the same cardinality and the corresponding arguments are
     * compatible as specified by argument_traits<argument_type>.
     */
    friend bool compatible(const domain& a, const domain& b);

    /**
     * Returns the number of values for a collection of discrete arguments.
     * This is equal to to the product of the numbers of values of the argument.
     *
     * This function is supported only when Arg is discrete.
     *
     * \throw std::out_of_range in case of possible overflow
     */
    std::size_t num_values() const;

    /**
     * Returns the overall dimensionality of discrete and continuous arguments
     * combined. For univariate arguments, this is simply the total cardinality
     * of the domain. For multivariate arguments, this is equal to the sum of
     * argument sizes.
     */
    std::size_t num_dimensions() const;

    /**
     * Substitutes arguments in-place according to a map. The keys of the map
     * must include all the arguments in this domain.
     *
     * \throw std::out_of_range if an argument is not present in the map
     * \throw std::invalid_argument if the arguments are not compatible
     */
    template <typename Map>
    void substitute(const Map& map);

  }; // concept Domain


  /**
   * A concept that represents an assignment.
   * An assignment is an UnorderedAssociativeContainer whose keys are arguments.
   */
  struct Assignment {

    /**
     * The domain associated with an assignment.
     */
    typedef typename Assignment::domain_type domain_type;

    /**
     * The vector of values associated with an assignment.
     */
    typedef typename Assignment::vector_type vector_type;

    /**
     * Returns the vector of values in this assignment for a subset
     * of arguments in the order specified by the given domain.
     *
     * \throw std::out_of_range if an argument is not present
     */
    vector_type values(const domain_type& args) const;

    /*
     * Inserts the keys drawn from a domain and the corresponding values
     * concatenated in a dense vector. If a key already exists, its original
     * value is preserved, similarly to std::unordered_map::insert.
     *
     * \return the number of values inserted
     */
    std::size_t insert(const domain_type& args, const vector_type& values);

    /**
     * Inserts the keys drawn from a domain and the corresponding values
     * concatenated in a dense vector. If a key already exists, its value is
     * is overwritten, similarly to std::unordereed_map::insert_or_assign.
     *
     * \return the number of values inserted
     */
    std::size_t insert_or_assign(const domain_type& args,
                                 const vector_type& values);

    /**
     * Returns true if all the arguments in the given domain are present in
     * the given assignment.
     */
    friend bool subset(const domain_type& args, const Assignment& a);

    /**
     * Returns true if none of the arguments in the given domain are present in
     * the given assignment.
     */
    friend bool disjoint(const domain<Arg>& args, const basic_assignment& a) {

  }; // concept Assignment

} // namespace libgm

#endif
