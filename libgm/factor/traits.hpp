#ifndef LIBGM_FACTOR_TRAITS_HPP
#define LIBGM_FACTOR_TRAITS_HPP

#include <type_traits>

namespace libgm {

  //! \addtogroup factor_traits
  //! @{

  // Aliases to member types
  //============================================================================

  /**
   * Alias for the argument type of a factor.
   * \tparam F A factor expression type or reference, possibly cv-qualified.
   */
  template <typename F>
  using argument_t = typename std::decay_t<F>::argument_type;

  /**
   * Alias for the domain type of a factor.
   * \tparam F A factor expression type or reference, possibly cv-qualified.
   */
  template <typename F>
  using domain_t = typename std::decay_t<F>::domain_type;

  /**
   * Alias for the assignment type of a factor.
   * \tparam F A factor expression type or reference, possibly cv-qualified.
   */
  template <typename F>
  using assignment_t = typename std::decay_t<F>::assignment_type;

  /**
   * Alias for the real type of a factor.
   * \tparam F A factor expression type or reference, possibly cv-qualified.
   */
  template <typename F>
  using real_t = typename std::decay_t<F>::real_type;

  /**
   * Alias for the result type of a factor.
   * \tparam F A factor expression type or reference, possibly cv-qualified.
   */
  template <typename F>
  using result_t = typename std::decay_t<F>::result_type;

  /**
   * Alias for the parameter type of a factor.
   * \tparam F A factor expression type or reference, possibly cv-qualified.
   */
  template <typename F>
  using param_t = typename std::decay_t<F>::param_type;

  /**
   * Alias for the factor type of a factor expression.
   * \tparam F A factor expression type or reference, possibly cv-qualified.
   */
  template <typename F>
  using factor_t = typename std::decay_t<F>::factor_type;

  /**
   * Alias for the space type of a discrete factor.
   * \tparam F A factor expression type or reference, possibly cv-qualified.
   */
  template <typename F>
  using space_t = typename std::decay_t<F>::space_type;

  /**
   * Alias for the base expression type for the given factor type.
   * \tparam F A factor type or reference, possibly cv-qualified.
   */
  template <typename F, typename Derived>
  using base_t = typename std::decay_t<F>::template base<Derived>;


  // Expression category
  //----------------------------------------------------------------------------

  namespace experimental {

    /**
     * A trait that represents whether the expression is primitive.
     * A primitive expression provides a constant-time access to its parameters,
     * and the parameters are returned by (const-)reference.
     *
     * This trait defaults to std::false_type and needs to be specialized
     * for all expression types that are known to be primitive. However,
     * we provide partial specializations for references and const-references
     * that delegate to class type, thus implicitly removing references.
     */
    template <typename Expr>
    struct is_primitive : std::false_type { };

    template <typename Expr>
    struct is_primitive<Expr&> : is_primitive<Expr> { };

    template <typename Expr>
    struct is_primitive<Expr&&> : is_primitive<Expr> { };

    template <typename Expr>
    struct is_primitive<const Expr&> : is_primitive<Expr> { };

    template <typename Expr>
    struct is_primitive<const Expr&&> : is_primitive<Expr> { };

    /**
     * A trait that represents whether the expression is mutable.
     * A mutable expression provides a constant-time access to its parameters,
     * and the parameters are returned by reference.
     *
     * This trait defaults to std::false_type and needs to be specialized
     * for all expression types that are known to be mutable. However,
     * we provide partial specializations for references that delegate to
     * the class type, thus implicitly removing the reference.
     */
    template <typename Expr>
    struct is_mutable : std::false_type { };

    template <typename Expr>
    struct is_mutable<Expr&> : is_mutable<Expr> { };

    template <typename Expr>
    struct is_mutable<Expr&&> : is_mutable<Expr> { };
  }

  /**
   * Represents the true_type if the two factor types have the same
   * real_type, result_type, variable_type, and assignment_type.
   */
  template <typename F, typename G>
  struct are_pairwise_compatible : public std::integral_constant<
    bool,
    std::is_same<real_t<F>, real_t<G> >::value &&
    std::is_same<result_t<F>, result_t<G> >::value &&
    std::is_same<argument_t<F>, argument_t<G> >::value &&
    std::is_same<assignment_t<F>, assignment_t<G> >::value
  > { };

  // Supported factor operations
  //============================================================================

// A macro that defines the detail for the has-binary-operator trait
#define LIBGM_HAS_BINARY_OPERATOR(trait_name, op)                       \
  invalid_op operator op(const any_arg&, const any_arg&);               \
  template <typename F, typename G>                                     \
  struct trait_name                                                     \
    : is_valid_op<decltype(std::declval<F>() op std::declval<G>())> { };

// A macro that defines the detail for the has-binary-function trait
#define LIBGM_HAS_BINARY_FUNCTION(fn)                                   \
  invalid_op fn(const any_arg&, const any_arg&);                        \
  template <typename... Args>                                           \
  struct has_##fn                                                       \
    : is_valid_op<decltype(fn(std::declval<Args>()...))> { };

// A macro that defines the detail for the has-ternary-function trait
#define LIBGM_HAS_TERNARY_FUNCTION(fn)                                  \
  invalid_op fn(const any_arg&, const any_arg&, const any_arg&);        \
  template <typename... Args>                                           \
  struct has_##fn                                                       \
    : is_valid_op<decltype(fn(std::declval<Args>()...))> { };

// A macro that defines the detail for the has-member-function trait
#define LIBGM_HAS_MEMBER_FN(fn)                                         \
  template <typename F, typename... Args>                               \
  struct has_##fn : has_member {                                        \
    using has_member::test;                                             \
    template <typename G = F>                                           \
    static std::true_type                                               \
    test(decltype((void)std::declval<G>().fn(std::declval<Args>()...))*); \
  };

  namespace detail {

    /**
     * A type that represents an invalid operation. This needs to be distinct
     * from any valid implementation of an operator / function.
     */
    struct invalid_op { };

    /**
     * A type that represents any argument. The functions taking this argument
     * will be less preferred than the functions declared for the factors.
     */
    struct any_arg {
      template <typename T> any_arg(const T& t);
    };

    /**
     * A trait that indicates whether the specified type is the result of
     * a valid operation.
     */
    template <typename T>
    struct is_valid_op
      : std::integral_constant<bool, !std::is_same<T, invalid_op>::value> { };

    /**
     * A base class that specifies that the member is not present.
     */
    struct has_member {
      static std::false_type test(const any_arg&);
    };

    LIBGM_HAS_BINARY_OPERATOR(has_plus, +)
    LIBGM_HAS_BINARY_OPERATOR(has_minus, -)
    LIBGM_HAS_BINARY_OPERATOR(has_multiplies, *)
    LIBGM_HAS_BINARY_OPERATOR(has_divides, /)

    LIBGM_HAS_BINARY_OPERATOR(has_plus_assign, +=)
    LIBGM_HAS_BINARY_OPERATOR(has_minus_assign, -=)
    LIBGM_HAS_BINARY_OPERATOR(has_multiplies_assign, *=)
    LIBGM_HAS_BINARY_OPERATOR(has_divides_assign, /=)

    LIBGM_HAS_TERNARY_FUNCTION(weighted_update)

    LIBGM_HAS_BINARY_FUNCTION(max)
    LIBGM_HAS_BINARY_FUNCTION(min)
    LIBGM_HAS_BINARY_FUNCTION(cross_entropy)
    LIBGM_HAS_BINARY_FUNCTION(kl_divergence)
    LIBGM_HAS_BINARY_FUNCTION(js_divergence)
    LIBGM_HAS_BINARY_FUNCTION(max_diff)

    LIBGM_HAS_MEMBER_FN(head)
    LIBGM_HAS_MEMBER_FN(tail)
    LIBGM_HAS_MEMBER_FN(start)
    LIBGM_HAS_MEMBER_FN(marginal)
    LIBGM_HAS_MEMBER_FN(maximum)
    LIBGM_HAS_MEMBER_FN(minimum)
    LIBGM_HAS_MEMBER_FN(conditional);
    LIBGM_HAS_MEMBER_FN(restrict);
    LIBGM_HAS_MEMBER_FN(sample);
    LIBGM_HAS_MEMBER_FN(distribution);
    LIBGM_HAS_MEMBER_FN(entropy);
    LIBGM_HAS_MEMBER_FN(mutual_information);
  }

  /**
   * A trait that specifies whether two types can be added.
   */
  template <typename F, typename G = F>
  struct has_plus : detail::has_plus<F, G> { };

  /**
   * A trait that specifies whether one type can be subtracted from another.
   */
  template <typename F, typename G = F>
  struct has_minus : detail::has_minus<F, G> { };

  /**
   * A trait that specifies whether two types s can be multiplied together.
   */
  template <typename F, typename G = F>
  struct has_multiplies : detail::has_multiplies<F, G> { };

  /**
   * A trait that specifies whether one type can be divided by another.
   */
  template <typename F, typename G = F>
  struct has_divides : detail::has_divides<F, G> { };

  /**
   * A trait that specifies whether one type can be incremented by
   * another one inplace.
   */
  template <typename F, typename G = F>
  struct has_plus_assign : detail::has_plus_assign<F, G> { };

  /**
   * A trait that specifies whether one type can be decremented by
   * another one inplace.
   */
  template <typename F, typename G = F>
  struct has_minus_assign : detail::has_minus_assign<F, G> { };

  /**
   * A trait that specifies whether one type can be multiplied by
   * another one inplace.
   */
  template <typename F, typename G = F>
  struct has_multiplies_assign : detail::has_multiplies_assign<F, G> { };

  /**
   * A trait that specifies whether one type can be divided by
   * another one inplace.
   */
  template <typename F, typename G = F>
  struct has_divides_assign : detail::has_divides_assign<F, G> { };

  /**
   * A trait that specifies whether we can compute elementwise weighted
   * update of two factor types.
   */
  template <typename F, typename G = F>
  struct has_weighted_update : detail::has_weighted_update<F, G, real_t<F> > {};

  /**
   * A trait that specifies whether we can compute elementwise maximum
   * of two factor types.
   */
  template <typename F, typename G = F>
  struct has_max : detail::has_max<F, G> { };

  /**
   * A trait that specifies whether we can compute elementwise minimum
   * of two factor types.
   */
  template <typename F, typename G = F>
  struct has_min : detail::has_min<F, G> { };

  /**
   * A trait that specifies whether we can compute cross entropy between
   * two factor types.
   */
  template <typename F, typename G = F>
  struct has_cross_entropy : detail::has_cross_entropy<F, G> { };

  /**
   * A trait that specifies whether we can commpute KL divergence between
   * two factor types.
   */
  template <typename F, typename G = F>
  struct has_kl_divergence : detail::has_kl_divergence<F, G> { };

  /**
   * A trait that specifies whether we can commpute JS divergence between
   * two factor types.
   */
  template <typename F, typename G = F>
  struct has_js_divergence : detail::has_js_divergence<F, G> { };

  /**
   * A trait that specifies whether we can commpute the maximum difference
   * of parameters between two factor types.
   */
  template <typename F, typename G = F>
  struct has_max_diff : detail::has_max_diff<F, G> { };

  /**
   * A trait that specifies whether the given factor type provides explicit
   * head arguments.
   */
  template <typename F>
  struct has_head
    : decltype(detail::has_head<F>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type provides explicit
   * tail arguments.
   */
  template <typename F>
  struct has_tail
    : decltype(detail::has_tail<F>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type stores mapping
   * from arguments to dimensions.
   */
  template <typename F>
  struct has_start
    : decltype(detail::has_start<F>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type supports a marginal
   * over the given domain.
   */
  template <typename F, typename Domain = domain_t<F> >
  struct has_marginal
    : decltype(detail::has_marginal<F, Domain>::test(nullptr)) { };

  /**
   * A specialization of has_marginal that specifies whether the factor
   * can compute the marginal over its entire domain (i.e., the normalization
   * constant).
   */
  template <typename F>
  struct has_marginal<F, void>
    : decltype(detail::has_marginal<F>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type supports a maximum
   * over the given domain. Can be also used to test if the factor supports
   * the maximum assignment exdtraction.
   */
  template <typename F, typename Domain = domain_t<F> >
  struct has_maximum
    : decltype(detail::has_marginal<F, Domain>::test(nullptr)) { };

  /**
   * A specialization of has_maximum that specifies whether the factor
   * can compute the maximum over its entire domain.
   */
  template <typename F>
  struct has_maximum<F, void>
    : decltype(detail::has_marginal<F>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type supports a minimum
   * over the given domain. Can be also used to test if the factor supports
   * the minimum assignment exdtraction.
   */
  template <typename F, typename Domain = domain_t<F> >
  struct has_minimum
    : decltype(detail::has_minimum<F, Domain>::test(nullptr)) { };

  /**
   * A specialization of has_maximum that specifies whether the factor
   * can compute the minimum over its entire domain.
   */
  template <typename F>
  struct has_minimum<F, void>
    : decltype(detail::has_marginal<F>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type provides a function
   * that computes the conditional over the tail of given type.
   */
  template <typename F, typename Domain = domain_t<F> >
  struct has_conditional
    : decltype(detail::has_conditional<F, Domain>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type provides a function
   * that computes the restriction for the given assignment type.
   */
  template <typename F, typename Assignment = assignment_t<F> >
  struct has_restrict
    : decltype(detail::has_restrict<F, Assignment>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type can draw samples
   * using the specified uniform random number generator.
   */
  template <typename F, typename Generator>
  struct has_sample
    : decltype(detail::has_sample<F, Generator>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor provides the
   * distribution object.
   */
  template <typename F>
  struct has_distribution
    : decltype(detail::has_distribution<F>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type can compute
   * entropy over the given domain type.
   */
  template <typename F, typename Domain = domain_t<F> >
  struct has_entropy
    : decltype(detail::has_entropy<F, Domain>::test(nullptr)) { };

  /**
   * A specialization of has_entropy that specifies whether the factor
   * can compute entropy over its entire domain.
   */
  template <typename F>
  struct has_entropy<F, void>
    : decltype(detail::has_entropy<F>::test(nullptr)) { };

  /**
   * A trait that specifies whether the given factor type can compute
   * the mutual informaiton between two subsets of arguments of given
   * types.
   */
  template <typename F,
            typename Domain1 = domain_t<F>,
            typename Domain2 = domain_t<F> >
  struct has_mutual_information
    : decltype(
        detail::has_mutual_information<F, Domain1, Domain2>::test(nullptr)) { };

  //! @}

} // namespace libgm

#endif
