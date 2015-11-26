#ifndef LIBGM_TEST_FACTOR_PREDICATES_HPP
#define LIBGM_TEST_FACTOR_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <libgm/factor/canonical_gaussian.hpp>

#include "../predicates.hpp"

// Checks the basic properties of finite table (and matrix) factors
template <typename F>
boost::test_tools::predicate_result
table_properties(const F& f, const typename F::domain_type& vars) {
  auto val = libgm::domain<libgm::var>(vars).num_values();
  std::size_t n =
    std::accumulate(val.begin(), val.end(), 1, std::multiplies<std::size_t>());

  if (f.empty()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }
  if (f.arity() != vars.size()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << vars.size() << "]";
    return result;
  }
  if (f.size() != n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor size ["
                     << f.size() << " != " << n << "]";
    return result;
  }
  if (f.arguments() != vars) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor domain ["
                     << f.arguments() << " != " << vars << "]";
    return result;
  }
  return true;
}

// Verifies that two factors are close enough
template <typename F>
boost::test_tools::predicate_result
are_close(const F& a, const F& b, typename F::result_type eps) {
  typename F::result_type norma = a.marginal();
  typename F::result_type normb = b.marginal();
  if (a.arguments() == b.arguments() &&
      max_diff(a, b) < eps &&
      (norma > normb ? norma - normb : normb - norma) < eps) {
     return true;
  } else {
    boost::test_tools::predicate_result result(false);
    result.message() << "the two factors differ [\n"
                     << a << "!=" << b << "]";
    return result;
  }
}

template <typename T, typename Var>
boost::test_tools::predicate_result
are_close(const libgm::canonical_gaussian<T, Var>& a,
          const libgm::canonical_gaussian<T, Var>& b,
          T eps) {
  T multa = a.log_multiplier();
  T multb = b.log_multiplier();
  if (a.arguments() == b.arguments() &&
      max_diff(a, b) < eps &&
      std::abs(multa - multb) < eps) {
     return true;
  } else {
    boost::test_tools::predicate_result result(false);
    result.message() << "the two factors differ [\n"
                     << a << "!=" << b << "]";
    return result;
  }
}

#endif
