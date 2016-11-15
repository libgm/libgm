#ifndef LIBGM_TEST_FACTOR_PREDICATES_HPP
#define LIBGM_TEST_FACTOR_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include "../predicates.hpp"

// Checks the basic properties of finite table (and matrix) factors
template <typename F>
boost::test_tools::predicate_result
table_properties(const F& f, const typename F::domain_type& vars) {
  auto val = libgm::domain<libgm::var>(vars).num_values();
  std::size_t n =
    std::accumulate(val.begin(), val.end(), std::size_t(1), std::multiplies<std::size_t>());

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

#endif
