#ifndef LIBGM_TEST_FACTOR_PREDICATES_HPP
#define LIBGM_TEST_FACTOR_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <libgm/datastructure/uint_vector.hpp>

#include "../../predicates.hpp"

// Checks the basic properties of finite table (and matrix) factors
template <typename F>
boost::test_tools::predicate_result
table_properties(const F& f, const libgm::uint_vector& shape) {
  std::size_t n =
    std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                    std::multiplies<std::size_t>());

  if (f.empty()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }
  if (f.arity() != shape.size()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << shape.size() << "]";
    return result;
  }
  if (f.size() != n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor size ["
                     << f.size() << " != " << n << "]";
    return result;
  }
  return true;
}

#endif
