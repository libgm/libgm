#ifndef LIBGM_TEST_FACTOR_PREDICATES_HPP
#define LIBGM_TEST_FACTOR_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <libgm/datastructure/uint_vector.hpp>

#include "../../predicates.hpp"

// Checks the basic properties of vector factors
template <typename F>
boost::test_tools::predicate_result
vector_properties(const F& f, std::size_t n) {
  if (f.empty()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }
  if (f.arity() != 1) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << 1 << "]";
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

// Checks the basic properties of matrix factors
template <typename F>
boost::test_tools::predicate_result
matrix_properties(const F& f, std::size_t m, std::size_t n) {
  if (f.empty()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }
  if (f.arity() != 2) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << 2 << "]";
    return result;
  }
  if (f.size() != m * n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor size ["
                     << f.size() << " != " << m * n << "]";
    return result;
  }
  if (f.rows() != m || f.cols() != n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor shape ["
                     << "(" << f.rows() << "," << f.cols() << ") !="
                     << "(" << m << "," << n << ")";
    return result;
  }
  return true;
}

// Checks the basic properties of table factors
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
  if (f.shape() != shape) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor shape";
    return result;
  }
  return true;
}

#endif
