#ifndef LIBGM_TEST_FACTOR_PREDICATES_HPP
#define LIBGM_TEST_FACTOR_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/dense.hpp>

#include "../predicates.hpp"

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

// Checks the basic properties of canonical_gaussians
template <typename F>
boost::test_tools::predicate_result
cg_properties(const F& f, std::size_t n) {
  if (f.empty() && n > 0) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }
  if (f.arity() != n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << n << "]";
    return result;
  }
  return true;
}

// Checks the parameters of the canonical_gaussian
template <typename F>
boost::test_tools::predicate_result
cg_params(const F& f,
          const libgm::dense_vector<>& eta,
          const libgm::dense_matrix<>& lambda,
          double lm) {
  if (!f.inf_vector().isApprox(eta, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Information vectors not close: "
                     << f.inf_vector().transpose() << " vs "
                     << eta.transpose();
    return result;
  }
  if (!f.inf_matrix().isApprox(lambda, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Information matrices not close: \n"
                     << f.inf_matrix() << "\n vs \n" << lambda;
    return result;
  }
  if (std::abs(f.log_multiplier() - lm) > 1e-8) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Log-multipliers not close: "
                     << f.log_multiplier() << " vs " << lm;
    return result;
  }
  return true;
}

template <typename F>
boost::test_tools::predicate_result
mg_properties(const F& f, std::size_t m, std::size_t n = 0) {
  if (f.empty() && m + n > 0) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }
  if (f.arity() != m + n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << m + n << "]";
    return result;
  }
  if (f.head_arity() != m) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid head arity ["
                     << f.head_arity() << " != " << m << "]";
    return result;
  }
  if (f.tail_arity() != n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.tail_arity() << " != " << n << "]";
    return result;
  }

  return true;
}

template <typename F>
boost::test_tools::predicate_result
mg_params(const F& f,
          const libgm::dense_vector<>& mean,
          const libgm::dense_matrix<>& cov,
          const libgm::dense_matrix<>& coef,
          double lm) {
  if (!f.mean().isApprox(mean, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Means not close: "
                     << f.mean().transpose() << " vs "
                     << mean.transpose();
    return result;
  }
  if (!f.covariance().isApprox(cov, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Covariance matrices not close: \n"
                     << f.covariance() << "\n vs \n" << cov;
    return result;
  }
  if (!f.coefficients().isApprox(coef, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Coefficient matrices not close: \n"
                     << f.coefficients() << "\n vs \n" << coef;
    return result;
  }
  if (std::abs(f.log_multiplier() - lm) > 1e-8) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Log-multipliers not close: "
                     << f.log_multiplier() << " vs " << lm;
    return result;
  }
  return true;
}

#endif
