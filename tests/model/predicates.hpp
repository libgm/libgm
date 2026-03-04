#ifndef LIBGM_TEST_MODEL_PREDICATES_HPP
#define LIBGM_TEST_MODEL_PREDICATES_HPP

#include <libgm/argument/uint_assignment_iterator.hpp>
#include <algorithm>
#include <vector>

#include "../predicates.hpp"

struct factor_arg_less {
  template <typename F>
  bool operator()(const F& a, const F& b) {
    return a.arguments() < b.arguments();
  }
};

template <typename Model>
boost::test_tools::predicate_result
model_equal_factors(const Model& a, const Model& b) {
  typedef typename Model::value_type factor_type;
  std::vector<factor_type> a_factors(a.factors().begin(), a.factors().end());
  std::vector<factor_type> b_factors(b.factors().begin(), b.factors().end());
  std::sort(a_factors.begin(), a_factors.end(), factor_arg_less());
  std::sort(b_factors.begin(), b_factors.end(), factor_arg_less());
  if (a_factors != b_factors) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The two models do not have equivalent factor sets:\n"
                     << a_factors << " != " << b_factors;
    return result;
  }
  return true;
}

template <typename Model>
boost::test_tools::predicate_result
model_close_log_likelihoods(const Model& a, const Model& b, double eps) {
  using namespace libgm;
  domain<var> arga(a.arguments());
  domain<var> argb(b.arguments());
  if (!equivalent(arga, argb)) {
    arga.unique();
    argb.unique();
    boost::test_tools::predicate_result result(false);
    result.message() << "The two models do not have identical argument sets: "
                     << arga << " != " << argb;
    return result;
  }
  uint_assignment_iterator<var> it(arga), end;
  for(; it != end; ++it) {
    if (std::abs(a.log(*it) - b.log(*it)) > eps) {
      boost::test_tools::predicate_result result(false);
      result.message() << "The two models differ on the assignment";
      for (const auto& p : *it) {
        result.message() << ' ' << p.first << "=>" << p.second;
      }
      result.message() << '[' << a.log(*it) << ',' << b.log(*it) << ']';
      return result;
    }
  }
  return true;
}

#endif
