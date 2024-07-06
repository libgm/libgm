#pragma once

#include <libgm/math/constants.hpp>

#include <cmath>

namespace libgm {

/**
 * Operator for computing entropy of a categorical distribution or cross
 * entropy from one distribution to another. This operator computes
 * \f$-x log(x)\f$ or \f$-x log(y)\f$ for its unary and binary version,
 * respectively.  By convention, \f$0 log(0) = 0\f$.
 */
template <typename T>
struct EntropyOp {
  T operator()(const T& x) const {
    return (x == T(0)) ? T(0) : -x * std::log(x);
  }
  T operator()(const T& x, const T& y) const {
    return (x == T(0)) ? T(0) : -x * std::log(y);
  }
};

/**
 * Operator for computing the entropy of a categorical distribution or
 * cross entropy from one distribution to another. Unlike entropy_op,
 * this operator assumes that the data is represented in logspace.
 */
template <typename T>
struct EntropyLogOp {
  T operator()(const T& x) const {
    return (x == -inf<T>()) ? T(0) : -std::exp(x) * x;
  }
  T operator()(const T& x, const T& y) const {
    return (x == -inf<T>()) ? T(0) : -std::exp(x) * y;
  }
};

/**
 * Operator for computing the Kullback-Leibler divergence from one
 * categorical distribution to another. This operator computes
 * \f$f(x, y) = x * \log \frac{x}{y}\f$. By convention,
 * \f$0 log(0) = 0\f$.
 */
template <typename T>
struct KldOp {
  T operator()(const T& x, const T& y) const {
    return (x == T(0)) ? T(0) : x * (std::log(x) - std::log(y));
  }
};

/**
 * Operator for computing the Kullback-Leibler divergence from one
 * categorical distribution to another, when the two distributions
 * are represented in the log space.
 */
template <typename T>
struct KldLogOp {
  T operator()(const T& x, const T& y) const {
    return (x == -inf<T>()) ? T(0) : std::exp(x) * (x - y);
  }
};

/**
 * Operator for computing the Jensen–Shannon divergence between two
 * categorical distributions.
 */
template <typename T>
struct JsdOp {
  KldOp<T> kld;
  T operator()(const T& p, const T& q) const {
    T m = (p + q) / 2;
    return (kld(p, m) + kld(q, m)) / 2;
  }
};

/**
 * Operator for computing the Jensen–Shannon divergence between two
 * categorical distributions that are represented in the log space.
 */
template <typename T>
struct JsdLogOp {
  JsdOp<T> jsd;
  T operator()(const T& lp, const T& lq) const {
    return jsd(std::exp(lp), std::exp(lq));
  }
};

} // namespace libgm
