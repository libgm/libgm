#pragma once

#include <libgm/argument/dims.hpp>
#include <libgm/argument/shape.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <type_traits>
#include <utility>
#include <vector>

namespace libgm {

/**
 * A finite mixture of factors with identical domain/shape.
 */
template <typename F>
class Mixture {
public:
  template <Argument Arg>
  using assignment_t = typename F::template assignment_t<Arg>;
  using component_result_type = typename F::result_type;
  using real_type = typename F::real_type;
  using result_type = real_type;
  using value_list = typename F::value_list;
  using value_type = typename F::value_type;

  Mixture() = default;

  explicit Mixture(size_t k)
    : components_(k) {}

  Mixture(size_t k, const F& factor)
    : components_(k, factor) {
    assert(k > 0);
  }

  explicit Mixture(size_t k, Shape shape)
    : components_() {
    assert(k > 0);
    components_.reserve(k);
    for (size_t i = 0; i < k; ++i) {
      components_.emplace_back(shape);
    }
  }

  Mixture(std::initializer_list<F> components)
    : components_(components) {
    assert(components_.size() > 0);
  }

  template <typename Iter>
  Mixture(Iter first, Iter last)
    : components_(first, last) {
    assert(components_.size() > 0);
  }

  friend void swap(Mixture& a, Mixture& b) {
    std::swap(a.components_, b.components_);
  }

  size_t size() const {
    return components_.size();
  }

  size_t arity() const {
    return components_.empty() ? 0 : components_.front().arity();
  }

  const Shape& shape() const {
    assert(!components_.empty());
    return components_.front().shape();
  }

  F& operator[](size_t i) {
    return components_[i];
  }

  const F& operator[](size_t i) const {
    return components_[i];
  }

  auto begin() {
    return components_.begin();
  }

  auto end() {
    return components_.end();
  }

  auto begin() const {
    return components_.begin();
  }

  auto end() const {
    return components_.end();
  }

  result_type operator()(const value_list& values) const {
    real_type result = 0;
    for (const F& component : components_) {
      result += static_cast<real_type>(component(values));
    }
    return result;
  }

  real_type log(const value_list& values) const {
    return std::log(static_cast<real_type>(operator()(values)));
  }

  result_type marginal() const {
    real_type result = 0;
    for (const F& component : components_) {
      result += static_cast<real_type>(component.marginal());
    }
    return result;
  }

  std::vector<result_type> marginals() const {
    std::vector<result_type> result;
    result.reserve(components_.size());
    for (const F& component : components_) {
      result.push_back(component.marginal());
    }
    return result;
  }

  Mixture marginal_front(unsigned n) const {
    return transform([n](const F& component) {
      return component.marginal_front(n);
    });
  }

  Mixture marginal_back(unsigned n) const {
    return transform([n](const F& component) {
      return component.marginal_back(n);
    });
  }

  Mixture marginal_dims(const Dims& dims) const {
    return transform([&dims](const F& component) {
      return component.marginal_dims(dims);
    });
  }

  Mixture restrict_front(const value_list& values) const {
    return transform([&values](const F& component) {
      return component.restrict_front(values);
    });
  }

  Mixture restrict_back(const value_list& values) const {
    return transform([&values](const F& component) {
      return component.restrict_back(values);
    });
  }

  Mixture restrict_dims(const Dims& dims, const value_list& values) const {
    return transform([&dims, &values](const F& component) {
      return component.restrict_dims(dims, values);
    });
  }

  Mixture& operator*=(component_result_type x) {
    for (F& component : components_) {
      component *= x;
    }
    return *this;
  }

  Mixture& operator/=(component_result_type x) {
    for (F& component : components_) {
      component /= x;
    }
    return *this;
  }

  void normalize() {
    *this /= make_component_result(marginal());
  }

private:
  static component_result_type make_component_result(real_type value) {
    if constexpr (std::is_same_v<component_result_type, real_type>) {
      return value;
    } else {
      using std::log;
      return component_result_type(log(value));
    }
  }

  template <typename Op>
  Mixture transform(Op op) const {
    Mixture result;
    result.components_.reserve(components_.size());
    for (const F& component : components_) {
      result.components_.push_back(op(component));
    }
    return result;
  }

  std::vector<F> components_;
};

template <typename F>
using mixture = Mixture<F>;

template <typename F>
std::ostream& operator<<(std::ostream& out, const Mixture<F>& mixture) {
  out << "Mixture([";
  for (size_t i = 0; i < mixture.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << mixture[i];
  }
  out << "])";
  return out;
}

} // namespace libgm
