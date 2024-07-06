#include "../mixture.hpp"

namespace libgm {

template <typename R>
struct Mixture<R>::Impl : Object::Impl {

  /// The weakly-typed components of the mixture.
  std::vector<Component> components;

  /// The virtual table for the factors.
  Factor::VTable vt;

  // Object operations
  //--------------------------------------------------------------------------

  void print(std::ostream& out) const override {
    out << "Mixture(["
    for (size_t i = 0; i < components.size(); ++i) {
      if (i > 0) out << ',';
      out << std::endl << components[i];
    }
    out << "])";
  }

  void save(oarchive& ar) const override {
    ar << components;
  }

  void load(iarchive& ar) override {
    ar >> components;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  ImplPtr multiply_result(const R& x) const {
    return componentwise([&x, this](const Component& component) {
      return multiply(component, x, vt);
    });
  }

  ImplPtr divide(const R& x) const {
    return componentwise([&x, this](const Component& component) {
      return divide(component, x, vt);
    });
  }

  ImplPtr multiply(const Object& other) const {
    const componentwise([&other, this](const Component& component) {
      Impl& x = impl(other);
    check(x.shape);
    return std::make_unique<Impl>(shape, eta + x.eta, lambda + x.lambda, lm + x.lm);
  }

  ImplPtr divide(const Object& other) const {
    const Impl& x = impl(other);
    check(x.shape);
    return std::make_unique<Impl>(shape, eta - x.eta, lambda - x.lambda, lm - x.lm);
  }

  void multiply_in(const R& x) {
    for (Component& component : components) {
      component.multiply_in(x, vt);
    }
  }

  void divide_in(const R& x) {
    for (Component& component : components) {
      component.divide_in(x, vt);
    }
  }

  void multiply_in(const Component& other) {
    for (Component& component : components) {
      component.multiply_in(other, vt);
    }
  }

  void divide_in(const Component& other) {
    for (Component& component: components) {
      component.divide_in(other, vt);
    }
  }

  // Direct operations
  //--------------------------------------------------------------------------

  R marginal() const {
    R result = components[0].marginal();
    for (size_t i = 1; i < components.size(); ++i) {
      result += components[i].marginal();
    }
    return result;
  }

#if 0
  /**
   * Draws a random component from this mixture.
   */
  template <typename Generator>
  size_t sample_component(Generator& rng) const {
    categorical_distribution<real_type> dist(probabilities(), prob_tag());
    return dist(rng);
  }

  /**
   * Draws a random sample from the distribution represented by this mixture.
   */
  index_type sample(Generator& rng) const {
    return factor(sample_component(rng)).sample(rng);
  }

  /**
   * Returns the distribution for this mixture.
   */
  auto distribution() const {
    return mixture_distribution<typename F::distribution_type>(param());
  }
#endif

};


}