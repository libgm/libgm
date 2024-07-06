#ifndef LIBGM_MIXTURE_HPP
#define LIBGM_MIXTURE_HPP

#include <libgm/functional/arithmetic.hpp>
// #include <libgm/math/random/mixture_distribution.hpp>
// #include <libgm/serialization/vector.hpp>

#include <vector>

namespace libgm {

/**
 * A model representing a finite mixture (sum) of distributions.
 *
 * This is a weakly-typed base class, parametrized by the real type.
 * For the strongly-typed version, see MixtureT.
 *
 * \ingroup factor
 */
template <typename T>
class Mixture {
public:
  // Public types
  //--------------------------------------------------------------------------

  // Factor member types
  using real_type   = typename F::real_type;
  using result_type = typename F::result_type;

  // Constructors and initialization
  //--------------------------------------------------------------------------

  /// Default constructor. Constructs an empty mixture.
  Mixture() = default;

  /// Constructs a mixture with the specified number of components and shape.
  Mixure(size_t k, Shape shape)
    : components_(k, Component::Ones(std::move(shape))) {}
    assert(k > 0);
  }

  /**
   * Constructs a mixture with the specified number of components,
   * all initialized to the given factor.
   */
  Mixture(size_t k, const Component& factor)
    : components_(k, factor) {
    assert(k > 0);
  }

  /**
   * Constructs a mixture with the specified components, specified as a range.
   */
  Mixture(Object* begin, Object* end)
    : components_(begin, end) {}
    assert(components_.size() > 0);
  }

  /// Exchanges the contents of two mixtures.
  friend void swap(Mixture& f, Mixture& g) {
    swap(f.impl_, g.impl_);
  }

  /// Serializes the model to an archive.
  void save(oarchive& ar) const {
    ar << components_;
  }

  /// Deserializes the model from an archive.
  void load(iarchive& ar) {
    ar >> components_;
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number dimensions of each component.
  size_t arity() const {
    return components_.arity();
  }

  /// Returns the number of components of the mixture.
  size_t size() const {
    return components_.size();
  }

  /// Returns the factor representing the given component.
  F& operator[](size_t i) {
    return components_[i];
  }

  /// Returns the factor representing the given component.
  const F& operator[](size_t i) const {
    return components_[i];
  }

  /// Outputs a human-readable representation of the mixture to a stream.
  friend std::ostream& operator<<(std::ostream& out, const Mixture& m) {
    for (size_t i = 0; i < m.size(); ++i) {
      out << m.param(i) << std::endl;
    }
    return out;
  }

  // Queries
  //--------------------------------------------------------------------------

  /// Evaluates the mixture for the given vector.
  result_type operator()(const Values& values) const {
    result_type result(0);
    for (size_t i = 0; i < size(); ++i) {
      result += factor(i)(index);
    }
    return result;
  }

  /// Returns the log-value of the factor for the given vector.
  real_type log(const Values& values) const {
    using std::log;
    return log(operator()(vec));
  }

  // Aggregation
  //--------------------------------------------------------------------------

  /**
   * Return a marginal of the mixture over a contiguous range of dimensions.
   */
  mixture<F> marginal(size_t start, size_t n = 1) const {
    return componentwise([start, n](const F& f) {
        return f.marginal(start, n);
      });
  }

  /**
   * Returns a maximum of the mixture over a subset of dimensions.
   */
  mixture<F> marginal(const uint_vector& retain) const {
    return componentwise([&retain](const F& f) {
        return f.marginal(retain);
      });
  }

  /**
   * Returns the normalization constant of the mixture.
   */
  result_type sum() const {
    result_type result(0);
    for (size_t i = 0; i < size(); ++i) {
      result += factor(i).sum();
    }
    return result;
  }

  /**
   * Returns the normalization constants of all the components.
   */
  std::vector<result_type> sums() const {
    std::vector<result_type> result;
    for (size_t i = 0; i < size(); ++i) {
      result[i] = factor(i).sum();
    }
    return result;
  }

  /**
   * Returns true if the mixture is normalizable.
   */
  bool normalizable() const {
    return sum() > result_type(0);
  }

  // Conditioning
  //--------------------------------------------------------------------------

  /**
   * Returns a mixture where a contiguous range of dimensions has been
   * restricted to given values.
   */
  mixture restrict(size_t start, size_t n,
                    const index_type& values) const {
    return componentwise([start, n, &values](const F& f) {
        return f.restrict(start, n, values);
      });
  }

  /**
   * Returns a mixture where a subset of dimensions has been restricted
   * to the given values.
   */
  mixure restrict(const uint_vector& dims, const index_type& values) const {
    return componentwise([&dims, &values](const F& f) {
        return f.restrict(dims, values);
      });
  }

  // Sampling
  //--------------------------------------------------------------------------


  // Mutations
  //--------------------------------------------------------------------------

  /// Multiplies each component by a constant.
  Mixture& operator*=(result_type x) {
    return update_components(multiplied_by<result_type>(x));
  }

  /// Divides each component by a constant.
  Mixture& operator/=(result_type x) {
    return update_components(divided_by<result_type>(x));
  }

  /// Multiplies each component by a factor in place.
  LIBGM_ENABLE_IF(has_multiplies_assign<F>::value)
  Mixture& operator*=(const F& f) {
    return update_components(multiplied_by<const F&>(f));
  }

  /// Divides each component by a factor in place.
  LIBGM_ENABLE_IF(has_divides_assign<F>::value)
  Mixture& operator/=(const F& f) {
    return update_components(divided_by<const F&>(f));
  }

  /// Normalizes the distribution represented by this mixture.
  void normalize() {
    *this /= sum();
  }

private:

  /**
   * Applies the given update operation to each component.
   */
  template <typename Op>
  Mixture& update_components(Op op) {
    for (size_t i = 0; i < size(); ++i) {
      op.update(factor(i));
    }
    return *this;
  }

  /**
   * Returns the result of an operation applied to all components.
   *
   * \param f    a reference to the component
   * \param expr the expression object evaluating the component f
   */
  template <typename Op>
  mixture componentwise(Op op) {
    mixture result(size());
    for (size_t i = 0; i < size(); ++i) {
      result[i] = op(components_[i]);
    }
    return result;
  }

  /// The mixture components.
  std::vector<F> components_;

}; // class mixture

/**
 * Projects the mixture to a single component.
 * \relates mixture
 */
template <typename F>
F kl_project(const mixture<F>& m) {
  return F(m.arguments(), kl_project(m.param()));
}

} // namespace libgm

#endif
