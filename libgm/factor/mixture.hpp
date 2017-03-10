#ifndef LIBGM_MIXTURE_HPP
#define LIBGM_MIXTURE_HPP

#include <libgm/enable_if.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/factor/substituted_param.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/math/random/mixture_distribution.hpp>
#include <libgm/serialization/vector.hpp>

#include <vector>

namespace libgm {

  /**
   * A model representing a finite mixture (sum) of distributions.
   *
   * \tparam F
   *         The factor type representing each mixture component.
   *
   * \ingroup model
   */
  template <typename F>
  class mixture {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // Factor member types
    using real_type   = typename F::real_type;
    using result_type = typename F::result_type;
    using factor_type = mixture<F>;

    // ParametricFactor member types
    using param_type = std::vector<typename F::param_type>;
    using shape_type = std::pair<std::size_t, typename F::shape_type>;
    using index_type = typename F::index_type;

    // Constructors and initialization
    //--------------------------------------------------------------------------

    //! Default constructor. Constructs an empty mixture.
    mixture() { }

    /**
     * Constructs a mixture with the specified number of components and
     * default-constructed mixture components.
     */
    explicit mixture(std::size_t k)
      : components_(k) {
      assert(k > 0);
    }

    /**
     * Constructs a mixture with the specified number of components,
     * each initialized to the given shape.
     */
    mixure(std::size_t k, const typename F::shape_type& shape)
      : components_(k) {
      assert(k > 0);
      for (std::size_t i = 0; i < k; ++i) {
        components_.reset(shape);
      }
    }

    /**
     * Constructs a mixture with the specified number of components,
     * each initialized to the given parameter.
     */
    mixture(std::size_t k, const typename F::param_type& param)
      : components_(k) {
      assert(k > 0);
      for (std::size_t i = 0; i < k; ++i) {
        components_[i].param() = param;
      }
    }

    /**
     * Constructs a mixutre with the specified number of components,
     * all initialized to the given factor.
     */
    mixture(std::size_t k, const F& factor)
      : components_(k, factor) {
      assert(k > 0);
    }

    /**
     * Constructs a mixture with the specified parameters.
     */
    explicit mixture(const param_type& param)
      : components_(param.size()) {
      assert(param.size() > 0);
      for (std::size_t i = 0; i < param.size(); ++i) {
        components_[i].param() = param[i];
      }
    }

    //! Exchanges the contents of two mixtures.
    friend void swap(mixture& m, mixture& n) {
      swap(m.components_, n.components_);
    }

    //! Serializes the model to an archive.
    void save(oarchive& ar) const {
      ar << components_;
    }

    //! Deserializes the model from an archive.
    void load(iarchive& ar) {
      ar >> components_;
    }

    //! Resets the mixture to the given number of components with given shape.
    void reset(std::size_t k, const typename F::shape_type& shape) {
      components_.resize(k);
      for (std::size_t i = 0; i < k; ++i) {
        components_[k].reset(shape);
      }
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the number dimensions of each component.
    std::size_t arity() const {
      return components_.arity();
    }

    //! Returns true if the mixture has no components.
    bool empty() const {
      return components_.empty();
    }

    //! Returns the number of components of the mixture.
    std::size_t size() const {
      return components_.size();
    }

    //! Returns the parameters associated with the given component index.
    param_type& param(std::size_t i) {
      return param_[i];
    }

    //! Returns the parameters associated with the given component index.
    const param_type& param(std::size_t i) const {
      return param_[i];
    }

    //! Returns the parameter vector (a copy).
    param_type param() const {
      param_type result(size());
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = param(i);
      }
      return result;
    }

    //! Returns the factor representing the given component.
    F& operator[](std::size_t i) {
      return components_[i];
    }

    //! Returns the factor representing the given component.
    const F& operator[](std::size_t i) const {
      return components_[i];
    }

    //! Outputs a human-readable representation of the mixture to a stream.
    friend std::ostream& operator<<(std::ostream& out, const mixture& m) {
      for (std::size_t i = 0; i < m.size(); ++i) {
        out << m.param(i) << std::endl;
      }
      return out;
    }

    // Queries
    //--------------------------------------------------------------------------

    //! Evaluates the mixture for the given vector.
    result_type operator()(const index_type& index) const {
      result_type result(0);
      for (std::size_t i = 0; i < size(); ++i) {
        result += factor(i)(index);
      }
      return result;
    }

    //! Returns the log-value of the factor for the given vector.
    real_type log(const index_type& vec) const {
      using std::log;
      return log(operator()(vec));
    }

    // Aggregation
    //--------------------------------------------------------------------------

    /**
     * Return a marginal of the mixture over a contiguous range of dimensions.
     */
    mixture<F> marginal(std::size_t start, std::size_t n = 1) const {
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
      for (std::size_t i = 0; i < size(); ++i) {
        result += factor(i).sum();
      }
      return result;
    }

    /**
     * Returns the normalization constants of all the components.
     */
    std::vector<result_type> sums() const {
      std::vector<result_type> result;
      for (std::size_t i = 0; i < size(); ++i) {
        result[i] = factor(i).sum();
      }
      return result;
    }

    /**
     * Returns true if the mixture is normalizable.
     */
    bool normlizable() const {
      return sum() > result_type(0);
    }

    // Conditioning
    //--------------------------------------------------------------------------

    /**
     * Returns a mixture where a contiguous range of dimensions has been
     * restricted to given values.
     */
    mixture restrict(std::size_t start, std::size_t n,
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

    /**
     * Draws a random component from this mixture.
     */
    template <typename Generator>
    std::size_t sample_component(Generator& rng) const {
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

    // Selectors
    //--------------------------------------------------------------------------

    /**
     * Returns a mixture selector referencing the head dimensions of each
     * component.
     */
    mixture_selector<span, const mixture>
    head(std::size_t n) const {
      return { front(n), *this };
    }

    /**
     * Returns a mutable mixture selector referencing the head dimensions of
     * each component.
     */
    mixture_selector<span, mixture>
    head(std::size_t n) {
      return { front(n), *this };
    }

    /**
     * Returns a mixture selector referencing the tail dimensions of each
     * component.
     */
    mixture_selector<span, const mixture>
    tail(std::size_t n) const {
      return { back(arity(), n), *this };
    }

    /**
     * Returns a mutable mixture selector referencing the tail dimensions of
     * each component.
     */
    mixture_selector<span, mixture>
    tail(std::size_t n) {
      return { back(arity(), n), *this };
    }

    /**
     * Returns a mixture selector referencing a single dimension of each
     * component.
     */
    mixture_selector<std::size_t, const mixture>
    dim(std::size_t index) const {
      return { index, *this };
    }

    /**
     * Returns a mutable mixture selector referencing a single dimension of
     * each component.
     */
    mixture_selector<std::size_t, mixture>
    dim(std::size_t index) {
      return { index, *this };
    }

    /**
     * Returns a mixture selector referencing a contiguous range of dimensions
     * of each component.
     */
    mixture_selector<span, const mixture>
    dims(std::size_t start, std::size_t n) const {
      return { span(start, n), *this };
    }

    /**
     * Returns a mutabl emixture selector referencing a contiguous range of
     * dimensions of each component.
     */
    mixture_selector<span, mixture>
    dims(std::size_t start, std::size_t n) {
      return { span(start, n), *this };
    }

    /**
     * Returns a mixture selector referencing a subset of dimensions of each
     * component.
     */
    mixture_selector<const uint_vector&, const mixture>
    dims(const uint_vector& indices) const {
      return { indices, *this };
    }

    /**
     * Returns a mixture selector referencing a subset of dimensions of each
     * component.
     */
    mixture_selector<const uint_vector&, mixture>
    dims(const uint_vector& indices) {
      return { indices, *this };
    }

    // Mutations
    //--------------------------------------------------------------------------

    //! Multiplies each component by a constant.
    mixture& operator*=(result_type x) {
      return update_components(multiplied_by<result_type>(x));
    }

    //! Divides each component by a constant.
    mixture& operator/=(result_type x) {
      return update_components(divided_by<result_type>(x));
    }

    //! Multiplies each component by a factor in place.
    LIBGM_ENABLE_IF(has_multiplies_assign<F>::value)
    mixture& operator*=(const F& f) {
      return update_components(multiplied_by<const F&>(f));
    }

    //! Divides each component by a factor in place.
    LIBGM_ENABLE_IF(has_divides_assign<F>::value)
    mixture& operator/=(const F& f) {
      return update_components(divided_by<const F&>(f));
    }

    //! Normalizes the distribution represented by this mixture.
    void normalize() {
      *this /= sum();
    }

  private:

    /**
     * Applies the given update operation to each component.
     */
    template <typename Op>
    mixture& update_components(Op op) {
      for (std::size_t i = 0; i < size(); ++i) {
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
      for (std::size_t i = 0; i < size(); ++i) {
        result[i] = op(components_[i]);
      }
      return result;
    }

    //! The mixture components.
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
