#ifndef LIBGM_EXPERIMENTAL_MIXTURE_HPP
#define LIBGM_EXPERIMENTAL_MIXTURE_HPP

#include <libgm/enable_if.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/substituted_param.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/math/random/mixture_distribution.hpp>
#include <libgm/serialization/vector.hpp>

#include <vector>

namespace libgm { namespace experimental {

  /**
   * A model representing a finite mixture (sum) of distributions.
   *
   * \tparam F The factor type representing each mixture component.
   *           Must model the ParametricFactor concept and support expressions.
   *
   * \ingroup model
   */
  template <typename F>
  class mixture {
  public:
    // Public type declarations
    //--------------------------------------------------------------------------

    // Factor shortcuts
    typedef typename F::argument_type   argument_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef typename F::real_type       real_type;
    typedef typename F::result_type     result_type;

    // ParametricFactor shortcuts
    typedef typename F::param_type  param_type;
    typedef typename F::vector_type vector_type;

    // Constructors and initialization
    //--------------------------------------------------------------------------

    /**
     * Constructs an empty mixture.
     */
    mixture() { }

    /**
     * Constructs a mixture with the specified arguments and initializes the
     * parameter vector to the given number of (empty) components.
     */
    explicit mixture(const domain_type& args, std::size_t k = 0)
      : prototype_(args), param_(k, prototype_.param()) { }

    /**
     * Constructs a mixture with components equivalent to the given factor.
     */
    explicit mixture(const F& factor, std::size_t k = 1)
      : prototype_(factor), param_(k, factor.param()) { }

    //! Exchanges the content of two factors.
    friend void swap(mixture& m, mixture& n) {
      swap(m.prototype_, n.prototype_);
      swap(m.param_, n.param_);
    }

    //! Resets the mixture to the given arguments and number of components.
    void reset(const domain_type& args, std::size_t k) {
      prototype_.reset(args);
      param_.resize(k);
    }

    //! Serializes the model to an archive.
    void save(oarchive& ar) const {
      ar << prototype_ << param_;
    }

    //! Deserializes the model from an archive.
    void load(iarchive& ar) {
      ar >> prototype_ >> param_;
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the arguments of this mixture.
    const domain_type& arguments() const {
      return prototype_.arguments();
    }

    //! Returns the number of arguments of this mixture.
    std::size_t arity() const {
      return prototype_.arity();
    }

    //! Returns true if the mixture has no components.
    bool empty() const {
      return param_.empty();
    }

    //! Returns the number of components of this mixture.
    std::size_t size() const {
      return param_.size();
    }

    //! Returns the parameter vector.
    std::vector<param_type>& param() {
      return param_;
    }

    //! Returns the parameter vector.
    const std::vector<param_type>& param() const {
      return param_;
    }

    //! Returns the parameters associated with the given component index.
    param_type& param(std::size_t i) {
      return param_[i];
    }

    //! Returns the parameters associated with the given component index.
    const param_type& param(std::size_t i) const {
      return param_[i];
    }

    //! Returns the factor expression representing the given component.
    substituted_param<F> factor(std::size_t i) {
      return { prototype_, param_[i] };
    }

    //! Returns the factor expression representing the given component.
    substituted_param<const F> factor(std::size_t i) const {
      return { prototype_, param_[i] };
    }

    //! Returns the shape of the parameters for each component.
    auto param_shape() const {
      return F::param_shape(prototype_.arguments());
    }

    //! Outputs a human-readable representation of the mixture to a stream.
    friend std::ostream& operator<<(std::ostream& out, const mixture& m) {
      out << m.arguments() << std::endl;
      for (std::size_t i = 0; i < m.size(); ++i) {
        out << m.param(i) << std::endl;
      }
      return out;
    }

    // Queries
    //--------------------------------------------------------------------------

    //! Evaluates the mixture for the given vector.
    result_type operator()(const vector_type& vec) const {
      result_type result(0);
      for (std::size_t i = 0; i < size(); ++i) {
        result += factor(i)(vec);
      }
      return result;
    }

    //! Evaluates the mixture for the given assignment.
    result_type operator()(const assignment_type& a) const {
      return operator()(a.values(arguments()));
    }

    //! Returns the log-value of the factor for the given vector.
    real_type log(const vector_type& vec) const {
      using std::log;
      return log(operator()(vec));
    }

    //! Returns the log-value of the factor for the given assignment.
    real_type log(const assignment_type& a) const {
      using std::log;
      return log(operator()(a));
    }

    // Queries
    //--------------------------------------------------------------------------

    /**
     * Returns the normalization constant of the mixture.
     */
    result_type marginal() const {
      result_type result(0);
      for (std::size_t i = 0; i < size(); ++i) {
        result += factor(i).marginal();
      }
      return result;
    }

    /**
     * Returns the normalization constants of all the components.
     */
    std::vector<result_type> marginals() const {
      std::vector<result_type> result;
      for (std::size_t i = 0; i < size(); ++i) {
        result[i] = factor(i).marginal();
      }
      return result;
    }

    /**
     * Return a marginal of the mixture to a subset of arguments.
     */
    LIBGM_ENABLE_IF(has_marginal<F>::value)
    mixture<F> marginal(const domain_type& dom) const {
      substituted_param<const F> f(prototype_);
      return componentwise(f, f.marginal(dom));
    }

    /**
     * Returns a restriction of the mixture to an assignment.
     */
    LIBGM_ENABLE_IF(has_restrict<F>::value)
    mixture<F> restrict(const assignment_type& a) const {
      substituted_param<const F> f(prototype_);
      return componentwise(f, f.restrict(a));
    }

    /**
     * Projects the mixture to a single component.
     */
    friend F kl_project(const mixture& m) {
      return F(m.args_, kl_project(m.param_));
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Draws a random component from this mixture.
     */
    template <typename Generator>
    std::size_t sample_component(Generator& rng) const {
      //return vector_distribution<real_type, 1>(marginals()).sample(rng);
      assert(false);
    }

    /**
     * Draws a random sample from the distribution represented by this mixture.
     */
    LIBGM_ENABLE_IF_D((has_sample<F, Generator>::value), typename Generator)
    vector_type sample(Generator& rng) const {
      return factor(sample_component(rng)).sample(rng);
    }

    /**
     * Returns the distribution for this mixture.
     */
    LIBGM_ENABLE_IF(has_distribution<F>::value)
    auto distribution() const {
      return mixture_distribution<typename F::distribution_type>(param_);
    }

    // Mutations
    //--------------------------------------------------------------------------

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
      *this /= marginal();
    }

  private:
    /**
     * Returns the result of an operation applied to all components.
     *
     * \param f    a reference to the component
     * \param expr the expression object evaluating the component f
     */
    template <typename Derived>
    mixture<F> componentwise(substituted_param<const F>& f,
                             const base_t<F, Derived>& expr) const {
      mixture<F> result(expr.derived().arguments(), size());
      for (std::size_t i = 0; i < size(); ++i) {
        f.reset(param(i));
        result.param(i) = expr.derived().param();
      }
      return result;
    }

    //! The prototype of the factor holding the arguments.
    F prototype_;

    //! The parameters of the mixture.
    std::vector<param_type> param_;

  }; // class mixture

} } // namespace libgm::experimental

#endif
