#ifndef LIBGM_MIXTURE_HPP
#define LIBGM_MIXTURE_HPP

#include <libgm/factor/base/factor.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/math/random/mixture_distribution.hpp>
#include <libgm/serialization/vector.hpp>

namespace libgm {

  /**
   * A factor representing a finite weighted mixture (sum) of distributions.
   *
   * \tparam F The factor type representing each mixture component.
   *           Must model the ParametricFactor concept and support prototypes.
   *
   * \ingroup factor_types
   */
  template <typename F>
  class mixture : public factor {
  public:
    // Public types
    //==========================================================================

    // Factor member types
    typedef typename F::real_type       real_type;
    typedef typename F::result_type     result_type;
    typedef typename F::argument_type   argument_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;

    // ParametricFactor member types
    typedef std::vector<typename F::param_type> param_type;
    typedef typename F::vector_type              vector_type;
    typedef mixture_distribution<typename F::distribution_type>
      distribution_type;

    // Constructors and conversion operators
    //==========================================================================

    //! Default constructor. Creates an empty factor.
    mixture() { }

    /**
     * Constructs a mixture with given number of components and arguments.
     * Allocates but does not initialize the parameters.
     */
    explicit mixture(std::size_t k, const domain_type& args)
      : prototype_(args), param_(k, prototype_.param()) {
      typename F::param_type(std::move(prototype_.param())); // free memory
    }

    /**
     * Constructs a mixture with k identical components equal to
     * the specified factor.
     */
    mixture(std::size_t k, const F& factor)
      : prototype_(factor), param_(k, factor.param()) {
      typename F::param_type(std::move(prototype_.param())); // free memory
    }

    /**
     * Constructs a mixture with a single component equal to
     * the specified factor.
     */
    explicit mixture(const F& factor)
      : prototype_(factor) {
      param_.push_back(std::move(prototype_.param()));
    }

    /**
     * Constructs a mixture equivalent to a constant.
     * The mixture has exactly one component and no arguments.
     */
    explicit mixture(result_type value)
      : prototype_(value) {
      param_.push_back(std::move(prototype_.param()));
    }

    //! Assigns a single component to this factor.
    mixture& operator=(const F& factor) {
      prototype_ = factor;
      param_.clear();
      param_.push_back(std::move(prototype_.param()));
      return *this;
    }

    //! Assigns a constant to this factor.
    mixture& operator=(result_type value) {
      prototype_ = value;
      param_.clear();
      param_.push_back(std::move(prototype_.param()));
      return *this;
    }

    //! Exchanges the contents of two mixtures.
    friend void swap(mixture& f, mixture& g) {
      swap(f.prototype_, g.prototype_);
      swap(f.param_, g.param_);
    }

    // Serialization
    //==========================================================================

    //! Serializes the factor to an archive.
    void save(oarchive& ar) const {
      ar << prototype_ << param_;
    }

    //! Deserializes the factor from an archive.
    void load(iarchive& ar) const {
      ar >> prototype_ >> param_;
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the arguments of this mixture.
    const domain_type& arguments() const {
      return prototype_.arguments();
    }

    //! Returns the number of arguments of this mixture.
    std::size_t arity() const {
      return prototype_.arity();
    }

    //! Returns true if the mixture is empty (has 0 components).
    bool empty() const {
      return param_.empty();
    }

    //! Returns the number of components of this mixture.
    std::size_t size() const {
      return param_.size();
    }

    //! Returns the prototype factor.
    const F& prototype() const {
      return prototype_;
    }

    //! Returns the parameter vector.
    param_type& param() {
      return param_;
    }

    //! Returns the parameter vector.
    const param_type& param() const {
      return param_;
    }

    //! Returns the parameters associated with component i.
    typename F::param_type& param(std::size_t i) {
      return param_[i];
    }

    //! Returns the parameters associated with component i.
    const typename F::param_type& param(std::size_t i) const {
      return param_[i];
    }

    //! Returns true if the two mixtures have the same domains and parameters.
    bool operator==(const mixture& other) const {
      return prototype_.arguments() == other.prototype_.arguments()
        && param_ == other.param_;
    }

    //! Returns true if the mixtures do not have the same domain or parameters.
    bool operator!=(const mixture& other) const {
      return !(*this == other);
    }

    // Indexing
    //==========================================================================

    /**
     * Converts the given index to an assignment over head variables.
     */
    void assignment(const vector_type& index, assignment_type& a) const {
      prototype_.assignment(index, a);
    }

    /**
     * Substitutes the arguments in-place according to the given map.
     */
    void subst_args(const std::unordered_map<argument_type, argument_type>& m) {
      prototype_.subst_args(m);
    }

    // Factor evaluation
    //==========================================================================

    //! Evaluates the factor for an assignment.
    result_type operator()(const assignment_type& a) const {
      return operator()(extract(a, arguments()));
    }

    //! Evaluates the factor for the given index.
    result_type operator()(const vector_type& index) const {
      result_type result(0);
      for (const auto& p : param_) {
        result += p(index);
      }
      return ;
    }

    //! Returns the log-value of the factor for an assignment.
    real_type log(const assignment_type& a) const {
      return log(extract(a, arguments()));
    }

    //! Returns the log-value of the factor for an index.
    real_type log(const vector_type& index) const {
      using std::log;
      return log(operator()(index));
    }

    // Factor operations
    //==========================================================================

    //! Multiplies a mixture by a constant in-place component-wise.
    friend mixture& operator*=(mixture& h, result_type val) {
      auto op = multiplies_assign_op(h.prototype_);
      for (auto& p : h.param_) { op(p, val); }
      return h;
    }

    //! Divides a mixture by a constant in-place component-wise.
    friend mixture& operator/=(mixture& h, result_type val) {
      auto op = divides_assign_op(h.prototype_);
      for (auto& p : h.param_) { op(p, val); }
      return h;
    }

    //! Multiplies a mixture by a factor in-place component-wise.
    friend mixture& operator*=(mixture& h, const F& f) {
      auto op = multiplies_assign_op(h.prototype_, f);
      for (auto& p : h.param_) { op(p, f.param()); }
      return h;
    }

    //! Divides a mixture by a factor in-place component-wise.
    friend mixture& operator/=(mixture& h, const F& f) {
      auto op = divides_assign_op(h.prototype_, f);
      for (auto& p : h.param_) { op(p, f.param()); }
      return h;
    }

    //! Multiplies a mixture by a constant component-wise.
    friend mixture operator*(mixture h, result_type val) {
      auto op = multiplies_assign_op(h.prototype_);
      for (auto& p : h.param_) { op(p, val); }
      return h;
    }

    //! Multiplies a mixture by a constant component-wise.
    friend mixture operator*(result_type val, mixture h) {
      auto op = multiplies_assign_op(h.prototype_);
      for (auto& p : h.param_) { op(p, val); }
      return h;
    }

    //! Divides a mixture by a constant component-wise.
    friend mixture operator/(mixture h, result_type val) {
      auto op = divides_assign_op(h.prototype_);
      for (auto& p : h.param_) { op(p, val); }
      return h;
    }

    //! Multiplies a mixture by a factor component-wise.
    friend mixture operator*(const mixture& h, const F& f) {
      mixture result(h.size());
      auto op = multiplies_op(h.prototype_, f, result.prototype_);
      for (std::size_t i = 0; i < h.size(); ++i) {
        op(h.param(i), f.param(), result.param(i));
      }
      return result;
    }

    //! Multiplies a factor by a mixture component-wise.
    friend mixture operator*(const F& f, const mixture& h) {
      mixture result(h.size());
      auto op = multiplies_op(f, h.prototype_, result.prototype_);
      for (std::size_t i = 0; i < h.size(); ++i) {
        op(f.param(), h.param(i), result.param(i));
      }
      return result;
    }

    //! Divides a mixture by a factor component-wise.
    friend mixture operator/(const mixture& h, const F& f) {
      mixture result(h.size());
      auto op = divides_op(h.prototype_, f, result.prototype_);
      for (std::size_t i = 0; i < h.size(); ++i) {
        op(h.param(i), f.param(), result.param(i));
      }
      return result;
    }

    //! Multiplies two mixtures together.
    friend mixture operator*(const mixture& f, const mixture& g) {
      mixture result(f.size() * g.size());
      auto op = multiplies_op(f.prototype_, g.prototype_, result.prototype_);
      for (std::size_t i = 0; i < f.size(); ++i) {
        for (std::size_t j = 0; j < g.size(); ++j) {
          op(f.param(i), g.param(j), result.param(i * g.size() + j));
        }
      }
      return result;
    }

    //! Multiplies two mixtures component-wise.
    friend mixture operator%(const mixture& f, const mixture& g) {
      assert(f.size() == g.size());
      mixture result(f.size());
      auto op = multiplies_op(f.prototype_, g.prototype_, result.prototype_);
      for (std::size_t i = 0; i < f.size(); ++i) {
        op(f.param(i), g.param(i), result.param(i));
      }
      return result;
    }

    /**
     * Computes a marginal of the mixture over a sequence of variables.
     * \throws invalid_argument if retained is not a subset of arguments
     */
    mixture marginal(const domain_type& retain) const {
      mixture result;
      maginal(retain, result);
      return result;
    }

    /**
     * Computes a marginal of the mixture over a sequence of variables.
     * \throws invalid_argument if retained is not a subset of arguments
     */
    void marginal(const domain_type& retain, mixture& result) const {
      auto op = marginal_op(prototype_, retain, result.prototype_);
      result.param_.resize(size());
      for (std::size_t i = 0; i < size(); ++i) {
        op(param(i), result.param(i));
      }
    }

    /**
     * Returns the normalization constant of the factor.
     */
    result_type marginal() const {
      result_type result(0);

    }

    //! implements DistributionFactor::normalize
    //! uses the standard parameterization
    mixture& normalize() {
      return (*this /= marginal());
    }

    //! implements DistributionFactor::is_normalizable
    bool normalizable() const {
      foreach(const F& factor, comps)
        if (factor.is_normalizable()) return true;
      return false;
    }

    //! implements Factor::restrict
    mixture restrict(const assignment_type& a) const {
      domain_type bound_vars = keys(a);

      // If the arguments are disjoint from the bound variables,
      // we can simply return a copy of the factor
      if (set_disjoint(bound_vars, arguments()))
        return *this;

      // Restrict each component factor
      domain_type retained = set_difference(arguments(), bound_vars);
      mixture factor(size(), retained);

      for(size_t i = 0; i < size(); i++)
        factor.comps[i] = comps[i].restrict(a);

      return factor;
    }

    // Sampling
    //==========================================================================

    //! Returns the distribution with the parameters of this factor.
    distribution_type  distribution() const {
      return multivariate_normal_distribution<T>(param_);

    // Private data members
    //==========================================================================
  private:

    //! The mixture components
    std::vector<F> comps;

  }; // class mixture

  //! \relates mixture
  template <typename F>
  std::ostream& operator<<(std::ostream& out, const mixture<F>& mixture) {
    out << "#F(M | " << mixture.arguments();
    foreach(const F& factor, mixture.components())
      out << "\n | " << factor;
    out << ")\n";
    return out;
  }

  // Free functions
  //============================================================================

  //! Computes the KL projection of a mixture of Gaussians to a Gausian
  //! using moment matching.
  //! \relates mixture
  moment_gaussian project(const mixture_gaussian& mixture) {
    // Do moment matching
    vector_var_vector args = make_vector(mixture.arguments());
    size_t n = mixture[0].size();
    double norm = mixture.norm_constant();
    assert(norm > 0);

    // Match the mean
    vec mean = zeros(n);
    // std::cout << mean << std::endl;
    for(size_t i = 0; i < mixture.size(); i++) {
      double w = mixture[i].norm_constant() / norm;
      mean += w * mixture[i].mean(args);
      // std::cout << mean << std::endl;
    }

    // Match the covariance
    mat cov = zeros(n, n);
    for(size_t i = 0; i < mixture.size(); i++) {
      double w = mixture[i].norm_constant() / norm;
      vec x = mixture[i].mean(args) - mean;
      cov += w * (mixture[i].covariance(args) + outer_product(x, x));
    }
    return moment_gaussian(args, mean, cov, norm);
  }

} // namespace libgm

#endif
