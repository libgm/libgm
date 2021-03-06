#ifndef LIBGM_CANONICAL_GAUSSIAN_HPP
#define LIBGM_CANONICAL_GAUSSIAN_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/argument/real_assignment.hpp>
#include <libgm/factor/base/gaussian_factor.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/param/canonical_gaussian_param.hpp>

namespace libgm {

  // forward declaration
  template <typename Arg, typename T> class moment_gaussian;

  /**
   * A factor of a Gaussian distribution in the natural parameterization.
   *
   * \tparam T the real type for representing the parameters.
   * \ingroup factor_types
   */
  template <typename Arg, typename T = double>
  class canonical_gaussian : public gaussian_factor<Arg> {
    typedef argument_traits<Arg> arg_traits;

  public:
    // Public types
    //==========================================================================
    // Base type
    typedef gaussian_factor<Arg> base;

    // Underlying storage
    typedef dense_matrix<T> mat_type;
    typedef dense_vector<T> vec_type;

    // Factor member types
    typedef T                       real_type;
    typedef logarithmic<T>          result_type;
    typedef Arg                     argument_type;
    typedef domain<Arg>             domain_type;
    typedef real_assignment<Arg, T> assignment_type;

    // ParametricFactor member types
    typedef canonical_gaussian_param<T> param_type;
    typedef dense_vector<T>              vector_type;
    typedef uint_vector    index_type;

    // ExponentialFamilyFactor member types
    typedef moment_gaussian<Arg, T> probability_type;

    // Constructors and conversion operators
    //==========================================================================

    //! Default constructor. Creats an empty factor.
    canonical_gaussian() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit canonical_gaussian(const domain_type& args) {
      reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit canonical_gaussian(logarithmic<T> value)
      : param_(0, value.lv) { }

    //! Constructs a factor with given arguments and constant value.
    canonical_gaussian(const domain_type& args, logarithmic<T> value)
      : base(args), args_(args), param_(args.num_dimensions(), value.lv) { }

    //! Constructs a factor with the given arguments and parameters.
    canonical_gaussian(const domain_type& args, const param_type& param)
      : base(args), args_(args), param_(param) {
      check_param();
    }

    //! Constructs a factor with the given arguments and parameters.
    canonical_gaussian(const domain_type& args, param_type&& param)
      : base(args), args_(args), param_(std::move(param)) {
      check_param();
    }

    //! Constructs a factor with the given arguments and parameters.
    canonical_gaussian(const domain_type& args,
                       const vec_type& eta,
                       const mat_type& lambda,
                       T lv = T(0))
      : base(args), args_(args), param_(eta, lambda, lv) {
      check_param();
    }

    //! Conversion from a moment_gaussian
    explicit canonical_gaussian(const moment_gaussian<Arg, T>& mg) {
      *this = mg;
    }

    //! Assigns a constant to this factor.
    canonical_gaussian& operator=(logarithmic<T> value) {
      reset();
      param_.lm = value.lv;
      return *this;
    }

    //! Assigns a moment_gaussian to this factor.
    canonical_gaussian& operator=(const moment_gaussian<Arg, T>& mg) {
      reset(mg.arguments());
      param_ = mg.param();
      return *this;
    }

    //! Casts this canonical_gaussian to a moment_gaussian.
    moment_gaussian<Arg, T> moment() const {
      return moment_gaussian<Arg, T>(*this);
    }

    //! Exchanges the content of two factors.
    friend void swap(canonical_gaussian& f, canonical_gaussian& g) {
      using std::swap;
      f.base_swap(g);
      swap(f.args_, g.args_);
      swap(f.param_, g.param_);
    }

    // Serialization and initialization
    //==========================================================================

    //! Serializes the factor to an archive.
    void save(oarchive& ar) const {
      ar << args_ << param_;
    }

    //! Deserializes the factor from an archive.
    void load(iarchive& ar) {
      ar >> args_ >> param_;
      this->compute_start(args_);
      check_param();
    }

    //! Sets the arguments to the given domain and allocates the memory.
    void reset(const domain_type& args = domain_type()) {
      if (args_ != args) {
        args_ = args;
        std::size_t n = this->compute_start(args);
        param_.resize(n);
      }
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the arguments of this factor.
    const domain_type& arguments() const {
      return args_;
    }

    //! Returns the number of arguments of this factor.
    std::size_t arity() const {
      return args_.size();
    }

    //! Returns true if the factor is empty.
    bool empty() const {
      return args_.empty();
    }

    //! Returns the number of dimensions of this Gaussian.
    std::size_t size() const {
      return param_.size();
    }

    //! Returns the parameter struct. The caller must not alter its size.
    param_type& param() {
      return param_;
    }

    //! Returns the parameter struct.
    const param_type& param() const {
      return param_;
    }

    //! Returns the log multiplier.
    T log_multiplier() const {
      return param_.lm;
    }

    //! Returns the information vector.
    const vec_type& inf_vector() const {
      return param_.eta;
    }

    //! Returns the information matrix.
    const mat_type& inf_matrix() const {
      return param_.lambda;
    }

    /**
     * Returns the information subvector for a single argument.
     * Supported for multivariate arguments.
     */
    Eigen::VectorBlock<const vec_type> inf_vector(Arg v) const {
      std::size_t n = arg_traits::num_dimensions(v);
      return param_.eta.segment(this->start_.at(v), n);
    }

    /**
     * Returns the information submatrix for a single argument.
     * Supported for multivariate arguments.
     */
    Eigen::Block<const mat_type> inf_matrix(Arg v) const {
      std::size_t i = this->start_.at(v);
      std::size_t n = arg_traits::num_dimensions(v);
      return param_.lambda.block(i, i, n, n);
    }

    //! Returns the information vector for a subset of the arguments
    vec_type inf_vector(const domain_type& args) const {
      index_type index = args.index(this->start_);
      return subvec(param_.eta, iref(index));
    }

    //! Returns the information matrix for a subset of the arguments
    mat_type inf_matrix(const domain_type& args) const {
      index_type index = args.index(this->start_);
      return submat(param_.eta, iref(index), iref(index));
    }

    //! Returns true of the two factors have the same domains and parameters.
    bool operator==(const canonical_gaussian& other) const {
      return args_ == other.args_ && param_ == other.param_;
    }

    //! Returns true if the two factors do not have the same domains or params.
    bool operator!=(const canonical_gaussian& other) const {
      return !(*this == other);
    }

    // Indexing
    //==========================================================================

    /**
     * Converts the given vector to an assignment.
     */
    void assignment(const vec_type& vec, assignment_type& a) const {
      a.insert_or_assign(args_, vec);
    }

    /**
     * Substitutes the arguments in-place according to the given map.
     */
    void subst_args(const std::unordered_map<Arg, Arg>& map) {
      base::subst_args(map);
      args_.substitute(map);
    }

    /**
     * Reorders the arguments according to the given domain.
     */
    canonical_gaussian reorder(const domain_type& args) const {
      if (!equivalent(args, args_)) {
        throw std::runtime_error(
          "canonical_gaussian::reorder: ordering changes the argument set"
        );
      }
      canonical_gaussian result(args);
      param_.reorder(iref(args.index(this->start_)), result.param_);
      return result;
    }

    /**
     * Checks if the size of the parameter struct matches this factor's
     * arguments.
     * \throw std::invalid_argument if the sizes do not match
     */
    void check_param() const {
      param_.check();
      if (param_.size() != args_.num_dimensions()) {
        throw std::runtime_error("canonical_gaussian: Invalid parameter size");
      }
    }

    /**
     * Checks if two factors have the same (sequence of) arguments.
     * \throw std::invalid_argument if the arguments do not match
     */
    friend void check_same_arguments(const canonical_gaussian& f,
                                     const canonical_gaussian& g) {
      if (f.arguments() != g.arguments()) {
        throw std::invalid_argument(
          "canonical_gaussian: incompatible arguments"
        );
      }
    }

    // Factor evaluation
    //==========================================================================

    //! Evaluates the factor for an assignment.
    logarithmic<T> operator()(const assignment_type& a) const {
      return logarithmic<T>(log(a), log_tag());
    }

    //! Evaluates the factor for a vector.
    logarithmic<T> operator()(const vec_type& x) const {
      return logarithmic<T>(log(x), log_tag());
    }

    //! Returns the log-value of the factor for an assignment.
    T log(const assignment_type& a) const {
      return param_(a.values(args_));
    }

    //! Returns the log-value of the factor for a vector.
    T log(const vec_type& x) const {
      return param_(x);
    }

    // Factor operations (the parameter operation objects are computed below)
    //==========================================================================

    //! Multiplies this factor by a constant in-place.
    canonical_gaussian& operator*=(logarithmic<T> x) {
      param_.lm += x.lv;
      return *this;
    }

    //! Divides this factor by a constant in-place.
    canonical_gaussian& operator/=(logarithmic<T> x) {
      param_.lm -= x.lv;
      return *this;
    }

    //! Multiplies this factor by another one in-place.
    canonical_gaussian& operator*=(const canonical_gaussian& f) {
      param_.update(plus_assign<>(),
                    iref(f.arguments().index(this->start_)),
                    f.inf_vector(),
                    f.inf_matrix(),
                    f.log_multiplier());
      return *this;
    }

    //! Divides this factor by another one in-place.
    canonical_gaussian& operator/=(const canonical_gaussian& f) {
      param_.update(minus_assign<>(),
                    iref(f.arguments().index(this->start_)),
                    f.inf_vector(),
                    f.inf_matrix(),
                    f.log_multiplier());
      return *this;
    }

    //! Multiplies a canonical_gaussian by a constant.
    friend canonical_gaussian operator*(canonical_gaussian f, logarithmic<T> x) {
      f.param_.lm += x.lv;
      return f;
    }

    //! Multiplies a canonical_gaussian by a constant.
    friend canonical_gaussian operator*(logarithmic<T> x, canonical_gaussian f) {
      f.param_.lm += x.lv;
      return f;
    }

    //! Divides a canonical_gaussian by a constant.
    friend canonical_gaussian operator/(canonical_gaussian f, logarithmic<T> x) {
      f.param_.lm -= x.lv;
      return f;
    }

    //! Divides a constant by a canonical_gaussian.
    friend canonical_gaussian operator/(logarithmic<T> x, canonical_gaussian f) {
      f.param_.eta = -f.param_.eta;
      f.param_.lambda = -f.param_.lambda;
      f.param_.lm = x.lv - f.param_.lm;
      return f;
    }

    //! Raises a canonical_gaussian to an exponent.
    friend canonical_gaussian pow(canonical_gaussian f, T x) {
      f.param_ *= x;
      return f;
    }

    //! Returns \f$f^{(1-a)} * g^a\f$.
    friend canonical_gaussian
    weighted_update(const canonical_gaussian& f,
                    const canonical_gaussian& g, T a) {
      check_same_arguments(f, g);
      return canonical_gaussian(f.args_, weighted_sum(f.param_, g.param_, a));
    }

    //! Multiplies two canonical_gaussian factors.
    friend canonical_gaussian
    operator*(const canonical_gaussian& f, const canonical_gaussian& g) {
      canonical_gaussian result(f.arguments() + g.arguments(), logarithmic<T>());
      result *= f;
      result *= g;
      return result;
    }

    //! Divides two canonical_gaussian factors.
    friend canonical_gaussian
    operator/(const canonical_gaussian& f, const canonical_gaussian& g) {
      canonical_gaussian result(f.arguments() + g.arguments(), logarithmic<T>());
      result *= f;
      result /= g;
      return result;
    }

    /**
     * Computes the aggregate of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     * \throws numerical_error if the information matrix over the
     *         mariginalized variables is singular.
     */
    canonical_gaussian aggregate(bool marginal,
                                 const domain_type& retain) const {
      canonical_gaussian result(retain);
      typename param_type::collapse_workspace ws;
      param_.collapse(marginal,
                      iref(retain.index(this->start_)),
                      iref((args_ - retain).index(this->start_)),
                      ws,
                      result.param_);
      return result;
    }

    /**
     * Computes the marginal of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     * \throws numerical_error if the information matrix over the
     *         mariginalized variables is singular.
     */
    canonical_gaussian marginal(const domain_type& retain) const {
      return aggregate(true, retain);
    }

    /**
     * Computes the maximum of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     * \throws numerical_error if the information matrix is singular
     */
    canonical_gaussian maximum(const domain_type& retain) const {
      return aggregate(false, retain);
    }

    //! Returns the normalization constant of the factor.
    logarithmic<T> marginal() const {
      return logarithmic<T>(param_.marginal(), log_tag());
    }

    //! Returns the maximum value in the factor.
    logarithmic<T> maximum() const {
      return logarithmic<T>(param_.maximum(), log_tag());
    }

    //! Computes the maximum value and stores the corresponding assignment.
    logarithmic<T> maximum(assignment_type& a) const {
      vec_type vec;
      T max = param_.maximum(vec);
      a.insert_or_assign(args_, vec);
      return logarithmic<T>(max, log_tag());
    }

    //! Normalizes the factor in-place.
    canonical_gaussian& normalize() {
      param_.lm -= param_.marginal();
      return *this;
    }

    //! Returns true if the factor is normalizable.
    bool normalizable() const {
      return std::isfinite(param_.marginal());
    }

    //! If this factor represents p(x, y), returns p(x | y).
    canonical_gaussian conditional(const domain_type& tail) const {
      return *this / marginal(tail);
    }

    //! Restricts the factor to the given assignment.
    canonical_gaussian restrict(const assignment_type& a) const {
      domain<Arg> y, x; // restricted, retained
      args_.partition(a, y, x);
      canonical_gaussian result(x);
      param_.restrict(iref(x.index(this->start_)),
                      iref(y.index(this->start_)),
                      a.values(y),
                      result.param_);
      return result;
    }

    // Entropy and divergences
    //==========================================================================

    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return param_.entropy();
    }

    //! Computes the entropy for a subset of variables.
    T entropy(const domain_type& dom) const {
      if (equivalent(args_, dom)) {
        return entropy();
      } else {
        return marginal(dom).entropy();
      }
    }

    //! Computes the mutual information bewteen two subsets of arguments.
    T mutual_information(const domain_type& a, const domain_type& b) const {
      return entropy(a) + entropy(b) - entropy(a + b);
    }

    //! Computes the Kullback-Liebler divergence from p to q.
    friend T kl_divergence(const canonical_gaussian& p,
                           const canonical_gaussian& q) {
      check_same_arguments(p, q);
      return kl_divergence(p.param_, q.param_);
    }

    friend T max_diff(const canonical_gaussian& f,
                      const canonical_gaussian& g) {
      check_same_arguments(f, g);
      return max_diff(f.param_, g.param_);
    }

    // Private data members
    //==========================================================================
  private:
    //! The sequence of arguments of the factor.
    domain_type args_;

    //! The parameters of the factor.
    param_type param_;

  }; // class canonical_gaussian


  // Common operations
  //============================================================================

  /**
   * Prints the canonical_gaussian to a stream.
   * \relates canonical_gaussian
   */
  template <typename Arg, typename T>
  std::ostream&
  operator<<(std::ostream& out, const canonical_gaussian<Arg, T>& f) {
    out << f.arguments() << std::endl
        << f.param() << std::endl;
    return out;
  }

} // namespace libgm

#endif
