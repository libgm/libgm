#ifndef LIBGM_MOMENT_GAUSSIAN_HPP
#define LIBGM_MOMENT_GAUSSIAN_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/real_assignment.hpp>
#include <libgm/factor/base/gaussian_factor.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/likelihood/moment_gaussian_mle.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>
#include <libgm/math/random/multivariate_normal_distribution.hpp>

namespace libgm {

  // forward declaration
  template <typename Arg, typename T> class canonical_gaussian;

  /**
   * A factor of a Gaussian distribution in the moment parameterization.
   *
   * \tparam T the real type for representing the parameters.
   * \ingroup factor_types
   */
  template <typename Arg, typename T = double >
  class moment_gaussian : public gaussian_factor<Arg> {
    typedef argument_traits<Arg> arg_traits;

  public:
    // Public types
    //==========================================================================
    // Base type
    typedef gaussian_factor<Arg> base;

    // Factor member types
    typedef T                       real_type;
    typedef logarithmic<T>          result_type;
    typedef Arg                     argument_type;
    typedef domain<Arg>             domain_type;
    typedef real_assignment<Arg, T> assignment_type;

    // ParametricFactor member types
    typedef moment_gaussian_param<T> param_type;
    typedef dense_vector<T>           vector_type;
    typedef uint_vector index_type;
    typedef multivariate_normal_distribution<T> distribution_type;

    // LearnableDistributionFactor member types
    typedef moment_gaussian_mle<T> mle_type;

    // Constructors and conversion operators
    //==========================================================================

    //! Default constructor. Creates an empty factor.
    moment_gaussian() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit moment_gaussian(const domain_type& head,
                             const domain_type& tail = domain_type()) {
      reset(head, tail);
    }

    //! Constructs a factor equivalent to a constant.
    explicit moment_gaussian(logarithmic<T> value)
      : param_(value.lv) { }

    //! Constructs a factor with the given head and parameters and empty tail.
    moment_gaussian(const domain_type& head,
                    const param_type& param)
      : base(head),
        head_(head),
        param_(param) {
      check_param();
    }

    //! Constructs a factor with the given head and parameters and empty tail.
    moment_gaussian(const domain_type& head,
                    param_type&& param)
      : base(head),
        head_(head),
        param_(std::move(param)) {
      check_param();
    }

    //! Constructs a factor with the given arguments and parameters.
    moment_gaussian(const domain_type& head,
                    const domain_type& tail,
                    const param_type& param)
      : base(head, tail),
        head_(head),
        tail_(tail),
        args_(head + tail),
        param_(param) {
      check_param();
    }

    //! Constructs a factor with the given arguments and parameters.
    moment_gaussian(const domain_type& head,
                    const domain_type& tail,
                    param_type&& param)
      : base(head, tail),
        head_(head),
        tail_(tail),
        args_(head + tail),
        param_(std::move(param)) {
      check_param();
    }

    //! Constructs a marginal moment Gaussian factor.
    moment_gaussian(const domain_type& head,
                    const dense_vector<T>& mean,
                    const dense_matrix<T>& cov,
                    T lm = T(0))
      : base(head),
        head_(head),
        param_(mean, cov, lm) {
      check_param();
    }

    //! Constructs a conditional moment Gaussian factor.
    moment_gaussian(const domain_type& head,
                    const domain_type& tail,
                    const dense_vector<T>& mean,
                    const dense_matrix<T>& cov,
                    const dense_matrix<T>& coeff,
                    T lm = T(0))
      : base(head, tail),
        head_(head),
        tail_(tail),
        args_(head + tail),
        param_(mean, cov, coeff, lm) {
      check_param();
    }

    //! Conversion from a canonical_gaussian.
    explicit moment_gaussian(const canonical_gaussian<Arg, T>& cg) {
      *this = cg;
    }

    //! Assigns a constant to this factor.
    moment_gaussian& operator=(logarithmic<T> value) {
      reset();
      param_.lm = value.lv;
      return *this;
    }

    //! Assigns a canonical_gaussian to this factor.
    moment_gaussian& operator=(const canonical_gaussian<Arg, T>& cg) {
      reset(cg.arguments());
      param_ = cg.param();
      return *this;
    }

    //! Casts this moment_gaussian to a canonical_gaussian.
    canonical_gaussian<Arg, T> canonical() const {
      return canonical_gaussian<Arg, T>(*this);
    }

    //! Exchanges the content of two factors.
    friend void swap(moment_gaussian& f, moment_gaussian& g) {
      using std::swap;
      f.base_swap(g);
      swap(f.args_, g.args_);
      swap(f.head_, g.head_);
      swap(f.tail_, g.tail_);
      swap(f.param_, g.param_);
    }

    // Serialization and initialization
    //==========================================================================

    //! Serializes the factor to an archive.
    void save(oarchive& ar) const {
      ar << head_ << tail_ << param_;
    }

    //! Deserializes the factor from an archive.
    void load(iarchive& ar) {
      ar >> head_ >> tail_ >> param_;
      args_ = head_ + tail_;
      this->compute_start(head_, tail_);
      check_param();
    }

    //! Sets the arguments to the given head & tail and allocates the memory.
    void reset(const domain_type& head = domain_type(),
               const domain_type& tail = domain_type()) {
      if (head_ != head || tail_ != tail) {
        head_ = head;
        tail_ = tail;
        if (tail.empty()) { args_.clear(); } else { args_ = head + tail; }
        std::size_t m, n;
        std::tie(m, n) = this->compute_start(head, tail);
        param_.resize(m, n);
      }
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the arguments of this factor.
    const domain_type& arguments() const {
      return tail_.empty() ? head_ : args_;
    }

    //! Returns the head arguments of this factor.
    const domain_type& head() const {
      return head_;
    }

    //! Returns the tail arguments of this factor.
    const domain_type& tail() const {
      return tail_;
    }

    //! Returns the number of arguments of this factor.
    std::size_t arity() const {
      return head_.size() + tail_.size();
    }

    //! Returns the number of head arguments of this factor.
    std::size_t head_arity() const {
      return head_.size();
    }

    //! Returns the number of tail arguments of this factor.
    std::size_t tail_arity() const {
      return tail_.size();
    }

    //! Returns true if the factor represents a marginal distribution.
    bool is_marginal() const {
      return tail_.empty();
    }

    //! Returns true if the factor is empty.
    bool empty() const {
      return arguments().empty();
    }

    //! Returns the number of dimensions (head + tail).
    std::size_t size() const {
      return param_.size();
    }

    //! Returns the number of head dimensions of this Gaussian.
    std::size_t head_size() const {
      return param_.head_size();
    }

    //! Returns the number of tail dimensions of this Gaussian.
    std::size_t tail_size() const {
      return param_.tail_size();
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

    //! Returns the mean vector.
    const dense_vector<T>& mean() const {
      return param_.mean;
    }

    //! Returns the covariance matrix.
    const dense_matrix<T>& covariance() const {
      return param_.cov;
    }

    //! Returns the coefficient matrix.
    const dense_matrix<T>& coefficients() const {
      return param_.coef;
    }

    //! Returns the mean for a single argument. Supported for univariate Arg.
    LIBGM_ENABLE_IF(is_univariate<Arg>::value)
    T mean(Arg v) const {
      return param_.mean[this->start_.at(v)];
    }

    //! Returns variance for a single argument. Supported for univariate Arg.
    LIBGM_ENABLE_IF(is_univariate<Arg>::value)
    T variance(Arg v) const {
      std::size_t i = this->start_.at(v);
      return param_.cov(i, i);
    }

    /**
     * Returns the mean vector for a single argument.
     * Supported for multivariate Arg.
     */
    LIBGM_ENABLE_IF(is_multivariate<Arg>::value)
    Eigen::VectorBlock<const dense_vector<T> > mean(Arg v) const {
      return param_.mean.segment(this->start_.at(v),
                                 arg_traits::num_dimensions(v));
    }

    /**
     * Returns the covariance matrix for a single argument.
     * Supported for multivariate Arg.
     */
    LIBGM_ENABLE_IF(is_multivariate<Arg>::value)
    Eigen::Block<const dense_matrix<T> > covariance(Arg v) const {
      std::size_t i = this->start_.at(v);
      std::size_t n = arg_traits::num_dimensions(v);
      return param_.cov.block(i, i, n, n);
    }

    //! Returns the mean for a subset of the arguments
    dense_vector<T> mean(const domain_type& args) const {
      index_type index = args.index(this->start_);
      return subvec(param_.mean, iref(index));
    }

    //! Returns the covariance matrix for a subset of the arguments
    dense_matrix<T> covariance(const domain_type& args) const {
      index_type index = args.index(this->start_);
      return submat(param_.cov, iref(index), iref(index));
    }

    //! Returns true of the two factors have the same domains and parameters.
    bool operator==(const moment_gaussian& other) const {
      return
        head_ == other.head_ &&
        tail_ == other.tail_ &&
        param_ == other.param_;
    }

    //! Returns true if the two factors do not have the same domains or params.
    bool operator!=(const moment_gaussian& other) const {
      return !(*this == other);
    }

    // Indexing
    //==========================================================================

    /**
     * Converts the given vector to an assignment over head arguments.
     */
    void assignment(const dense_vector<T>& vec, assignment_type& a) const {
      a.insert_or_assign(head(), vec);
    }

    /**
     * Substitutes the arguments in-place according to the given map.
     */
    void subst_args(const std::unordered_map<Arg, Arg>& map) {
      base::subst_args(map);
      head_.substitute(map);
      tail_.substitute(map);
      args_.substitute(map);
    }

    /**
     * Reorders the arguments according to the given head and tail.
     */
    moment_gaussian reorder(const domain_type& head,
                            const domain_type& tail = domain_type()) const {
      if (!equivalent(head, head_)) {
        throw std::runtime_error("moment_gaussian::reorder: invalid head");
      }
      if (!equivalent(tail, tail_)) {
        throw std::runtime_error("moment_gaussian::reorder: invalid tail");
      }
      moment_gaussian result(head, tail);
      param_.reorder(iref(head.index(this->start_)),
                     iref(tail.index(this->start_)),
                     result.param_);
      return result;
    }

    /**
     * Checks if the size of the parameter struct matches this factor's
     * arguments.
     * \throw std::invalid_argument if the sizes do not match
     */
    void check_param() const {
      param_.check();
      if (param_.head_size() != head_.num_dimensions()) {
        throw std::runtime_error("moment_gaussian: Invalid head size");
      }
      if (param_.tail_size() != tail_.num_dimensions()) {
        throw std::runtime_error("moment_gaussian: Invalid tail size");
      }
    }

    /**
     * Checks if two factors have the same (sequence of) arguments.
     * \throw std::invalid_argument if the arguments do not match
     */
    friend void check_same_arguments(const moment_gaussian& f,
                                     const moment_gaussian& g) {
      if (f.head() != g.head() || f.tail() != g.tail()) {
        throw std::invalid_argument("moment_gaussian: incompatible arguments");
      }
    }

    // Factor evaluation
    //==========================================================================

    //! Evaluates the factor for an assignment.
    logarithmic<T> operator()(const assignment_type& a) const {
      return logarithmic<T>(log(a), log_tag());
    }

    //! Evaluates the factor for a vector.
    logarithmic<T> operator()(const vector_type& x) const {
      return logarithmic<T>(log(x), log_tag());
    }

    //! Returns the log-value of the factor for an assignment.
    T log(const assignment_type& a) const {
      return param_(a.values(arguments()));
    }

    //! Returns the log-value of the factor for a vector.
    T log(const vector_type& x) const {
      return param_(x);
    }

    // Factor operations (the parameter operation objects are computed below)
    //==========================================================================

    //! Multiplies this factor by a constant in-place.
    moment_gaussian& operator*=(logarithmic<T> x) {
      param_.lm += x.lv;
      return *this;
    }

    //! Divides this factor by a constant in-place.
    moment_gaussian& operator/=(logarithmic<T> x) {
      param_.lm -= x.lv;
      return *this;
    }

    //! Multiplies a moment_gaussian by a constant.
    friend moment_gaussian operator*(moment_gaussian f, logarithmic<T> x) {
      f.param_.lm += x.lv;
      return f;
    }

    //! Multiplies a moment_gaussian by a constant.
    friend moment_gaussian operator*(logarithmic<T> x, moment_gaussian f) {
      f.param_.lm += x.lv;
      return f;
    }

    //! Divides a moment_gaussian by a constant.
    friend moment_gaussian operator/(moment_gaussian f, logarithmic<T> x) {
      f.param_.lm -= x.lv;
      return f;
    }

    //! Multiplies two moment_gaussian factors.
    friend moment_gaussian
    operator*(const moment_gaussian& f, const moment_gaussian& g) {
      const moment_gaussian& p = f.is_marginal() ? f : g;
      const moment_gaussian& q = f.is_marginal() ? g : f;
      if (p.is_marginal() && disjoint(f.head(), g.head())) {
        domain<Arg> x = q.tail() & p.head();
        domain<Arg> z = q.tail() - p.head();
        moment_gaussian result(concat(f.head(), g.head()), z);
        multiply_head_tail(p.param_, q.param_,
                           iref(x.index(p.start())), iref(x.index(q.start())),
                           f.is_marginal(),
                           result.param_);
        return result;
      } else {
        throw std::invalid_argument(
          "moment_gaussian::operator*: unsupported argument domains."
       );
      }
    }

    /**
     * Computes an aggregate (marginal or maximum) of the factor
     * over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     */
    moment_gaussian aggregate(bool marginal, const domain_type& retain) const {
      moment_gaussian result(retain & head(), retain & tail());
      if (result.arity() != retain.size()) {
        throw std::invalid_argument(
          "moment_gaussian::marginal: some of the retained arguments "
          "are not present in the factor"
        );
      }
      if (result.tail_arity() != tail_arity()) {
        throw std::invalid_argument(
          "moment_gaussian::marginal cannot eliminate tail arguments"
        );
      }
      param_.collapse(marginal,
                      iref(result.head().index(this->start_)),
                      iref(result.tail().index(this->start_)),
                      result.param_);
      return result;
    }

    /**
     * Computes the marginal of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     */
    moment_gaussian marginal(const domain_type& retain) const {
      return aggregate(true, retain);
    }

    /**
     * Computes the maximum of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     */
    moment_gaussian maximum(const domain_type& retain) const {
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
      dense_vector<T> vec;
      T max = param_.maximum(vec);
      a.insert_or_assign(head_, vec);
      return logarithmic<T>(max, log_tag());
    }

    //! Normalizes the factor in-place.
    moment_gaussian& normalize() {
      param_.lm = 0;
      return *this;
    }

    //! Returns true if the factor is normalizable.
    bool normalizable() const {
      return std::isfinite(param_.lm);
    }

    //! If this factor represents p(x, y), returns p(x | y).
    moment_gaussian conditional(const domain_type& tail) const {
      assert(is_marginal());
      domain<Arg> head = head_ - tail;
      if (head_.size() != head.size() + tail.size()) {
        throw std::invalid_argument(
          "moment_gaussian::conditional: some of the tail arguments "
          "are note present in the factor"
        );
      }
      typename param_type::conditional_workspace ws;
      moment_gaussian reordered = reorder(concat(head, tail));
      moment_gaussian result(head, tail);
      reordered.param_.conditional(result.head_size(), ws, result.param_);
      return result;
    }

    //! Restricts the factor to the given assignment.
    moment_gaussian restrict(const assignment_type& a) const {
      if (subset(tail_, a)) {
        // case 1: partially restricted head, fully restricted tail
        domain<Arg> y, x; // restricted, retained
        head_.partition(a, y, x);
        moment_gaussian result(x);
        typename param_type::restrict_workspace ws;
        param_.restrict_both(iref(x.index(this->start_)),
                             iref(y.index(this->start_)),
                             a.values(y),
                             a.values(tail_),
                             ws,
                             result.param_);
        return result;
      } else if (disjoint(head_, a)) {
        // case 2: unrestricted head, partially restricted tail
        domain<Arg> y, x; // restricted, retained
        tail_.partition(a, y, x);
        moment_gaussian result(head_, x);
        param_.restrict_tail(iref(x.index(this->start_)),
                             iref(y.index(this->start_)),
                             a.values(y),
                             result.param_);
        return result;
      } else {
        throw std::invalid_argument(
          "moment_gaussian::restrict: unsuported operation"
        );
      }
    }

    // Sampling
    //==========================================================================

    //! Returns the distribution with the parameters of this factor.
    multivariate_normal_distribution<T> distribution() const {
      return multivariate_normal_distribution<T>(param_);
    }

    //! Draws a random sample from a marginal distribution.
    template <typename Generator>
    dense_vector<T> sample(Generator& rng) const {
      return param_.sample(rng);
    }

    //! Draws a random sample from a conditional distribution.
    template <typename Generator>
    dense_vector<T> sample(Generator& rng, const dense_vector<T>& tail) const {
      assert(tail.size() == tail_size());
      return param_.sample(rng, tail);
    }

    /**
     * Draws a random sample from a marginal distribution,
     * storing the result in an assignment.
     */
    template <typename Generator>
    void sample(Generator& rng, assignment_type& a) const {
      a.insert_or_assign(head_, sample(rng, a.values(tail_)));
    }

    /**
     * Draws a random sample from a conditional distribution,
     * extracting the tail from and storing the result to an assignment.
     * \param ntail the tail arguments (must be equivalent to factor tail).
     */
    template <typename Generator>
    void sample(Generator& rng, const domain_type& tail,
                assignment_type& a) const {
      assert(equivalent(tail, tail_));
      a.insert_or_assign(head_, sample(rng, a.values(tail_)));
    }

    // Entropy and divergences
    //==========================================================================

    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return param_.entropy();
    }

    //! Computes the entropy for a subset of arguments.
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
    friend T kl_divergence(const moment_gaussian& p,
                           const moment_gaussian& q) {
      check_same_arguments(p, q);
      return kl_divergence(p.param_, q.param_);
    }

    friend T max_diff(const moment_gaussian& f,
                      const moment_gaussian& g) {
      check_same_arguments(f, g);
      return max_diff(f.param_, g.param_);
    }

    // Private data members
    //==========================================================================
  private:
    //! The head arguments of the factor.
    domain_type head_;

    //! The tail arguments of the factor.
    domain_type tail_;

    //! The concatenation of head_ and tail_ when tail_ is not empty.
    domain_type args_;

    //! The parameters of the factor.
    param_type param_;

  }; // class moment_gaussian


  /**
   * Prints the moment_gaussian to a stream.
   * \relates moment_gaussian
   */
  template <typename Arg, typename T>
  std::ostream& operator<<(std::ostream& out,
                           const moment_gaussian<Arg, T>& f) {
    out << "moment_gaussian(" << f.head() << ", " << f.tail() << ")\n"
        << f.param() << std::endl;
    return out;
  }

} // namespace libgm

#endif
