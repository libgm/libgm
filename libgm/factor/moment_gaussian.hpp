#ifndef LIBGM_MOMENT_GAUSSIAN_HPP
#define LIBGM_MOMENT_GAUSSIAN_HPP

#include <libgm/argument/vector_assignment.hpp>
#include <libgm/factor/base/gaussian_factor.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/math/likelihood/moment_gaussian_mle.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>
#include <libgm/math/random/gaussian_distribution.hpp>

namespace libgm {

  // forward declaration
  template <typename T, typename Var> class canonical_gaussian;

  /**
   * A factor of a Gaussian distribution in the moment parameterization.
   *
   * \tparam T the real type for representing the parameters.
   * \ingroup factor_types
   */
  template <typename T, typename Var>
  class moment_gaussian : public gaussian_factor<Var> {
  public:
    // Public types
    //==========================================================================
    // Base type
    typedef gaussian_factor<Var> base;

    // Underlying storage
    typedef dynamic_matrix<T> mat_type;
    typedef dynamic_vector<T> vec_type;

    // Factor member types
    typedef T                         real_type;
    typedef logarithmic<T>            result_type;
    typedef Var                       variable_type;
    typedef basic_domain<Var>         domain_type;
    typedef vector_assignment<T, Var> assignment_type;

    // ParametricFactor member types
    typedef moment_gaussian_param<T> param_type;
    typedef dynamic_vector<T>        index_type;
    typedef gaussian_distribution<T> distribution_type;

    // LearnableDistributionFactor member types
    typedef moment_gaussian_mle<T> mle_type;
    
    // Constructors and conversion operators
    //==========================================================================

    //! Default constructor. Creats an empty factor.
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
                    const vec_type& mean,
                    const mat_type& cov,
                    T lm = T(0))
      : base(head),
        head_(head),
        param_(mean, cov, lm) {
      check_param();
    }

    //! Constructs a conditional moment Gaussian factor.
    moment_gaussian(const domain_type& head,
                    const domain_type& tail,
                    const vec_type& mean,
                    const mat_type& cov,
                    const mat_type& coeff,
                    T lm = T(0))
      : base(head, tail),
        head_(head),
        tail_(tail),
        args_(head + tail),
        param_(mean, cov, coeff, lm) {
      check_param();
    }

    //! Conversion from a canonical_gaussian.
    explicit moment_gaussian(const canonical_gaussian<T, Var>& cg) {
      *this = cg;
    }
    
    //! Assigns a constant to this factor.
    moment_gaussian& operator=(logarithmic<T> value) {
      reset();
      param_.lm = value.lv;
      return *this;
    }

    //! Assigns a moment_gaussian to this factor.
    moment_gaussian& operator=(const canonical_gaussian<T, Var>& cg) {
      reset(cg.arguments());
      param_ = cg.param();
      return *this;
    }

    //! Casts this moment_gaussian to a canonical_gaussian.
    canonical_gaussian<T, Var> canonical() const {
      return canonical_gaussian<T, Var>(*this);
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
        if (tail.empty()) args_.clear(); else args_ = head + tail;
        size_t m, n;
        std::tie(m, n) = this->compute_start(head, tail);
        param_.resize(m, n);
      }
    }

    //! Sets the arguments for the factor and deallocates the memory.
    void reset_prototype(const domain_type& head,
                         const domain_type& tail = domain_type()) {
      if (head_ != head || tail_ != tail) {
        head_ = head;
        tail_ = tail;
        if (tail.empty()) args_.clear(); else args_ = head + tail;
        this->compute_start(head, tail);
        param_.resize(0, 0);
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
    size_t arity() const {
      return head_.size() + tail_.size();
    }

    //! Returns the number of head arguments of this factor.
    size_t head_arity() const {
      return head_.size();
    }

    //! Returns the number of tail arguments of this factor.
    size_t tail_arity() const {
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
    size_t size() const {
      return param_.size();
    }

    //! Returns the number of head dimensions of this Gaussian.
    size_t head_size() const {
      return param_.head_size();
    }

    //! Returns the number of tail dimensions of this Gaussian.
    size_t tail_size() const {
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
    const vec_type& mean() const {
      return param_.mean;
    }

    //! Returns the covariance matrix.
    const mat_type& covariance() const {
      return param_.cov;
    }

    //! Returns the coefficient matrix.
    const mat_type& coefficients() const {
      return param_.coef;
    }

    //! Returns the mean for a single variable.
    Eigen::VectorBlock<const vec_type> mean(Var v) const {
      return param_.mean.segment(this->start(v), v.size());
    }

    //! Returns the covariance matrix for a single variable.
    Eigen::Block<const mat_type> covariance(Var v) const {
      size_t i = this->start(v);
      return param_.cov.block(i, i, v.size(), v.size());
    }

    //! Returns the mean for a subset of the arguments
    vec_type mean(const domain_type& args) const {
      matrix_index map = this->index_map(args);
      return subvec(param_.mean, map).plain();
    }

    //! Returns the covariance matrix for a subset of the arguments
    mat_type covariance(const domain_type& args) const {
      matrix_index map = this->index_map(args);
      return submat(param_.cov, map, map).plain();
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
     * Converts the given vector to an assignment over head variables.
     */
    void assignment(const vec_type& vec, assignment_type& a) const {
      assert(vec.size() == head_size());
      size_t i = 0;
      for (Var v : head()) {
        a[v] = vec.segment(i, v.size());
        i += v.size();
      }
    }

    /**
     * Substitutes the arguments in-place according to the given map.
     */
    void subst_args(const std::unordered_map<Var, Var>& map) {
      base::subst_args(map);
      head_.subst(map);
      tail_.subst(map);
      args_.subst(map);
    }

    /**
     * Reorders the arguments according to the given head and tail.
     */
    moment_gaussian
    reorder(const domain_type& head,
            const domain_type& tail = domain_type()) const {
      if (!equivalent(head, head_)) {
        throw std::runtime_error("moment_gaussian::reorder: invalid head");
      }
      if (!equivalent(tail, tail_)) {
        throw std::runtime_error("moment_gaussian::reorder: invalid tail");
      }
      matrix_index head_map = this->index_map(head);
      matrix_index tail_map = this->index_map(tail);
      return moment_gaussian(head, tail, param_.reorder(head_map, tail_map));
    }
    
    /**
     * Checks if the size of the parameter struct matches this factor's
     * arguments.
     * \throw std::invalid_argument if the sizes do not match
     */
    void check_param() const {
      param_.check();
      if (param_.head_size() != vector_size(head_)) {
        throw std::runtime_error("moment_gaussian: Invalid head size");
      }
      if (param_.tail_size() != vector_size(tail_)) {
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
    logarithmic<T> operator()(const vec_type& x) const {
      return logarithmic<T>(log(x), log_tag());
    }

    //! Returns the log-value of the factor for an assignment.
    T log(const assignment_type& a) const {
      return param_(extract(a, arguments()));
    }

    //! Returns the log-value of the factor for a vector.
    T log(const vec_type& x) const {
      return param_(x);
    }

    // Factor operations (the parameter operation objects are computed below)
    //==========================================================================

    //! Multiplies this factor by a constant in-place.
    moment_gaussian& operator*=(logarithmic<T> x) {
      multiplies_assign_op(*this)(param_, x.lv);
      return *this;
    }

    //! Divides this factor by a constant in-place.
    moment_gaussian& operator/=(logarithmic<T> x) {
      divides_assign_op(*this)(param_, x.lv);
      return *this;
    }

    //! Multiplies two moment_gaussian factors.
    friend moment_gaussian
    operator*(const moment_gaussian& f, const moment_gaussian& g) {
      moment_gaussian result;
      multiplies_op(f, g, result)(f.param_, g.param_, result.param_);
      return result;
    }

    //! Multiplies a moment_gaussian by a constant.
    friend moment_gaussian
    operator*(moment_gaussian f, logarithmic<T> x) {
      multiplies_assign_op(f)(f.param_, x.lv);
      return f;
    }

    //! Multiplies a moment_gaussian by a constant.
    friend moment_gaussian
    operator*(logarithmic<T> x, moment_gaussian f) {
      multiplies_assign_op(f)(f.param_, x.lv);
      return f;
    }

    //! Divides a moment_gaussian by a constant.
    friend moment_gaussian
    operator/(moment_gaussian f, logarithmic<T> x) {
      divides_assign_op(f)(f.param_, x.lv);
      return f;
    }

    /**
     * Computes the marginal of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     */
    moment_gaussian marginal(const domain_type& retain) const {
      moment_gaussian result;
      marginal(retain, result);
      return result;
    }

    /**
     * Computes the maximum of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     */
    moment_gaussian maximum(const domain_type& retain) const {
      moment_gaussian result;
      maximum(retain, result);
      return result;
    }

    /**
     * If this factor represents p(x, y), returns p(x | y).
     */
    moment_gaussian conditional(const domain_type& tail) const {
      moment_gaussian result;
      conditional_op(*this, tail, result)(param_, result.param_);
      return result;
    }

    /**
     * Computes the marginal of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a subset of arguments
     */
    void marginal(const domain_type& retain, moment_gaussian& result) const {
      marginal_op(*this, retain, result)(param_, result.param_);
    }

    /**
     * Computes the maximum of the factor over a sequence of arguments.
     * \throws invalid_argument if retained is not a ubset of arguments
     */
    void maximum(const domain_type& retain, moment_gaussian& result) const {
      maximum_op(*this, retain, result)(param_, result.param_);
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
      assignment(vec, a);
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

    //! Restricts the factor to the given assignment.
    moment_gaussian restrict(const assignment_type& a) const {
      moment_gaussian result;
      restrict(a, result);
      return result;
    }

    //! Restricts the factor to the given assignment.
    void restrict(const assignment_type& a, moment_gaussian& result) const {
      restrict_op(*this, a, result)(param_, result.param_);
    }

    // Sampling
    //==========================================================================
    
    //! Returns the distribution with the parameters of this factor.
    gaussian_distribution<T> distribution() const {
      return gaussian_distribution<T>(param_);
    }

    //! Draws a random sample from a marginal distribution.
    template <typename Generator>
    vec_type sample(Generator& rng) const {
      return param_.sample(rng);
    }

    //! Draws a random sample from a conditional distribution.
    template <typename Generator>
    vec_type sample(Generator& rng, const vec_type& tail) const {
      assert(tail.size() == tail_size());
      return param_.sample(rng, tail);
    }

    /**
     * Draws a random sample from a marginal distribution,
     * storing the result in an assignment.
     */
    template <typename Generator>
    void sample(Generator& rng, assignment_type& a) const {
      this->assignment(sample(rng, extract(a, tail_)), a);
    }

    /**
     * Draws a random sample from a conditional distribution,
     * extracting the tail from and storing the result to an assignment.
     * \param ntail the tail variables (must be equivalent to factor tail).
     */
    template <typename Generator>
    void sample(Generator& rng, const domain_type& tail,
                assignment_type& a) const {
      assert(equivalent(tail, tail_));
      this->assignment(sample(rng, extract(a, tail_)), a);
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
      return entropy(a) + entropy(b) - entropy(a | b);
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
   * A moment_gaussian factor using double precision.
   * \relates moment_gaussian
   */
  typedef moment_gaussian<double, variable> mgaussian;
  
  // Input / outputx
  //============================================================================

  /**
   * Prints the moment_gaussian to a stream.
   * \relates moment_gaussian
   */
  template <typename T, typename Var>
  std::ostream& operator<<(std::ostream& out,
                           const moment_gaussian<T, Var>& f) {
    out << "moment_gaussian(" << f.head() << ", " << f.tail() << ")\n"
        << f.param() << std::endl;
    return out;
  }

  // Factor operation objects
  //============================================================================

  /**
   * Returns an object that can add a constant to the log-multiplier of
   * a moment Gaussian in-place.
   */
  template <typename T, typename Var>
  moment_gaussian_join_inplace<T, libgm::plus_assign<> >
  multiplies_assign_op(moment_gaussian<T, Var>& h) {
    return { };
  }

  /**
   * Returns an object that can subtract a constant from the log-multiplier
   * of moment Gaussian in-place.
   */
  template <typename T, typename Var>
  moment_gaussian_join_inplace<T, libgm::minus_assign<> >
  divides_assign_op(moment_gaussian<T, Var>& h) {
    return { };
  }

  /**
   * Returns an object that can compute the parameters corresponding to the
   * product of two moment Gaussians. Initializes the arguments of the result.
   */
  template <typename T, typename Var>
  moment_gaussian_multiplies<T>
  multiplies_op(const moment_gaussian<T, Var>& f,
                const moment_gaussian<T, Var>& g,
                moment_gaussian<T, Var>& h) {
    const moment_gaussian<T, Var>& p = f.is_marginal() ? f : g;
    const moment_gaussian<T, Var>& q = f.is_marginal() ? g : f;
    if (p.is_marginal() && disjoint(f.head(), g.head())) {
      basic_domain<Var> x1 = q.tail() & p.head();
      basic_domain<Var> y  = q.tail() - p.head();
      h.reset_prototype(f.head() + g.head(), y);
      moment_gaussian_multiplies<T> op;
      op.p1 = p.index_map(x1);
      op.q1 = q.index_map(x1);
      op.qy = q.index_map(y);
      op.hp = h.index_map(p.head());
      op.hq = h.index_map(q.head());
      return op;
    } else {
      throw std::invalid_argument(
        "moment_gaussian::operator*: unsupported argument domains."
      );
    }
  }

  /**
   * Returns an object that can compute the parameters corresponding to
   * the marginal of a moment Gaussian over a subset of arguments.
   * Initializes the arguments of the result.
   */
  template <typename T, typename Var>
  moment_gaussian_collapse<T>
  marginal_op(const moment_gaussian<T, Var>& f,
              const basic_domain<Var>& retain,
              moment_gaussian<T, Var>& h) {
    basic_domain<Var> head = retain & f.head();
    basic_domain<Var> tail = retain & f.tail();
    if (head.size() + tail.size() != retain.size()) {
      throw std::invalid_argument(
        "moment_gaussian::marginal: some of the retained variables "
        "are not present in the factor"
      );
    }
    if (tail.size() != f.tail_size()) {
      throw std::invalid_argument(
        "moment_gaussian::marginal cannot eliminate tail variables"
      );
    }
    h.reset_prototype(head, tail);
    return { f.index_map(head), f.index_map(tail) };
  }
  
  /**
   * Returns an object that can compute the parameters corresponding to
   * the maximum of a moment Gaussian over a subset of arguments.
   * Initializes the arguments of the result.
   */
  template <typename T, typename Var>
  moment_gaussian_collapse<T>
  maximum_op(const moment_gaussian<T, Var>& f,
             const basic_domain<Var>& retain,
             moment_gaussian<T, Var>& h) {
    basic_domain<Var> head = retain & f.head();
    basic_domain<Var> tail = retain & f.tail();
    if (head.size() + tail.size() != retain.size()) {
      throw std::invalid_argument(
        "moment_gaussian::maximum: some of the retained variables "
        "are not present in the factor"
      );
    }
    if (tail.size() != f.tail_size()) {
      throw std::invalid_argument(
        "moment_gaussian::maximum cannot eliminate tail variables"
      );
    }
    h.reset_prototype(head, tail);
    return { f.index_map(head), f.index_map(tail), true /* preserve max */ };
  }

  /**
   * Returns an object that can compute the parameters corresponding to
   * conditioning a marginal moment Gaussian distribution.
   * Initializes the arguments of the result.
   */
  template <typename T, typename Var>
  moment_gaussian_conditional<T>
  conditional_op(const moment_gaussian<T, Var>& f,
                 const basic_domain<Var>& tail,
                 moment_gaussian<T, Var>& h) {
    assert(f.is_marginal());
    basic_domain<Var> head = f.head() - tail;
    if (f.size() != head.size() + tail.size()) {
      throw std::invalid_argument(
        "moment_gaussian::conditional: some of the tail variables "
        "are note present in the factor"
      );
    }
    return { f.index_map(head), f.index_map(tail) };
  }

  /**
   * Returns an object that can compute the parameters corresponding to
   * restricting a moment Gaussian to the given assignment.
   * Initializes the arguments of the result.
   */
  template <typename T, typename Var>
  moment_gaussian_restrict<T>
  restrict_op(const moment_gaussian<T, Var>& f,
              const vector_assignment<T>& a,
              moment_gaussian<T, Var>& h) {
    if (subset(f.tail(), a)) {
      // case 1: partially restricted head, fully restricted tail
      basic_domain<Var> y, x; // restricted, retained
      f.head().partition(a, y, x);
      h.reset_prototype(x);
      moment_gaussian_restrict<T> op(moment_gaussian_restrict<T>::MARGINAL);
      op.x = f.index_map(x);
      op.y = f.index_map(y);
      op.vec_y = extract(a, y);
      op.vec_z = extract(a, f.tail());
      return op;
    } else if (disjoint(f.head(), a)) {
      // case 2: unrestricted head, partially restricted tail
      basic_domain<Var> y, x; // restricted, retained
      f.tail().partition(a, y, x);
      h.reset_prototype(f.head(), x);
      moment_gaussian_restrict<T> op(moment_gaussian_restrict<T>::CONDITIONAL);
      op.x = f.index_map(x);
      op.y = f.index_map(y);
      op.vec_y = extract(a, y);
      return op;
    } else {
      throw std::invalid_argument(
        "moment_gaussian::restrict: unsuported operation"
      );
    }
  }

  // Traits
  //============================================================================

  //! \addtogroup factor_traits
  //! @{

  template <typename T, typename Var>
  struct has_multiplies<moment_gaussian<T, Var>> : public std::true_type { };

  template <typename T, typename Var>
  struct has_multiplies_assign<moment_gaussian<T, Var>> : public std::true_type { };

  template <typename T, typename Var>
  struct has_divides_assign<moment_gaussian<T, Var>> : public std::true_type { };

  template <typename T, typename Var>
  struct has_marginal<moment_gaussian<T, Var>> : public std::true_type { };

  template <typename T, typename Var>
  struct has_maximum<moment_gaussian<T, Var>> : public std::true_type { };

  template <typename T, typename Var>
  struct has_arg_max<moment_gaussian<T, Var>> : public std::true_type { };
  
  //! @}

} // namespace libgm

//#include <libgm/factor/gaussian_common.hpp>

#endif
