#ifndef LIBGM_SOFTMAX_HPP
#define LIBGM_SOFTMAX_HPP

#include <libgm/argument/hybrid_assignment.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/datastructure/hybrid_vector.hpp>
#include <libgm/factor/base/factor.hpp>
#include <libgm/factor/probability_array.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/likelihood/softmax_ll.hpp>
#include <libgm/math/likelihood/softmax_mle.hpp>
#include <libgm/math/param/softmax_param.hpp>
#include <libgm/math/random/softmax_distribution.hpp>

#include <iostream>
#include <sstream>

namespace libgm {

  /**
   * A factor that represents a conditional distribution over a discrete
   * variable given a collection of continuous variables. The conditional
   * distribution is given by a normalized exponential,
   * p(y = j | x) \propto exp(b_j + x^T w_j).
   *
   * \tparam T a real type for representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T, typename Var>
  class softmax : public factor {
  public:
    // Public types
    //==========================================================================
    // Factor member types
    typedef T                         real_type;
    typedef T                         result_type;
    typedef Var                       variable_type;
    typedef hybrid_domain<Var>        domain_type;
    typedef hybrid_assignment<T, Var> assignment_type;

    // ParametricFactor member types
    typedef softmax_param<T> param_type;
    typedef hybrid_vector<T> index_type;
    typedef softmax_distribution<T> distribution_type;

    // LearnableFactor member types
    typedef softmax_ll<T>  ll_type;
    typedef softmax_mle<T> mle_type;

    // Types to represent the parameters
    typedef real_matrix<T> mat_type;
    typedef real_vector<T> vec_type;

    // Constructors and conversion operators
    //==========================================================================

    /**
     * Default constructor. Creates an empty factor.
     */
    softmax() { }

    /**
     * Constructs a factor with the given arguments. The discrete component
     * of the domain must have exactly one variable.
     * Allocates the parameters but doesnot initialize their values.
     */
    explicit softmax(const domain_type& args) {
      reset(args);
    }

    /**
     * Constructs a factor with the given label variable and feature arguments.
     * Allocates the parameters but does not initialize their values.
     */
    softmax(Var head, const basic_domain<Var>& tail) {
      reset(head, tail);
    }

    /**
     * Constructs a factor with the given domain which must contain exactly
     * one discreteargument. Sets the parameters to the given parameter vector.
     */
    softmax(const domain_type& args, const param_type& param)
      : args_(args), param_(param) {
      check_param();
    }

    /**
     * Constructs a factor with the given domain which must contain exactly
     * one discrete argument. Sets the parameters to the given parameter vector.
     */
    softmax(const domain_type& args, param_type&& param)
      : args_(args), param_(std::move(param)) {
      check_param();
    }

    /**
     * Exchanges the arguments and the parameters of two factors.
     */
    friend void swap(const softmax& f, const softmax& g) {
      if (&f != &g) {
        swap(f.args_, g.args_);
        swap(f.param_, g.param_);
      }
    }

    /**
     * Resets thsi factor to the given domain. The parameters may become
     * invalidated.
     */
    void reset(const domain_type& args) {
      if (args_ != args) {
        assert(args.discrete().size() == 1);
        args_ = args;
        param_.resize(num_values(args.discrete()[0]),
                      num_dimensions(args.continuous()));
      }
    }

    /**
     * Resets the content of this factor to the given head and tail
     * arguments. The parameter values may become invalidated.
     */
    void reset(Var head, const basic_domain<Var>& tail) {
      args_.discrete().assign(1, head);
      args_.continuous() = tail;
      param_.resize(num_values(head), num_dimensions(tail));
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the arguments set of this factor
    const domain_type& arguments() const {
      return args_;
    }

    //! Returns the label variable.
    Var head() const {
      assert(!args_.empty());
      return args_.discrete()[0];
    }

    //! Returns the feature arguments of this factor.
    const basic_domain<Var>& tail() const {
      return args_.continuous();
    }

    //! Returns true if the factor is empty.
    bool empty() const {
      return args_.empty();
    }

    //! Returns the number of arguments of this factor or 0 if it is empty.
    std::size_t arity() const {
      return args_.size();
    }

    //! Returns the number of assignments to the head variable.
    std::size_t labels() const {
      return param_.labels();
    }

    //! Returns the dimensionality of the underlying feature vector.
    std::size_t features() const {
      return param_.features();
    }

    //! Returns the parameters of this factor.
    const param_type& param() const {
      return param_;
    }

    //! Provides mutable access to the parameters of this factor.
    param_type& param() {
      return param_;
    }

    //! Returns the weight matrix.
    const mat_type& weight() const {
      return param_.weight();
    }

    //! Returns the bias vector.
    const vec_type& bias() const {
      return param_.bias();
    }

    /**
     * Returns the value of the factor for the given index.
     * The first integral value is assumed to be the label.
     */
    T operator()(const hybrid_vector<T>& index) const {
      return param_(index.real())[index.uint()[0]];
    }

    /**
     * Returns the value of the factor (conditional probability) for the
     * given assignment.
     * \param strict if true, requires all the arguments to be present;
     *        otherwise, only the label variable must be present and the
     *        missing features are assumed to be 0.
     */
    T operator()(const assignment_type& a, bool strict = true) const {
      std::size_t label = a.uint().at(head());
      if (strict) {
        vec_type features;
        extract_features(a.real(), features);
        return param_(features)[label];
      } else {
        assert(false); // not implemented yet
      }
    }

    /**
     * Returns the log-value of the factor for the given index.
     * The first integral value is assumed ot be the label.
     */
    T log(const hybrid_vector<T>& index) const {
      return std::log(operator()(index));
    }

    /**
     * Returns the log of the value of the factor (conditional probability)
     * for the given assignment.
     * \param strict if true, requires all the arguments to be present;
     *        otherwise, only the label variable must be present and the
     *        missing features are assumed to be 0.
     */
    T log(const assignment_type& a, bool strict = true) const {
      return std::log(operator()(a, strict));
    }

    /**
     * Returns true if the two factors have the same domains and
     * parameters.
     */
    friend bool operator==(const softmax& f, const softmax& g) {
      return f.args_ == g.args_ && f.param_ == g.param_;
    }

    /**
     * Returns true if the two factors do not have the same domains
     * or parameters.
     */
    friend bool operator!=(const softmax& f, const softmax& g) {
      return !(f == g);
    }

    // Indexing
    //==========================================================================

    /**
     * Extracts a dense feature vector from an assignment. All the tail
     * variables must be present in the assignment.
     */
    void extract_features(const real_assignment<T, Var>& a,
                          vec_type& result) const {
      result.resize(features());
      std::size_t row = 0;
      for (Var v : tail()) {
        auto it = a.find(v);
        if (it != a.end()) {
          result.segment(row, num_dimensions(v)) = it->second;
          row += num_dimensions(v);
        } else {
          std::ostringstream out;
          out << "The assignment does not contain the tail variable " << v;
          throw std::invalid_argument(out.str());
        }
      }
    }

#if 0
    /**
     * Extracts a sparse vector of features from an assignment. Tail variables
     * that are missing in the assignment are assumed to have a value of 0.
     */
    void extract_features(const real_assignment<T>& a,
                          sparse_index<T>& result) const {
      result.clear();
      result.reserve(num_dimensions(tail()));
      std::size_t id = 0;
      for (Var v : tail()) {
        auto it = a.find(v);
        if (it != a.end()) {
          for (std::size_t i = 0; i < v->size(); ++i) {
            result.emplace_back(id + i, it->second[i]);
          }
        }
        id += v->size();
      }
    }
#endif

    /**
     * Checks if the dimensions of the parameters match this factor's arguments.
     * \throw std::runtime_error if some of the dimensions do not match.
     */
    void check_param() const {
      if (empty()) {
        if (!param_.empty()) {
          throw std::runtime_error(
            "The factor is empty but the parameters are not!"
          );
        }
      } else {
        if (param_.labels() != num_values(head())) {
          throw std::runtime_error("Invalid number of labels");
        }
        if (param_.features() != num_dimensions(tail())) {
          throw std::runtime_error("Invalid number of features");
        }
      }
    }

    // Factor operations
    //==========================================================================

    /**
     * Returns true if the factor represents a valid distribution.
     * This is true if none of the parameters are infinite / nan.
     */
    bool is_normalizable() const {
      return param_.is_finite();
    }

    /**
     * Conditions the factor on the given features in the factor's internal
     * ordering of tail variables.
     */
    probability_array<T, 1, Var>
    condition(const vec_type& index) const {
      return probability_array<T, 1, Var>({head()}, param_(index));
    }

#if 0
    /**
     * Conditions the factor on the assignment to its tail variables.
     * \param strict if true, requires that all the tail arguments are present
     *        in the assignment.
     */
    probability_array<T, 1, Var>
    condition(const real_assignment<T, Var>& a, bool strict = true) const {
      if (strict) {
        vec_type features;
        extract_features(a, features);
        return probability_array<T, 1, Var>({head()}, param_(features));
      } else {
        sparse_index<T> features;
        extract_features(a, features);
        return probability_array<T, 1, Var>({head()}, param_(features));
      }
    }

    /**
     * Returns the accuracy of predictions for this model.
     */
    T accuracy(const hybrid_dataset<T>& ds) const {
      T correct(0);
      T weight(0);
      foreach(const hybrid_record<T>& r, ds.records({head_}, tail_)) {
        arma::uword prediction;
        param_(r.values.vector).max(prediction);
        correct += r.weight * (prediction == r.values.finite[0]);
        weight += r.weight;
      }
      return correct / weight;
    }
#endif

    // Sampling
    //==========================================================================

    //! Returns the distribution with the parameters of this factor.
    softmax_distribution<T> distribution() const {
      return softmax_distribution<T>(param_);
    }

    //! Draws a random sample from a conditional distribution.
    template <typename Generator>
    std::size_t sample(Generator& rng, const vec_type& tail) const {
      return param_.sample(rng, tail);
    }

    /**
     * Draws a random sample from a conditional distribution,
     * extracting the tail from and storing the result to an assignment.
     * \param ntail the tail variables (must be a suffix of the domain).
     */
    template <typename Generator>
    void sample(Generator& rng, assignment_type& a) const {
      a.uint()[head()] = param_.sample(rng, extract(a, tail()));
    }

    // Private members
    //==========================================================================
  private:
    //! The arguments of this factor.
    domain_type args_;

    //! The underlying softmax parameters.
    softmax_param<T> param_;

  }; // class softmax

  /**
   * Prints a human-readable representation of the CPD to a stream.
   * \relates softmax
   */
  template <typename T, typename Var>
  std::ostream& operator<<(std::ostream& out, const softmax<T, Var>& f) {
    if (f.empty()) {
      out << "softmax()" << std::endl;
    } else {
      out << "softmax(" << f.head() << "|" << f.tail() << ")" << std::endl
          << f.param();
    }
    return out;
  }


  // Utility classes
  //==========================================================================

  /**
   * A specialization of factor_mle to softmax. By the very nature of softmax,
   * this estimator only supports conditional distributions.
   *
   * \tparam T the real type representing the parameters of softmax
   */
  template <typename T, typename Var>
  class factor_mle<softmax<T, Var> > {
  public:
    //! The domain type of the factor.
    typedef hybrid_domain<Var> domain_type;

    //! The maximum likelihoo estimator of factor parameters.
    typedef softmax_mle<T> mle_type;

    //! The regularization paramters for the MLE.
    typedef typename mle_type::regul_type regul_type;

    typedef softmax<T, Var> factor_type;

    /**
     * Constructs a factor estimator with the specified regularization
     * parameters.
     */
    explicit factor_mle(const regul_type& regul = regul_type())
      : mle_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a conditional distribution
     * p(f | v), where f and v are components of a hybrid domain.
     */
    template <typename Dataset>
    factor_type operator()(const Dataset& ds, const domain_type& args) const {
      return factor_type(args, mle_(ds(args), factor_type::param_shape(args)));
    }

    /**
     * Computes the maximun likelihood estimate of a conditional distribution
     * p(head | tail).
     */
    template <typename Dataset>
    factor_type operator()(const Dataset& ds,
                           Var head, const basic_domain<Var>& tail) const {
      domain_type args({head}, tail);
      return factor_type(args, mle_(ds(args), factor_type::param_shape(args)));
    }

  private:
    //! The maximum likelihood estimator of the factor parameters.
    mle_type mle_;

  }; // class factor_mle<softmax<T>, Dataset>

} // namespace libgm

#endif
