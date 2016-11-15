#ifndef LIBGM_NAIVE_BAYES_HPP
#define LIBGM_NAIVE_BAYES_HPP

#include <libgm/argument/traits.hpp>
#include <libgm/math/likelihood/range_ll.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/range/joined.hpp>

#include <unordered_map>

namespace libgm {

  /**
   * A class that represents the Maive Bayes model.
   *
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam LabelF
   *         A type representing the prior distribution.
   * \tparam FeatureF
   *         A type representing the conditional distribution p(feature | label)
   */
  template <typename Arg, typename LabelF, typename FeatureF = LabelF>
  class naive_bayes {
  public:
    static_assert(are_pairwise_compatible<LabelF, FeatureF>::value,
                  "The prior and feature factors are not pairwise compatible.");

    using feature_map =
      std::unordered_map<Arg, F, typename argument_traits<Arg>::hasher>;

    // Public type declarations
    //--------------------------------------------------------------------------
  public:
    // Argument types
    using argument_type     = Arg;
    using argument_hasher   = typename argument_traits<Arg>::hasher;
    using feature_iterator  = map_key_iterator<feature_map>;
    using argument_iterator = join_iterator<const Arg*, feature_iterator>;

    // Factor types
    using real_type   = typename LabelF::real_type;
    using result_type = typename LabelF::result_type;

    typedef typename LabelF::assignment_type assignment_type; // ???

    // Constructors and initialization
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty naive Bayes model.
    naive_bayes()
      : label_(vertex_traits<Arg>::null()) { }

    //! Creates a naive Bayes model with the given label and uniform prior.
    explicit naive_bayes(Arg label)
      : label_(label), prior_(NodeF::param_shape({label}), result_type(1)) { }

    //! Creates a naive Bayes model with given prior distribution and CPDs.
    explicit naive_bayes(Arg label, const LabelF& prior)
      : label_(label), prior_(prior) {
      check_prior();
    }

    /**
     * Sets the prior factor. Must not change shape of the factor.
     */
    void prior(const F& prior) {
      prior_ = prior;
      check_prior();
    }

    /**
     * Adds a new feature or overwrites the existing one.
     * The prior must have already been set.
     */
    void add_feature(Arg feature, const FeatureF& cpd) {
      assert(feature != label_);
      domain<Arg> dom = { feature, label_ };
      if (cpd.shape() != FeatureF::param_shape(dom)) {
        throw std::invalid_argument("Invalid shape of the CPD");
      }
      cpds_[feature] = cpd;
    }

    // Queries
    //--------------------------------------------------------------------------

    //! Returns the label argument.
    Arg label() const {
      return label_;
    }

    //! Returns the features in the model.
    iterator_range<feature_iterator> features() const {
      return { feature_iterator(feature_.begin()),
               feature_iterator(feature_.end()) };
    }

    //! Returns all the arguments in the model.
    iterator_range<argument_iterator> arguments() const {
      return make_joined(make_iterator_range(&label_, &label_ + 1), features());
    }

    //! Returns the prior distribution.
    const LabelF& prior() const {
      return prior_;
    }

    //! Returns the feature CPD.
    const FeatureF& cpd(Arg v) const {
      return cpds_.at(v);
    }

    //! Returns true if the model contains the given argument.
    bool contains(Arg v) const {
      return v == label_ || cpds_.count(v);
    }

    //! Returns the prior multiplied by the likelihood of the assignment.
    void restrict(const assignment_type& a, LabelF& result) const {
      result = prior_;
      for (const auto& p : feature_) {
        if (a.count(p.first)) {
          p.second.restrict_multiply(a, result);
        }
      }
    }

    //! Returns the posterior distribution conditioned on an assignment.
    LabelF posterior(const assignment_type& a) const {
      LabelF tmp;
      restrict(a, tmp);
      tmp.normalize();
      return tmp;
    }

    //! Returns the probability of an assignment to label and features.
    result_type operator()(const assignment_type& a) const {
      result_type result = prior_(a);
      for (const auto& p : feature_) {
        result *= p.second(a);
      }
      return result;
    }

    //! Returns the log-probability of an assignment to label and features.
    real_type log(const assignment_type& a) const {
      real_type result = prior_.log(a);
      for (const auto & p : feature_) {
        result += p.second.log(a);
      }
      return result;
    }

    //! Returns the complete log-likelihood of a dataset.
    template <typename Dataset>
    typename std::enable_if<is_dataset<Dataset>::value, real_type>::type
    log(const Dataset& ds) const {
      typedef range_ll<typename LabelF::ll_type> prior_ll;
      typedef range_ll<typename FeatureF::ll_type> feature_ll;
      real_type result;
      result = prior_ll(prior_.param()).value(ds(prior_.arguments()));
      for (const auto& p : feature_) {
        const FeatureF& cpd = p.second;
        result += feature_ll(cpd.param()).value(ds(cpd.arguments()));
      }
      return result;
    }

    //! Returns the conditional log-likelihood of a dataset.
    template <typename Dataset>
    typename std::enable_if<is_dataset<Dataset>::value, real_type>::type
    real_type conditional_log(const Dataset& ds) const {
      real_type result(0);
      for (const auto& p : ds.assignments(arguments())) {
        result += posterior(p.first).log(p.first) * p.second;
      }
      return result;
    }

    //! Computes the accuracy of the predictions for a dataset.
    template <typename Dataset>
    typename std::enable_if<is_dataset<Dataset>::value, real_type>::type
    accuracy(const dataset_type& ds) const {
      argument_type label = label_var();
      real_type result(0);
      real_type weight(0);
      assignment_type a;
      F posterior;
      for (const auto& p : ds.assignments(arguments())) {
        joint(p.first, posterior);
        posterior.maximum(a);
        result += p.second * (a.at(label) == p.first[label]);
        weight += p.second;
      }
      return result / weight;
    }

    friend std::ostream& operator<<(std::ostream& out, const naive_bayes& nb) {
      out << "Prior:" << std::endl << nb.prior_ << std::endl;
      out << "CPDs: " << std::endl;
      for (const auto p : nb.feature_) {
        out << p.second;
      }
      return out;
    }

  private:
    void check_prior(const F& prior) const {
      if (prior.arguments().size() != 1) {
        throw std::invalid_argument("the prior must have exactly one argument");
      }
    }

    //! The prior distribution (empty if this naive_bayes is uninitialized).
    LabelF prior_;

    //! The map from feature arguments to CPDs.
    feature_map feature_;

  }; // class naive_bayes

} // namespace libgm

#endif
