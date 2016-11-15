#ifndef LIBGM_NAIVE_BAYES_MLE_HPP
#define LIBGM_NAIVE_BAYES_MLE_HPP

#include <libgm/model/naive_bayes.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>

namespace libgm {

  /**
   * A class that can learn naive Bayes models in a fully supervised manner.
   *
   * Models the Learner concept.
   */
  template <typename Arg, typename LabelF, typename FeatureF = LabelF>
  class naive_bayes_mle {
  public:
    // Learner concept types
    using model_type = naive_bayes<Arg, LabelF, FeatureF>;
    using real_type  = typename LabelF::real_type;

    // The algorithm parameters
    struct param_type {
      typename LabelF::mle_type::regul_type prior;
      typename FeatureF::mle_type::regul_type cpd;
    };

    /**
     * Constructs a learner with the given parameters.
     */
    explicit naive_bayes_mle(const param_type& param = param_type())
      : param_(param) { }

    /**
     * Fits a model using the supplied dataset for the given label argument
     * and feature vector.
     */
    template <typename Dataset>
    naive_bayes_mle& fit(const Dataset& ds, Arg y, const domain<Arg>& x) {
      typename LabelF::mle_type prior_mle(param_.prior);
      typename FeatureF::mle_type cpd_mle(param_.cpd);
      model_ = model_type(label, prior_mle(ds.project(y), LabelF::shape(y)));
      for (Arg v : x) {
        model_.add_feature(v, cpd_mle(ds.project(v), FeatureF::shape(v)));
      }
      return *this;
    }

    //! Returns the trained model
    model_type& model() { return model_; }

  private:
    param_type param_; //!< The regularization parameters.
    model_type model_; //!< The trained model.

  }; // class naive_bayes_mle

} // namespace libgm

#endif
