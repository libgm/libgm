#ifndef LIBGM_NAIVE_BAYES_MLE_HPP
#define LIBGM_NAIVE_BAYES_MLE_HPP

#include <libgm/model/naive_bayes.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>

#include <vector>

namespace libgm {

  /**
   * A class that can learn naive Bayes models in a fully supervised manner.
   *
   * Models the Learner concept.
   */
  template <typename LabelF, typename FeatureF = LabelF>
  class naive_bayes_mle {
  public:
    // Learner concept types
    typedef naive_bayes<LabelF, FeatureF> model_type;
    typedef typename LabelF::real_type    real_type;

    // The algorithm parameters
    struct param_type {
      typedef typename factor_mle<LabelF>::regul_type   label_regul_type;
      typedef typename factor_mle<FeatureF>::regul_type feature_regul_type;
      label_regul_type prior_regul;
      feature_regul_type cpd_regul;
    };

    // Additional types
    typedef typename LabelF::variable_type variable_type;
    typedef std::vector<variable_type>     var_vector_type;

    /**
     * Constructs a learner with the given parameters.
     */
    explicit naive_bayes_mle(const param_type& param = param_type())
      : param_(param) { }

    /**
     * Fits a model using the supplied dataset for the given label variable
     * and feature vector.
     */
    template <typename Dataset>
    naive_bayes_mle& fit(const Dataset& ds,
                         variable_type label,
                         const var_vector_type& features) {
      factor_mle<LabelF> prior_mle(param_.prior_regul);
      factor_mle<FeatureF> cpd_mle(param_.cpd_regul);
      model_ = model_type(prior_mle(ds, {label}));
      for (variable_type v : features) {
        model_.add_feature(cpd_mle(ds, {v}, {label}));
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
