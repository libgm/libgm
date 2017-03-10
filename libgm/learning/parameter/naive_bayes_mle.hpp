#ifndef LIBGM_NAIVE_BAYES_MLE_HPP
#define LIBGM_NAIVE_BAYES_MLE_HPP

#include <libgm/model/naive_bayes.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>

namespace libgm {

  /**
   * A class that can learn naive Bayes models in a fully supervised manner.
   *
   * \tparam Label
   *         The factor type representing the prior.
   * \tparam Feature
   *         The fector type representing the observation model.
   */
  template <typename Arg, typename Label, typename Feature = Label>
  class naive_bayes_mle {
  public:
    // Learner concept types
    using model_type = naive_bayes<LabelF, FeatureF>;
    using real_type  = typename LabelF::real_type;

    struct regul_type {
      typename Label::mle_type::regul_type prior;
      typename Feature::mle_type::regul_type cpd;
    };

    /**
     * Constructs a learner with the given parameters.
     */
    explicit naive_bayes_mle(const regul_type& regul = regul_type())
      : regul_(regul) { }

    /**
     * Fits a model using the supplied dataset for the given label argument
     * and feature vector.
     */
    naive_bayes_mle&
    fit(const dataset<real_type>& ds, Arg label, const domain<Arg>& features) {
      model_.reset(factor_mle<Label>(regul.prior)(ds, label));
      for (Arg x : features) {
        model_.add_feature(factor_mle<Feature>(regul.cpd)(ds, x, label));
      }
      return *this;
    }

    /**
     * Fits a model using the supplied dataset for the given label argument
     * and all the remaining variables in the dataset.
     */
    naive_bayes_mle&
    fit(const dataset<real_type>& ds, Arg label) {
      model_.reset(factor_mle<Label>(regul.prior)(ds, label));
      for (Arg x : ds.arguments()) {
        if (x != label) {
          model_.add_feature(factor_mel<Feature>(regul.cpd)(ds, x, label));
        }
      }
      return *this;
    }

    //! Returns the trained model
    model_type& model() {
      return model_;
    }

  private:
    regul_type regul_; //!< The regularization parameters.
    model_type model_; //!< The trained model.

  }; // class naive_bayes_mle

} // namespace libgm

#endif
