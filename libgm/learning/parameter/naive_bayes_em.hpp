#ifndef LIBGM_NAIVE_BAYES_EM_HPP
#define LIBGM_NAIVE_BAYES_EM_HPP

#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/model/naive_bayes.hpp>
#include <libgm/learning/parameter/em_parameters.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace libgm {

  /**
   * A class that learns a naive Bayes model when the label variable
   * is not observed. The objective value of this algorithm is
   * a lower-bound on the log-likelihood of the model.
   *
   * \tparam LabelF a type representing the label prior in the probability space
   * \tparam FeatureF a type representing the feature CPD
   */
  template <typename LabelF, typename FeatureF = LabelF>
  class naive_bayes_em {
    typedef typename FeatureF::mle_type::regul_type regul_type;

  public:
    // Learner concept types
    typedef naive_bayes<LabelF, FeatureF>        model_type;
    typedef typename LabelF::real_type           real_type;
    typedef em_parameters<regul_type, real_type> param_type;

    // Other types
    typedef typename LabelF::variable_type variable_type;
    typedef std::vector<variable_type>     var_vector_type;

    /**
     * Constructs a naive Bayes learner with given parameters.
     */
    explicit naive_bayes_em(const param_type& param = param_type())
      : param_(param) { }

    /**
     * Fits a model using the supplied dataset for the given label variable
     * and feature vector.
     */
    template <typename Dataset>
    model_type& fit(const Dataset& ds,
                    variable_type label,
                    const var_vector_type& features) {
      reset(&ds, label, features);
      return fit();
    }

    /**
     * Fits a model that was previously initialized using reset().
     */
    model_type& fit() {
      do {
        iterate();
        if (param_.verbose) {
          std::cerr << "Iteration " << iteration_ << ": ll >= " << objective_
                    << std::endl;
        }
      } while (!converged_ && iteration_ < param_.max_iter);
      return model_;
    }

    /**
     * Initializes the model for table-like factors by drawing the
     * parameters uniormly at random and sets the dataset for training.
     */
    template <typename Dataset>
    void reset(const Dataset* ds,
               variable_type label,
               const var_vector_type& features) {
      // initialize the model
      uniform_table_generator<LabelF> prior_gen;
      uniform_table_generator<FeatureF> cpd_gen;
      std::mt19937 rng(param_.seed);
      model_type model(prior_gen({label}, rng).normalize());
      for (variable_type feature : features) {
        model_.add_feature(cpd_gen({feature}, {label}, rng));
      }
      reset(ds, std::move(model));
    }

    /**
     * Initializes the estimate to the given model and sets the
     * dataset for training.
     */
    template <typename Dataset>
    void reset(const Dataset* ds, model_type&& model) {
      model_ = std::move(model);
      updater_ = updater<Dataset>(ds, &model_, param_.regul);
      weight_ = ds->weight();
      iteration_ = 0;
      objective_ = std::numeric_limits<real_type>::infinity();
      converged_ = false;
    }

    //! Performs one iteration of EM.
    real_type iterate() {
      real_type prev = objective_;
      objective_ = updater_();
      converged_ = (objective_ - prev) / weight < param_.tol;
      ++iteration_;
      return objective_;
    }

    //! Returns the estimated model.
    model_type& model() { return model_; }

    //! Returns the number of iterations.
    size_t iteration() const { return iteration_; }

    //! Returns the objective value.
    real_type objective() const { return objective_; }

    //! Returns true if the iteration has converged.
    bool converged() const { return converged_; }

  private:
    /**
     * A class that performs EM updates. Implementing this functionality here
     * instead of the enclosing class erases the type of the dataset.
     */
    template <typename Dataset>
    class updater {
    public:
      updater(const Dataset* ds, model_type* model, const regul_type& regul)
        : ds_(ds), model_(model), regul_(regul) {
        label_ = model->label();
        features_.assign(model->features().begin(), model->features().end());
      }

      real_type operator()() {
        // initialize the iterators and MLEs over (feature, label) domains
        size_t n = features_.size();
        std::vector<typename Dataset::const_iterator> it(n);
        std::vector<typename FeatureF::mle_type> mle(n, regul_);
        for (size_t i = 0; i < n; ++i) {
          it[i]  = (*ds_)({features_[i], label_}).begin();
          mle[i].initialize(FeatureF::param_shape({features_[i], label_}));
        }

        // expectation: the probability of the labels given each datapoint
        // maximization: accumulate the new prior and the feature CPDs
        real_type bound(0);
        LabelF prior(label_, result_type(0));
        LabelF ptail;
        for (const auto& p : ds_->assignments(features)) {
          model_->restrict(p.first, ptail);
          real_type norm = ptail.marginal();
          bound += p.second * std::log(norm);
          ptail *= p.second / norm;
          prior.param() += ptail.param();
          for (size_t i = 0; i < features.size(); ++i) {
            mle_.process(it[i]->first, ptail.param());
            ++it[i];
          }
        }

        // set the parameters of the new model
        model_->prior(prior.normalize());
        for (size_t i = 0; i < n; ++i) {
          FeatureF joint({features_[i], label_}, mle[i].param());
          model_->add_feature(joint.conditional({label_}));
        }

        return bound;
      }

    private:
      const Dataset* ds_;   //!< The pointer to the underlying dataset.
      model_type* model_;   //!< The pointer to the updated model.
      variable_type label_; //!< The hidden label.
      std::vector<variable_type> features_; //! The observed features.

    }; // class updater

  private:
    param_type param_;      //!< The parameters of the learner.
    model_type model_;      //!< The learned model.
    std::function<real_type()> updater_; //!< Performs the iteration updates.
    real_type weight_;      //!< The total weight of the samples in the dataset.
    std::size_t iteration_; //!< The current iteration.
    real_type objective_;   //!< The objective value.
    bool converged_;        //!< If true, the estimator has converged.

  }; // class naive_bayes_em

} // namespace libgm

#endif
