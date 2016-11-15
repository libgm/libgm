#ifndef LIBGM_MIXTURE_EM_HPP
#define LIBGM_MIXTURE_EM_HPP

#include <libgm/learning/parameter/em_parameters.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>
#include <libgm/math/numeric.hpp>
#include <libgm/math/random/permutations.hpp>
#include <libgm/model/mixture.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace libgm {

  /**
   * // TODO: this needs to be fixed
   * A class that learns a mixture model using Expectation Maximization.
   * The objective value of this algorithm is a lower-bound on the
   * log-likelihood of the model.
   *
   * \tparam F the mixture component type
   */
  template <typename F>
  class mixture_em {
    typedef typename F::mle_type::regul_type regul_type;

  public:
    // Learner concept types
    typedef experimental::mixture<F>  model_type;
    typedef typename F::real_type     real_type;
    typedef em_parameters<regul_type> param_type;

    // Other types
    typedef typename F::argument_type argument_type;
    typedef typename F::domain_type   domain_type;
    typedef typename F::result_type   result_type;
    typedef typename F::mle_type      mle_type;
    typedef typename F::ll_type       ll_type;

    /**
     * Constructs a mixture EM learner with given prameters.
     */
    explicit mixture_em(const param_type& param = param_type())
      : param_(param) { }

    /**
     * Fits a model using the supplied dataset, k mixture components,
     * and all the arguments in the dataset.
     */
    template <typename Dataset>
    model_type& fit(const Dataset& ds, std::size_t k) {
      return fit(ds, ds.arguments(), k);
    }

    /**
     * Fits a model using the supplied dataset, k mixture components,
     * and given arguments that must be present in the dataset.
     */
    template <typename Dataset>
    model_type& fit(const Dataset& ds, const domain_type& args, std::size_t k) {
      reset(&ds, args, k);
      return fit();
    }

    /**
     * Fits a model that was previusly initialized using reset().
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
     * Initializes a mixture Gaussian model.
     * Sets the mean of each component to a random data point and the covariance
     * to the covariance of the whole population.
     */
    template <typename Dataset>
    void reset(const Dataset* ds, const domain_type& args, std::size_t k) {
      // compute the covariance and set the random means
      factor_mle<F> mle(param_.regul);
      experimental::mixture<F> model(mle(*ds, args), k);
      std::mt19937 rng(param_.seed);
      std::size_t i = 0;
      for (size_t row : randperm(rng, ds->size(), k)) {
        model.param(i++).mean = ds->sample(row, args).first;
      }
      model.normalize();
      reset(ds, std::move(model));
    }

    /**
     * Initilizes the estimate to the given model and sets the dataset
     * for training.
     */
    template <typename Dataset>
    void reset(const Dataset* ds, model_type&& model) {
      model_ = std::move(model);
      updater_ = updater<Dataset>(ds, &model_, param_.regul);
      weight_ = ds->weight();
      iteration_ = 0;
      objective_ = -std::numeric_limits<real_type>::infinity();
      converged_ = false;
    }

    //! Performs one iteration of EM
    real_type iterate() {
      real_type prev = objective_;
      objective_ = updater_();
      converged_ = (objective_ - prev) / weight_ < param_.tol;
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
     * A class that performs EM updates.
     */
    template <typename Dataset>
    class updater {
    public:
      updater(const Dataset* ds, model_type* model, const regul_type& regul)
        : ds_(ds), model_(model), mle_(model->size(), mle_type(regul)) { }

      real_type operator()() {
        std::size_t k = model_->size();

        // initialize the log-likelihood evaluator and the MLE of each component
        std::vector<ll_type> ll;
        for (std::size_t i = 0; i < k; ++i) {
          ll.emplace_back(model_->param(i));
          mle_[i].initialize(model_->param_shape());
        }

        // iterate once over the entire dataset
        std::vector<real_type> p(k), logp(k);
        real_type bound(0);
        for (const auto& s : ds_->samples(model_->arguments())) {
          // compute the probability of the sample under each component
          for (std::size_t i = 0; i < k; ++i) {
            logp[i] = ll[i].value(s.first);
          }
          bound += s.second * log_sum_exp(logp.begin(), logp.end(), p.begin());

          // add the sample weighted by the probability under each component
          for (std::size_t i = 0; i < k; ++i) {
            mle_[i].process(s.first, p[i] * s.second);
          }
        }

        // recompute the components
        for (size_t i = 0; i < k; ++i) {
          model_->param(i) = mle_[i].param();
          model_->factor(i) *= result_type(mle_[i].weight());
        }
        model_->normalize();

        return bound;
      }

    private:
      const Dataset* ds_;  //!< The pointer to the underlying dataset.
      model_type* model_;  //!< The pointer to the updated model.
      std::vector<mle_type> mle_; //!< A vector of component estimators.

    }; // class updater

  private:
    param_type param_;      //!< The parameters of the learner.
    model_type model_;      //!< The learned model.
    std::function<real_type()> updater_; //!< Performs the iteration updates.
    real_type weight_;      //!< The total weight of the samples in the dataset.
    std::size_t iteration_; //!< The current iteration.
    real_type objective_;   //!< The objective value.
    bool converged_;        //!< If true, the estimator has converged.

  }; // class mixture_em

} // namespace libgm

#endif
