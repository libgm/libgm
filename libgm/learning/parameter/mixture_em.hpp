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
   * A class that learns a mixture distribution using Expectation Maximization.
   * For this algorith, the objective value of this algorithm is a lower-bound
   * on the log-likelihood of the model.
   *
   * \tparam F the mixture component type
   */
  template <typename Arg, typename F>
  class mixture_em {
  public:
    // Learner concept types
    using model_type = mixture<F>;
    using real_type  = typename F::real_type;
    using regul_type = typename F::mle_type::regul_type;

//     // Other types
//     typedef typename F::argument_type argument_type;
//     typedef typename F::domain_type   domain_type;
//     typedef typename F::result_type   result_type;
//     typedef typename F::mle_type      mle_type;
//     typedef typename F::ll_type       ll_type;

    /**
     * Constructs a mixture EM learner with the given number of components
     * and regularization parameter.
     */
    explicit mixture_em(std::size_t k,
                        const regul_type& regul = regul_type())
      : k_(k), regul_(regul) { }

    /**
     * Fits a model using the suppplied dataset for the given arguments.
     */
    mixture_em& fit(const dataset<real_type>& ds, const domain<Arg>& args,
                    const convergence_control<real_type>& convergence) {

    }

    /**
     * Fits a model using the supplied dataset and all argument in the dataset.
     */
    mixture_em& fit(const dataset<real_type>& ds,
                    const convergence_control<real_type>& convergence) {
    }

    /**
     * Initializes the solution.
     */
    mixture_em& initialize(model_type&& model) {
      model_ = std::move(model);
      return *this;
    }

    /**
     * Performs an update using the given dataset.
     */
    real_type iterate(std::size_t n = 1) {
      // compute the log-likelihoods for each component
      dense_matrix<real_type> ll(ds.size(), components_);
      for (std::size_t k = 0; i < components_; ++k) {
        ll.col(k) = typename F::ll_type(model_.param(k)).values(data);
      }

      // compute the normalized probabilities and the bound
      dense_matrix<real_type> p(ds.size(), components_);
      real_type bound(0);
      for (std::size_t i = 0; i < ds.size(); ++i) {
        bound += weights[i] * log_sum_exp(ll.row(i), p.row(i));
        p.row(i) *= wights[i];
      }

      // recompute the components
      typename F::mle_type mle(regul_);
      for (std::size_t k = 0; k < compoennts_; ++i) {
        model_.param(k) = mle(data, p.col(k));
      }
      model_.normalize();

      return bound;
    }



//     /**
//      * Fits a model that was previusly initialized using reset().
//      */
//     model_type& fit() {
//       do {
//         iterate();
//         if (param_.verbose) {
//           std::cerr << "Iteration " << iteration_ << ": ll >= " << objective_
//                     << std::endl;
//         }
//       } while (!converged_ && iteration_ < param_.max_iter);
//       return model_;
//     }

//     /**
//      * Initializes a mixture Gaussian model.
//      * Sets the mean of each component to a random data point and the covariance
//      * to the covariance of the whole population.
//      */
//     template <typename Dataset>
//     void reset(const Dataset* ds, const domain_type& args, std::size_t k) {
//       // compute the covariance and set the random means
//       factor_mle<F> mle(param_.regul);
//       experimental::mixture<F> model(mle(*ds, args), k);
//       std::mt19937 rng(param_.seed);
//       std::size_t i = 0;
//       for (size_t row : randperm(rng, ds->size(), k)) {
//         model.param(i++).mean = ds->sample(row, args).first;
//       }
//       model.normalize();
//       reset(ds, std::move(model));
//     }

//     /**
//      * Initilizes the estimate to the given model and sets the dataset
//      * for training.
//      */
//     template <typename Dataset>
//     void reset(const Dataset* ds, model_type&& model) {
//       model_ = std::move(model);
//       updater_ = updater<Dataset>(ds, &model_, param_.regul);
//       weight_ = ds->weight();
//       iteration_ = 0;
//       objective_ = -std::numeric_limits<real_type>::infinity();
//       converged_ = false;
//     }

//     //! Performs one iteration of EM
//     real_type iterate() {
//       real_type prev = objective_;
//       objective_ = updater_();
//       converged_ = (objective_ - prev) / weight_ < param_.tol;
//       ++iteration_;
//       return objective_;
//     }

    //! Returns the estimated model.
    model_type& model() {
      return model_;
    }

    //! Returns the number of iterations.
    size_t iteration() const {
      return iteration_;
    }

    //! Returns the objective value.
    real_type objective() const {
      return objective_;
    }

    //! Returns true if the iteration has converged.
    bool converged() const {
      return converged_;
    }

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
