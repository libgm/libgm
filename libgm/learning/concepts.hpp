#ifndef LIBGM_LEARNING_CONCEPTS_HPP
#define LIBGM_LEARNING_CONCEPTS_HPP

namespace libgm {

  /**
   * A concept that represents a machine learning algorithm that is able
   * to learn a model from the supplied dataset. Note that the learn()
   * functions are allowed to modify the state of the learner, so they
   * should not be executed in parallel on the same learner object
   * (they can be executed on copies).
   *
   * \todo Make this inherit CopyConstructible.
   */
  template <typename L>
  struct Learner {
    //! The model type produced by this learner
    typedef typename L::model_type model_type;

    //! The type this learner uses to represent real numbers.
    //! Should match the real type of the model.
    typedef typename L::real_type real_type;

    //! The dataset type used to train the model
    typedef typename L::dataset_type dataset_type;

    //! A struct that represents all the tuning parameters for the learning
    typedef typename L::param_type param_type;

    /**
     * Learns a model using the supplied dataset and default parameter set.
     * \return the objective value of the trained model
     */
    real_type learn(const dataset_type& ds,
                    model_type& model);

    /**
     * Learns a model using the supplied dataset and parameters.
     * \return the objective value of the trained model
     */
    real_type learn(const dataset_type& ds,
                    const param_type& params,
                    model_type& model);

  }; // concept Learner

  /**
   * A concept that represents an iterative machine learning algorithm.
   * In addition to the functions provided by the Learner concept, this
   * concept provides function to iterate on the model.
   */
  template <typename L>
  struct IterativeLearner : public Learner<L> {

    /**
     * Performs one iteration on the model. It is assumed that the
     * dataset and parameters were passed in a separate initialization
     * function.
     * \return The current objective value of the model
     */
    real_type iterate();

    /**
     * Returns the current estimate of the model.
     */
    const model_type& current_model();

  }; // concept IterativeLearner

} // namespace libgm

#endif
