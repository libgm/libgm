#ifndef LIBGM_HIDDEN_MARKOV_MODEL_HPP
#define LIBGM_HIDDEN_MARKOV_MODEL_HPP

#include <libgm/iterator/counting_iterator.hpp>
#include <libgm/range/iterator_range.hpp>

#include <vector>

namespace libgm {

  /**
   * Implements a first-order hidden Markov model with multiple independent
   * observations at each step.
   *
   * \tparam Prior
   *         The factor type representing the initial state and node beliefs.
   * \tparam Transition
   *         The factor type representing the transition distribution and
   *         edge beliefs.
   * \tparam Feature
   *         The factor type representing the observation model.
   */
  template <typename Prior, typename Transition, typename Feature = Transition>
  class hidden_markov_model {

    // Public functions
    //--------------------------------------------------------------------------
  public:

    //! Default constructor. Creates an empty model.
    hidden_markov_model() { }

    //! Constructs a hidden Markov model with the specified properties.
    hidden_markov_model(const Prior& prior, const Transition& transition)
      : prior_(prior), transition_(transition) { }

    //! Constructs a hidden Markov model with the specified properties.
    hidden_markov_model(Prior&& prior, Transition&& transition)
      : prior_(std::move(prior)), transition_(std::move(transition)) { }

    //! Returns the features of this model.
    iterator_range<counting_iterator>
    features() const {
      return { 0, features_.size() };
    }

    //! Returns the state indices of this model.
    iterator_range<counting_iterator>
    states(std::size_t start, std::size_t end) const {
      assert(start <= end);
      return { start, end };
    }

    //! Returns the number of features.
    std::size_t num_features() const {
      return features_.size();
    }

    //! Returns the prior of this model.
    const Prior& prior() const {
      return prior_;
    }

    //! Returns the prior of this model.
    Prior& prior() {
      return prior_;
    }

    //! Returns the transition distribution of this model.
    const Transition& transition() const {
      return transition_;
    }

    //! Returns the transition distribution of this model.
    Transition& transition() {
      return transition_;
    }

    //! Returns the conditional distribution for the given feature.
    const Feature& feature(std::size_t i) const {
      return features_[i];
    }

    //! Returns the conditional distribution for the given feature.
    Feature& feature(std::size_t i) {
      return features_[i];
    }

    //! Adds a new feature.
    void add_feature(const Feature& f) {
      features_.push_back(f);
    }

    //! Adds a new feature.
    void add_feature(Feature&& f) {
      features_.push_back(std::move(f));
    }

    // Private members
    //--------------------------------------------------------------------------
  private:
    Prior prior_;
    Transition transition_;
    std::vector<Feature> features_;

  }; // class hidden_markov_model

} // namespace libgm

#endif
