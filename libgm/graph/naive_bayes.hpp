#ifndef LIBGM_NAIVE_BAYES_HPP
#define LIBGM_NAIVE_BAYES_HPP

#include <libgm/iterator/counting_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/range/zip.hpp>

#include <vector>

namespace libgm {

  /**
   * Implements a naive Bayes model.
   *
   * \tparam Label
   *         The factor type representing the prior.
   * \tparam Feature
   *         The factor type representing the observation model.
   */
  template <typename Label, typename Feature>
  class naive_bayes {
  public:
    static_assert(pariwise_compatible<Label, Feature>::value,
                  "The label and feature factors must be pairwise compatible.");

    // Public type declarations
    //--------------------------------------------------------------------------
  public:
    // Factor types
    using real_type    = typename Label::real_type;
    using result_type  = typename Label::result_type;
    using feature_type = typename Feature::head_index_type;

    // Constructors and initialization
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an an empty model.
    naive_bayes() { }

    //! Creates a naive Bayes model with the given prior distribution.
    explicit naive_bayes(const Label& prior)
      : prior_(prior) { }

    //! Creates a naive Bayes model with the given prior distribution.
    explicit naive_bayes(Label&& prior)
      : prior_(std::move(prior)) { }

    //! Returns the features of this model.
    iterator_range<counting_iterator>
    features() const {
      return { 0, features_.size() };
    }

    //! Returns the number of features.
    std::size_t num_features() const {
      return features_.size();
    }

    //! Returns the prior of this model.
    const Label& prior() const {
      return prior_;
    }

    //! Returns the prior of this model.
    Label& prior() {
      return prior_;
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

    // Queries
    //--------------------------------------------------------------------------

    /**
     * Returns the posterior distribution, given a vector of observations.
     * The number of observations must match the number of features.
     */
    template <typename Derived>
    Label posterior(const Eigen::DenseBase<Derived>& observations,
                    bool normalize = true) const {
      assert(observations.cols() == num_features());
      Label result = prior_;
      for (std::size_t i = 0; i < num_features(); ++i) {
        result *= feature(i).restrict_head(observations.col(i));
        if (normalize) {
          result.normalize();
        }
      }
      return result;
    }

    /**
     * Returns the logarithmic of the posterior distribution for
     * a dataset.
     */
    template <typename ObsIt, typename LabIt>
    real_type log(iterator_range<ObsIt> observations,
                  iterator_range<LabIt>& labels) const {
      real_type result = 0;
      for (const auto& p : zip(observations, labels)) {
        result += posterior(p.first).log(p.second);
      }
      return result;
    }

    /**
     * Computes the accuracy of the predictions for a dataset.
     */
    template <typename ObsIt, typename LabIt>
    real_type accuracy(iterator_range<ObsIt> observations,
                       iterator_range<LabIt>& labels) const {
      real_type result(0);
      real_type weight(0);
      for (const auto& p : zip(observations, labels)) {
        result += posterior(p.first).arg_max() == p.second;
        weight += real_type(1);
      }
      return result / weight;
    }

    friend std::ostream& operator<<(std::ostream& out, const naive_bayes& nb) {
      out << "Prior:" << std::endl << nb.prior() << std::endl;
      out << "CPDs:" << std::endl;
      for (std::size_t i = 0; i < nb.num_features(); ++i) {
        out << nb.feature(i) << std::endl;
      }
      return out;
    }

  private:
    Prior prior_;
    std::vector<Feature> features_;

  }; // class naive_bayes

} // namespace libgm

#endif
