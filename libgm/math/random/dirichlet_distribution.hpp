#pragma onace

#include <random>

#include <Eigen/Core>

namespace libgm {

/**
 * Dirichlet(n, alpha) distribution.
 *
 * \ingroup math_random
 */
template <typename T = double>
class DirichletDistribution {
public:
  /// The type representing the parameter set.
  typedef Eigen::Array<T, Eigen::Dynamic, 1> param_type;

  /// The type representing a random draw.
  typedef std::vector<T> result_type;

  /**
   * Constructs a Dirichlet distribution with the given shape parameters.
   * The dimensionality is specified implicitly by alpha.size().
   */
  explicit DirichletDistribution(const param_type& alpha)
    : alpha_(alpha) {
    assert(!alpha.empty());
    for (size_t i = 0; i < alpha.size(); ++i) {
      gamma_.emplace_back(alpha[i], T(1));
    }
  }

  /**
   * Constructs a Dirichlet distribution with given dimentionality n and
   * a fixed alpha.
   */
  DirichletDistribution(size_t n, T alpha)
    : gamma_(n, std::gamma_distribution<T>(alpha, T(1))) {
    alpha_.setScalar(n, alpha);
  }

  /// Returns the dimensionality of the random vector.
  size_t n() const {
    return alpha_.size();
  }

  /// Returns the shape parameters.
  const param_type& param() const {
    return alpha_;
  }

  /// Generates a random vector distributed as Dirichlet(alpha)
  template <typename Generator>
  result_type operator()(Generator& rng) {
    result_type result(alpha_.size());
    T total(0);
    for (size_t i = 0; i < alpha_.size(); ++i) {
      result[i] = gamma_[i](rng);
      total += result[i];
    }
    for (size_t i = 0; i < alpha_.size(); ++i) {
      result[i] /= total;
    }
    return result;
  }

private:
  /// Shape parameters.
  param_type alpha_;

  /// The gamma distributions.
  std::vector<gamma_distribution<T> > gamma_;

}; // class DirichletDistribution

} // namespace libgm
