#pragma once

#include <libgm/factor/interfaces.hpp>

namespace libgm {

template <typename T>
struct CanonicalGaussian

  // Constructs the implementation object
  MomentGaussian();
  ~MomentGaussian();

  // Custom accessors
  const Eigen::Vector<REAL>& mean() const;

protected:
  template <typename IMPL>
  struct Impl;

  std::unique_ptr<Impl> impl_;
};

class Impl {
  void multiply_in(const Logarithmic<T>& other) {
    ll += other.ll;
  }

  void multiply_in(const CanonicalGaussian& other) {
    ll += other.ll;
    eta += other.eta;
    lambda += other.lambda;
  }

  CanonicalGausasian multiply(const CanonicalGaussian& other) {

  }

};

}  // namespace libgm
