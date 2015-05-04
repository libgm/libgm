#ifndef LIBGM_FUNCTIONAL_EIGEN_HPP
#define LIBGM_FUNCTIONAL_EIGEN_HPP

namespace libgm {

  //! An operator that invokes the sum() function on the given object.
  struct sum_op {
    template <typename A>
    auto operator()(A&& a) const -> decltype(a.sum()) {
      return a.sum();
    }
  };

  //! An operator that invokes the prod() function on the given object.
  struct prod_op {
    template <typename A>
    auto operator()(A&& a) const -> decltype(a.prod()) {
      return a.prod();
    }
  };

  //! An operator that invokes the maxCoeff() function on the given object.
  struct max_coeff_op {
    template <typename A>
    auto operator()(A&& a) const -> decltype(a.maxCoeff()) {
      return a.maxCoeff();
    }
  };

  //! An operator that invokes the minCoeff() function on the given object.
  struct min_coeff_op {
    template <typename A>
    auto operator()(A&& a) const -> decltype(a.minCoeff()) {
      return a.minCoeff();
    }
  };

  //! An operator that computes the exponent (of possibly shifted) object.
  template <typename T>
  struct exp_op {
    T offset;
    explicit exp_op(T offset = T(0)) : offset(offset) { }
    template <typename A>
    auto operator()(A&& a) const -> decltype(exp(a + offset)) {
      return exp(a + offset);
    }
  };

  //! An operator that computes the (possibly shifted) log(a.sum()) of object a.
  template <typename T>
  struct log_sum_op {
    T offset;
    explicit log_sum_op(T offset = T(0)) : offset(offset) { }
    template <typename A>
    auto operator()(A&& a) const -> decltype(log(a.sum()) + offset) {
      return log(a.sum()) + offset;
    }
  };


} // namespace libgm

#endif
