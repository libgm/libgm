#ifndef LIBGM_FUNCTIONAL_MEMBER_HPP
#define LIBGM_FUNCTIONAL_MEMBER_HPP

namespace libgm {

  /**
   * An operator that invokes the sum() member function on the given object.
   */
  struct member_sum {
    template <typename A>
    auto operator()(A&& a) const -> decltype(a.sum()) {
      return a.sum();
    }
  };

  /**
   * An operator that invokes the prod() member function on the given object.
   */
  struct member_prod {
    template <typename A>
    auto operator()(A&& a) const -> decltype(a.prod()) {
      return a.prod();
    }
  };

  /**
   * An operator that invokes the maxCoeff() function on the given object.
   */
  struct member_maxCoeff {
    template <typename A>
    auto operator()(A&& a) const -> decltype(a.maxCoeff()) {
      return a.maxCoeff();
    }
  };

  /**
   * An operator that invokes the minCoeff() function on the given object.
   */
  struct member_minCoeff {
    template <typename A>
    auto operator()(A&& a) const -> decltype(a.minCoeff()) {
      return a.minCoeff();
    }
  };

  /**
   * An operator that invokes maxCoeff(row, col) on the given object.
   */
  struct member_maxCoeffIndex {
    std::size_t* row;
    std::size_t* col;

    explicit member_maxCoeffIndex(std::size_t* row, std::size_t* col = nullptr)
      : row(row), col(col) { }

    template <typename A>
    auto operator()(A&& a) const
      -> decltype(a.maxCoeff(std::declval<std::ptrdiff_t*>(),
                             std::declval<std::ptrdiff_t*>())) {
      std::ptrdiff_t i, j;
      auto result = a.maxCoeff(&i, &j);
      if (row) { *row = i; }
      if (col) { *col = j; }
      return result;
    }
  };

  /**
   * An operator that invokes minCoeff(row, col) on the given object.
   */
  struct member_minCoeffIndex {
    std::size_t* row;
    std::size_t* col;

    explicit member_minCoeffIndex(std::size_t* row, std::size_t* col = nullptr)
      : row(row), col(col) { }

    template <typename A>
    auto operator()(A&& a) const
      -> decltype(a.minCoeff(std::declval<std::ptrdiff_t*>(),
                             std::declval<std::ptrdiff_t*>())) {
      std::ptrdiff_t i, j;
      auto result = a.minCoeff(&i, &j);
      if (row) { *row = i; }
      if (col) { *col = j; }
      return result;
    }
  };

  //! An operator that computes coefficient-wise maximum.
  struct member_max {
    template <typename A, typename B>
    auto operator()(A&& a, B&& b) -> decltype(a.max(std::forward<B>(b))) {
      return a.max(std::forward<B>(b));
    }
  };

  //! An operator that computes coefficient-wise minimum.
  struct member_min {
    template <typename A, typename B>
    auto operator()(A&& a, B&& b) -> decltype(a.min(std::forward<B>(b))) {
      return a.min(std::forward<B>(b));
    }
  };

  //! An operator that returns an array expression for the given matrix.
  struct member_array {
    template <typename M>
    auto operator()(M&& m) -> decltype(m.array()) {
      return m.array();
    }
  };

  //! An operator that returns a matrix expression for the given array.
  struct member_matrix {
    template <typename A>
    auto operator()(A&& a) -> decltype(a.matrix()) {
      return a.matrix();
    }
  };

  //! An operator that returns the parameters for the given expression.
  struct member_param {
    template <typename Expr>
    decltype(auto) operator()(Expr&& expr) {
      return std::forward<Expr>(expr).param();
    }
  };

  //! An operator that computes the exponent (of possibly shifted) object.
  template <typename T>
  struct exp_op {
    T offset;
    exp_op(T offset = T(0)) : offset(offset) { }
    template <typename A>
    auto operator()(A&& a) const -> decltype(exp(a + offset)) {
      return exp(a + offset);
    }
  };

  //! An operator that computes the (possibly shifted) log(a.sum()) of object a.
  template <typename T>
  struct log_sum_op {
    T offset;
    log_sum_op(T offset = T(0)) : offset(offset) { }
    template <typename A>
    auto operator()(A&& a) const -> decltype(log(a.sum()) + offset) {
      return log(a.sum()) + offset;
    }
  };

} // namespace libgm

#endif
