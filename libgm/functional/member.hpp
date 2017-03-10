#ifndef LIBGM_FUNCTIONAL_MEMBER_HPP
#define LIBGM_FUNCTIONAL_MEMBER_HPP

#include <Eigen/Core>

#include <cmath>

namespace libgm {

  /**
   * Invokes the at function on the given container.
   */
  struct member_at {
    template <typename Container, typename Index>
    decltype(auto) operator()(Container&& c, Index&& index) const {
      return std::forward<Container>(c).at(std::forward<Index>(index));
    }
  };

  /**
   * Invokes the subscript function on the given container.
   */
  struct member_subscript {
    template <typename Container, typename Index>
    decltype(auto) operator()(Container&& c, Index&& index) const {
      return std::forward<Container>(c)[std::forward<Index>(index)];
    }
  };

  /**
   * Invokes the count function on the given container.
   */
  struct member_count {
    template <typename Container, typename Key>
    auto operator()(Container&& c, Key&& key) const {
      return std::forward<Container>(c).count(std::forward<Key>(key));
    }
  };

  /**
   * An operator that invokes the marginal() member function on the given
   * object.
   */
  struct member_marginal {
    template <typename F, typename... Args>
    auto operator()(F&& f, Args&&... args) {
      return std::forward<F>(f).marginal(std::forward<Args>(args)...);
    }
  };

  /**
   * An operator that invokes the maximum() member function on the given
   * object.
   */
  struct member_maximum {
    template <typename F, typename... Args>
    auto operator()(F&& f, Args&&... args) {
      return std::forward<F>(f).maximum(std::forward<Args>(args)...);
    }
  };

  /**
   * An operator that invokes the minimum() member function on the given
   * object.
   */
  struct member_minimum {
    template <typename F, typename... Args>
    auto operator()(F&& f, Args&&... args) {
      return std::forward<F>(f).minimum(std::forward<Args>(args)...);
    }
  };

  /**
   * An operator that invokes the head() member function on the
   * given object.
   */
  struct member_head {
    template <typename F, typename... Args>
    auto operator()(F&& f, Args&&... args) {
      return std::forward<F>(f).head(std::forward<Args>(args)...);
    }
  };

  /**
   * An operator that invokes the tail() member function on the
   * given object.
   */
  struct member_tail {
    template <typename F, typename... Args>
    auto operator()(F&& f, Args&&... args) {
      return std::forward<F>(f).tail(std::forward<Args>(args)...);
    }
  };

  /**
   * An operator that inovkes the restrict() member function on the given
   * object.
   */
  struct member_restrict {
    template <typename F, typename... Args>
    auto operator()(F&& f, Args&&... args) {
      return std::forward<F>(f).restrict(std::forward<Args>(args)...);
    }
  };

  /**
   * An operator that inovkes the restrict_head() member function on the given
   * object.
   */
  struct member_restrict_head {
    template <typename F, typename... Args>
    auto operator()(F&& f, Args&&... args) {
      return std::forward<F>(f).restrict_head(std::forward<Args>(args)...);
    }
  };

  /**
   * An operator that inovkes the restrict_tail() member function on the given
   * object.
   */
  struct member_restrict_tail {
    template <typename F, typename... Args>
    auto operator()(F&& f, Args&&... args) {
      return std::forward<F>(f).restrict_tail(std::forward<Args>(args)...);
    }
  };


  /**
   * An operator that invokes the sum() member function on the given object.
   */
  struct member_sum {
    template <typename A>
    decltype(auto) operator()(A&& a) const {
      return a.sum();
    }
  };

  /**
   * An operator that invokes the prod() member function on the given object.
   */
  struct member_prod {
    template <typename A>
    decltype(auto) operator()(A&& a) const {
      return a.prod();
    }
  };

  /**
   * An operator that computes log-sum-exp on the given object.
   * This does not work for vectorwise operations.
   * \sa member_logSumExpVectorwise
   */
  struct member_logSumExp {
    template <typename Expr>
    auto operator()(Expr&& e) const {
      decltype(auto) array = e.array();
      auto offset = array.maxCoeff();
      return log((array - offset).exp().sum()) + offset;
    }
  };

  /**
   * An operator that computes log-sum-exp for vectorwise operations.
   * \sa member_logSumExp
   */
  struct member_logSumExpVectorwise {
    template <typename Expr>
    auto operator()(const Eigen::VectorwiseOp<Expr, Eigen::Vertical>& e) const {
      decltype(auto) array = e._expression().array();
      auto offset = array.maxCoeff();
      return log((array - offset).exp().colwise().sum()) + offset;
    }

    template <typename Expr>
    auto operator()(const Eigen::VectorwiseOp<Expr, Eigen::Horizontal>& e) const {
      decltype(auto) array = e._expression().array();
      auto offset = array.maxCoeff();
      return log((array - offset).exp().rowwise().sum()) + offset;
    }
  };

  /**
   * An operator that invokes the maxCoeff() function on the given object.
   */
  struct member_maxCoeff {
    template <typename A>
    decltype(auto) operator()(A&& a) const {
      return a.maxCoeff();
    }
  };

  /**
   * An operator that invokes the minCoeff() function on the given object.
   */
  struct member_minCoeff {
    template <typename A>
    decltype(auto) operator()(A&& a) const {
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
    auto operator()(A&& a) const {
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
    auto operator()(A&& a) const {
      std::ptrdiff_t i, j;
      auto result = a.minCoeff(&i, &j);
      if (row) { *row = i; }
      if (col) { *col = j; }
      return result;
    }
  };

  //! An operator that computes coefficient-wise maximum.
  struct member_max {
    template <typename F, typename... Args>
    decltype(auto) operator()(F&& f, Args&&... args) const {
      return std::forward<F>(f).max(std::forward<Args>(args)...);
    }
  };

  //! An operator that computes coefficient-wise minimum.
  struct member_min {
    template <typename F, typename... Args>
    decltype(auto) operator()(F&& f, Args&&... args) const {
      return std::forward<F>(f).min(std::forward<Args>(args)...);
    }
  };

  //! An operator that returns an array expression for the given matrix.
  struct member_array {
    template <typename M>
    decltype(auto) operator()(M&& m) const {
      return m.array();
    }
  };

  //! An operator that returns an expression casting the object to a type.
  template <typename RealType>
  struct member_cast {
    template <typename A>
    decltype(auto) operator()(A&& a) const {
      return a.template cast<RealType>();
    }
  };

  //! An operator that returns the parameters for the given expression.
  struct member_param {
    template <typename Expr>
    decltype(auto) operator()(Expr&& expr) {
      return std::forward<Expr>(expr).param();
    }
  };

  //! An operator that returns true if the given expression aliases parameters.
  template <typename Param>
  struct member_alias {
    const Param& param;
    explicit member_alias(const Param& param) : param(param) { }
    template <typename Expr>
    bool operator()(Expr&& expr) {
      return std::forward<Expr>(expr).alias(param);
    }
  };

  /**
   * Creates a member_alias object, automatically deducing its type.
   * \relates member_alias
   */
  template <typename Param>
  member_alias<Param> make_member_alias(const Param& param) {
    return member_alias<Param>(param);
  }

  //! An operator that computes the exponent (of possibly shifted) object.
  //! \deprecated remove once we migrate to new factors
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
  //! \deprecated remove once we migrate to new factors
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
