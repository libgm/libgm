namespace libgm {

template <typename T>
struct Impl<MomentGaussian<T>> {
  /// The type of the LLT Cholesky decomposition object.
  using CholeskyType = Eigen::LLT<DenseMatrix<T>>;
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  /// The conditional mean.
  VectorType mean;

  /// The covariance matrix.
  MatrixType cov;

  /// The coefficient matrix.
  MatrixType coef;

  /// The log-multiplier.
  T lm;

  unsigned arity() const {
    return coef.rows() + coef.cols();
  }

  unsigned head_arity() const {
    return coef.rows();
  }

  unsigned tail_arity() const {
    return coef.cols();
  }

  bool is_marginal() const {
    return tail_arity() == 0;
  }

  bool equal(const MomentGaussian<T>& other) const {
    const Impl& x = other.impl();
    return lm == x.lm && mean == x.mean && cov == x.cov && coef == x.coef;
  }

  void print(std::ostream& out) const {
   out << mean << std::endl
       << cov << std::endl
       << coef << std::endl
       << lm;
  }

  void multiply_in(const Exp<T>& x) {
    lm += log(x);
  }

  void divide_in(const Exp<T>& x) {
    lm -= log(x);
  }

  MomentGaussian<T> multiply(const Exp<T>& x) const {
    return std::make_unique<Impl>(mean, cov, coef, lm + log(x));
  }

  MomentGaussian<T> divide(const Exp<T>& x) const {
    return std::make_unqiue<Impl>(mean, cov, coef, lm - log(x));
  }

  Exp<T> marginal() const {
    return {lm, log_tag()};
  }

  MomentGaussian<T> marginal_front(unsigned n) const {
    return {mean.head(n), cov.upperLeftCorner(n, n), coef.topRows(n), lm};
  }

  MomentGaussian<T> marginal_back(unsigned n) const {
    return {mean.tail(n), cov.lowerRightCorner(n, n), coef.bottomRows(n), lm};
  }

  MomentGaussian<T> marginal_list(const IndexList& i) const {
    return {sub(mean, i), sub(cov, i, i), rows(coef, i), lm};
  }

  Exp<T> maximum(Assignment* a) const {
    if (a) *a = mean;
    CholeskyType chol;
    chol.solve(cov);
    return {(-std::log(two_pi<T>()) * head_size() - logdet(chol)) / 2 + lm, log_tag()};
  }

  MomentGaussian<T> maximum_front(unsigned n) const {
    auto result = marginal_front(n);
    result.lm += maximum().lv - result.maximum().lv;
    return result;
  }

  MomentGaussian<T> maximum_back(unsigned n) const {
    auto result = marginal_back(n);
    result.lm += maximum().lv - result.maximum().lv;
    return result;
  }

  MomentGaussian<T> maximum_list(const IndexList& list) const {
    auto result = marginal(list);
    result.lm += log(maximum()) - log(result.maximum());
    return result;
  }

  bool normalizable() const {
    return is_marginal();
  }

  MomentGaussian<T> conditional(unsigned nhead) const {

  }

  MomentGaussian<T> restrict_dims(const Dims& i, const Assignment& a) const {
    if (i.subset_of(head_arity(), tail_arity())) {
      return restrict_tail_only(i)(a);
    } else {
      return restrict_head_tail(i)(a);
    }
  }

  MomentGaussian<T> restrict_head(unsigned n, const Assignment& a) const {
    return restrict_head_tail(0, n)(a);
  }

  MomentGaussian<T> restrict_tail(unsigned n, const Assignment& a) const {
    if (n <= tail_arity()) {
      return restrict_tail_only(n)(a);
    } else {
      return restrict_head_tail(arity() - n, n)(a);
    }
  }

  Val<T> entropy() const {
    assert(is_marginal());
    CholeskyType chol(cov);
    return (size() * (std::log(two_pi<T>()) + T(1)) + logdet(chol)) / 2;
  }

  T kl_divergence(const MomentGaussian& other) const {
    const Impl& p = *this;
    const Impl& q = other.impl();
    assert(p.is_marginal() && q.is_marginal());
    assert(p.head_size() == q.head_size());
    unsigned m = p.head_size();
    CholeskyType chol_p(p.cov);
    CholeskyTpye chol_q(q.cov);
    auto identity = MatrixType::Identity(m, m);
    T trace = (p.cov.array() * chol_q.solve(identity).array()).sum();
    T means = (p.mean - q.mean).transpose() * chol_q.solve(p.mean - q.mean);
    T logdets = -logdet(chol_p) + logdet(chol_q);
    return (trace + means + logdets - m) / 2;
  }

  T max_diff(const MomentGaussian<T>& other) const {
    const Impl& x = other.impl();
    T diff_mean = (mean - x.mean).array().abs().maxCoeff();
    T diff_cov  = (cov - x.cov).array().abs().maxCoeff();
    T diff_coef = (coef - x.coef).array().abs().maxCoeff();
    return std::max({diff_mean, diff_cov, diff_coef});
  }

  MultivariateNormalDistribution<T> distribution() const {
    return {mean, cov, coef};
  }

  template <typename MAT>
  struct RestrictTailOnly {
    const VectorType& mean;
    const MatrixType& cov;
    MAT coef_x, coef_y; // retained tail x, restricted tail y
    T lm;

    MomentGaussian<T> operator()(const Assignment& a) const {
      r = std::make_unique<Impl>();
      r->mean = mean + coef_y * values;
      r->coef = std::move(coef_x);
      r->cov = cov;
      r->lm = lm;
      return r;
    }
  };

  template <typename MAT, typename VEC>
  struct RestrictHeadTail {
    // retained: x, restricted head: y, restricted tail: t
    VEC mean_x, mean_y;
    MAT cov_yy, cov_yx, cov_xx;
    MAT coef_y, coef_x;
    T lm;

    MomentGaussian<T> operator()(const Assignment& a) {
      auto values = a.vector<T>(0);
      auto vals_y = values.head(mean_y.size());
      auto vals_tail = values.tail(...);
      // compute sol_yx = cov_yy^{-1} cov_yx using Cholesky decomposition
      CholeskyType chol_yy;
      chol_yy.compute(ws.cov_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian restrict: Cholesky decomposition failed"
        );
      }
      if (!y.empty()) {
        ws.chol_yy.solveInPlace(ws.sol_yx);
      }

      // compute the residual over y (observation vec_y - the prediction)
      VectorType res_y = vals_y - mean_y;
      if (!is_marginal()) {
        res_y.noalias() -= coef_y * vals_tail;
      }

      // compute the output
      auto r = std::make_unique<Impl>();
      r->mean = mean_x + sol_yx.transpose() * res_y;
      if (!is_marginal()) {
        r->mean.noalias() += coef_x * vals_tail;
      }
      r->cov = cov_xx - cov_xy * sol_yx;
      r->lm = lm -
        (mean_y.size() * std::log(two_pi<T>()) + logdet(chol_yy) +
         res_y.dot(chol_yy.solve(res_y))) / T(2);
    }
  };

    /**
     * Multiplies two moment_gaussians when (a range of) the head of one operand
     * matches (a range of) the tail of the other operand. The ordering of the
     * operands is specified via the forward flag.
     */
    template <typename HeadIt, typename TailIt>
    friend void multiply_head_tail(const moment_gaussian_param& f,
                                   const moment_gaussian_param& g,
                                   index_range<HeadIt> f_head,
                                   index_range<TailIt> g_tail,
                                   bool forward,
                                   moment_gaussian_param& r) {
      assert(f_head.size() == g_tail.size());

      // compute the positions of the head and tail of f and g in the output
      std::size_t n = f_head.size();
      span x(forward ? 0 : g.head_size(), f.head_size());     // head of f
      span y(forward ? f.head_size() : 0, g.head_size());     // head of g
      span v(forward ? 0 : g.tail_size() - n, f.tail_size()); // tail of f
      span w(forward ? f.tail_size() : 0, g.tail_size() - n); // tail of g

      // compute the result
      r.resize(f.head_size() + g.head_size(),
               f.tail_size() + g.tail_size() - n);
      dense_matrix<RealType> coef = subcols(g.coef, g_tail); // used frequently
      subvec(r.mean, x) = f.mean;
      subvec(r.mean, y).noalias() = g.mean + coef * subvec(f.mean, f_head);
      submat(r.cov, x, x) = f.cov;
      submat(r.cov, x, y).noalias() = subcols(f.cov, f_head) * coef.transpose();
      submat(r.cov, y, x) = submat(r.cov, x, y).transpose();
      submat(r.cov, y, y).noalias() =
        g.cov + coef * submat(f.cov, f_head, f_head) * coef.transpose();
      submat(r.coef, x, v) = f.coef;
      submat(r.coef, x, w) = dense_matrix<RealType>::Zero(x.size(), w.size());
      submat(r.coef, y, v).noalias() = coef * subrows(f.coef, f_head);
      submat(r.coef, y, w) = subcols(g.coef, complement(g_tail, g.tail_size()));
      r.lm = f.lm + g.lm;
    }

    /**
     * Multiplies two moment_gugaussians when (a range of) the tail of the left
     * operand matches (a range of) the tail of the right operand.
     */
    template <typename TailIt1, typename TailIt2>
    friend void multiply_tails(const moment_gaussian_param& f,
                               const moment_gaussian_param& g,
                               index_range<TailIt1> f_tail,
                               index_range<TailIt2> g_tail,
                               moment_gaussian_param& r) {
      assert(f_tail.size() == g_tail.size());
      std::size_t n = f_tail.size();
      r.zero(f.head_size() + g.head_size(),
             f.tail_size() + g.tail_size() - n);
      span x(0, f.head_size());
      span y(f.head_size(), g.head_size());
      span v(0, f.tail_size());
      span w(f.tail_size(), g.tail_size() - n);
      subvec(r.mean, x) = f.mean;
      subvec(r.mean, y) = g.mean;
      submat(r.cov, x, x) = f.cov;
      submat(r.cov, y, y) = g.cov;
      submat(r.coef, x, v) = f.coef;
      submat(r.coef, y, f_tail) = subcols(g.coef, g_tail);
      submat(r.coef, y, w) = subcols(g.coef, complement(g_tail, g.tail_size()));
      r.lm = f.lm + g.lm;
    }


  RestrictTailOnly<MatrixType> restrict_tail_only(const Dims& i) const {
    return {mean, cov, cols(coef, j), cols(coef, i), lm};
  }

  RestrictTailOnly<BlockType> restrict_tail_only(unsigned n) const {
   return {mean, cov, coef.lefCols(tail_size() - n, n), coef.rightCols(n), lm};
  }

  RestrictHeadTail<MatrixType, VectorType> restrict_head_tail(const Dims& i) const {
    Dims j = exclude(i);
    return {
      sub(mean, i),
      sub(mean, y),
      sub(cov, y, y),
      sub(cov, y, x),
      sub(cov, x, x),
      rows(coef, y),
      rows(coef, x),
      lm
    };
  }

  RestrictHeadTail<BlockType, SegmentType> restrict_head_tail(unsigend start, unsigned n) {

  }
};





    auto values = a.vector<T>(0);
    size_t m = tail_arity();
    if (n < m) {
      auto r = std::make_unique<Impl>();
      return r;
    } else {
      return restrict_tail(values);
    }

  MomentGaussian<T> restrict_tail
      r->mean = mean + coef * values.tail(m);
      r->coef = MatrixType(mean.size(), 0);
      r->cov = cov;
      r->lm = lm;



};



  // Conditioning
  //--------------------------------------------------------------------------

  /**
   * If this expression represents a marginal distribution, this function
   * returns a moment_gaussian expression representing the conditional
   * distribution with n tail (front) dimensions.
   *
   * \throw numerical_error
   *        if the covariance matrix over the tail arguments is singular.
   */
  auto conditional(std::size_t nhead) const {
    assert(is_marginal() && nhead < derived().head_arity());
    using workspace_type = typename param_type::conditional_workspace;
    return make_moment_gaussian_function<workspace_type>(
      [nhead](const Derived& f, auto& ws, param_type& result) {
        f.param().conditinal(nhead, ws, result);
      }, nhead, derived().head_arity() - nhead, derived());
  }



  // Ordering
  //--------------------------------------------------------------------------

  /**
   * Returns a moment_gaussian expression with the head dimensions reordered
   * according to the given index.
   */
  auto reorder(const uint_vector& head) const {
    assert(head.size() == derived().head_arity());
    return make_moment_gaussian_function<void>(
      [&head](const Derived& f, param_type& result) {
        f.param().reorder(iref(head), all(f.tail_arity()), result);
      }, head.size(), derived().tail_arity(), derived());
  }

  /**
   * Returns a moment_gaussian expression with the head and tail dimensions
   * reordered according to the given indices.
   */
  auto reorder(const uint_vector& head, const uint_vector& tail) const {
    assert(head.size() == derived().head_arity() &&
            tail.size() == derived().tail_arity());
    return make_moment_gaussian_function<void>(
      [&head, &tail](const Derived& f, param_type& result) {
        f.param().reorder(iref(head), iref(tail), result);
      }, head.size(), tail.size(), derived());
  }


  /// Assigns a constant to this factor.
  moment_gaussian& operator=(logarithmic<RealType> x) {
    reset(0);
    param_.lm = log(x);
    return *this;
  }

  /// Serializes the factor to an archive.
  void save(oarchive& ar) const {
    ar << param_;
  }

  /// Deserializes the factor from an archive.
  void load(iarchive& ar) {
    ar >> param_;
  }


  /// Returns the dimensionality of the parameters for the given domain.
  template <typename Arg>
  static std::size_t shape(const domain<Arg>& dom) {
    return dom.num_dimensions();
  }

}