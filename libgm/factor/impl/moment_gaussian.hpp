namespace libgm {

template <typename T>
struct MomentGaussian<T>::Impl {
  /// The type of the LLT Cholesky decomposition object.
  using CholeskyType = Eigen::LLT<DenseMatrix<T>>;
  using VectorType = DenseVector<T>;
  using MatrixType = DenseMatrix<T>;

  /// The shape of the head arguments.
  Shape head_shape;

  /// The shape of the tail arguments
  Shape tail_shape;

  /// The conditional mean.
  DenseVector<T> mean;

  /// The covariance matrix.
  DenseMatrix<T> cov;

  /// The coefficient matrix.
  DenseMatrix<T> coef;

  /// The log-multiplier.
  T lm;

  // Accessors
  //--------------------------------------------------------------------------

  bool is_marginal() const {
    return tail_arity() == 0;
  }

  // Object operations
  //--------------------------------------------------------------------------

  bool equals(const Object& other) const {
    const Impl& x = other.impl();
    return head_shape == x.head_shape && tail_shape == x.tail_shape &&
      lm == x.lm && mean == x.mean && cov == x.cov && coef == x.coef;
  }

  void print(std::ostream& out) const override {
   out << head_shape << " " << tail_shape << std::endl
       << mean << std::endl
       << cov << std::endl
       << coef << std::endl
       << lm;
  }

  void save(oarchive& ar) const override {
    ar << head_shape << tail_shape << mean << cov << coef << lm;
  }

  void load(iarchive& ar) override {
    ar >> head_shape >> tail_shape >> mean >> cov >> coef >> lm;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  ImplPtr multiply(const Exp<T>& x) const {
    return std::make_unique<Impl>(head_shape, tail_shape, mean, cov, coef, lm + log(x));
  }

  ImplPtr divide(const Exp<T>& x) const {
    return std::make_unqiue<Impl>(head_shape, tail_shape, mean, cov, coef, lm - log(x));
  }

  void multiply_in(const Exp<T>& x) {
    lm += log(x);
  }

  void divide_in(const Exp<T>& x) {
    lm -= log(x);
  }

  // Join operations
  //--------------------------------------------------------------------------

  ImplPtr multiply_head(const Object& other) const {
    assert(0);
  }

  ImplPtr multiply_tail(const Object& other) const {
    const Impl& x = impl(other);
    assert(tail_shape == x.head_shape + y.tail_shape);
    return {
      head_shape + x.head_shape,
      y.tail_shape,
      ...;
    }
  }





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


  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const {
    return Exp<T>(lm);
  }

  Exp<T> maximum(Values* values) const {
    if (values) *values = mean;
    CholeskyType chol;
    chol.solve(cov);
    check(chol, "maximum");
    return Exp<T>((-log_two_pi * head_size() - logdet(chol)) / T(2) + lm);
  }

  ImplPtr marginal_front(unsigned n) const {
    size_t nx = head_shape.prefix_sum(n);
    return std::make_unique<Impl>(
      head_shape.prefix(n),
      tail_shape,
      mean.head(nx),
      cov.topLeftCorner(nx, nx),
      coef.topRows(nx),
      lm
    );
  }

  ImplPtr marginal_back(unsigned n) const {
    size_t nx = head_shape.suffix_sum(n);
    return std::make_unique<Impl>(
      head_shape.suffix(n),
      tail_shape,
      mean.tail(nx),
      cov.bottomRightCorner(nx, nx),
      coef.bottomRows(nx),
      lm,
    );
  }

  ImplPtr marginal_dims(const Dims& i) const {
    Spans is = head_shape.spans(i);
    return std::make_unique<Impl>(
      head_shape.subseq(i),
      tail_shape,
      sub(mean, is),
      sub(cov, is, is),
      sub(coef, is, Span(0, coef.cols())),
      lm,
    );
  }

  ImplPtr maximum_front(unsigned n) const {
    return marginal_to_maximum(marginal_front(n));
  }

  ImplPtr maximum_back(unsigned n) const {
    return marginal_to_maximum(marginal_back(n));
  }

  ImplPtr maximum_dims(const Dims& dims) const {
    return marginal_to_maximum(marginal_dims(dims));
  }

  ImplPtr marginal_to_maximum(ImplPtr impl) const {
    Impl& result = static_cast<Impl&>(*impl);
    result.lm += maximum().lv - result.maximum().lv;
    return std::move(impl);
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    lm = 0;
  }

  void normalize(unsigned nhead) {

  }

  // Restriction
  //--------------------------------------------------------------------------

  ImplPtr restrict_head(const Values& values) const {
    return restrict_head_tail(0, n)(a);
  }

  ImplPtr restrict_tail(const Values& values) const {
    if (n <= tail_arity()) {
      return restrict_tail_only(n)(a);
    } else {
      return restrict_head_tail(arity() - n, n)(a);
    }
  }

  ImplPtr restrict_dims(const Dims& dims, const Values& values) const {
    if (i.subset_of(head_arity(), tail_arity())) {
      return restrict_tail_only(i)(a);
    } else {
      return restrict_head_tail(i)(a);
    }
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------


  T entropy() const {
    assert(is_marginal());
    CholeskyType chol;
    chol.compute(cov);
    return (size() * (log_two_pi + T(1)) + logdet(chol)) / T(2);
  }

  T kl_divergence(const Object& other) const {
    const Impl& p = *this;
    const Impl& q = impl(other);
    assert(p.is_marginal() && q.is_marginal());
    assert(p.head_shape == q.head_shape);
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

  // Utility classes
  //--------------------------------------------------------------------------

  /// Implements the restrict operation for restricting tail only.
  template <typename MAT>
  struct RestrictTailOnly {
    const VectorType& mean;
    const MatrixType& cov;
    MAT coef_x, coef_y; // retained tail x, restricted tail y
    T lm;

    ImplPtr operator()(const Values& values) const {
      auto r = std::make_unique<Impl>();
      r->mean = mean + coef_y * values;
      r->coef = std::move(coef_x);
      r->cov = cov;
      r->lm = lm;
      return r;
    }
  };

  /// Implements the restrict operation for restricting head and (optionally) tail.
  template <typename MAT, typename VEC>
  struct RestrictHeadTail {
    // retained: x, restricted head: y, restricted tail: t
    VEC mean_x, mean_y;
    MAT cov_yy, cov_yx, cov_xx;
    MAT coef_y, coef_x;
    T lm;

    ImplPtr operator()(const Assignment& a) {
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

  ImplPtr restrict_tail
      r->mean = mean + coef * values.tail(m);
      r->coef = MatrixType(mean.size(), 0);
      r->cov = cov;
      r->lm = lm;



};





  /// Assigns a constant to this factor.
  moment_gaussian& operator=(logarithmic<RealType> x) {
    reset(0);
    param_.lm = log(x);
    return *this;
  }

  /// Serializes the factor to an archive.
  void save(oarchive& ar) const override {
    ar << param_;
  }

  /// Deserializes the factor from an archive.
  void load(iarchive& ar) override {
    ar >> param_;
  }


  /// Returns the dimensionality of the parameters for the given domain.
  template <typename Arg>
  static std::size_t shape(const domain<Arg>& dom) {
    return dom.num_dimensions();
  }

void canonical() {
    CholeskyType chol(mg.covariance());
    if (chol.info() != Eigen::Success) {
      throw numerical_error(
        "CanonicalGaussian: Cannot invert the covariance matrix. "
        "Are you passing in a non-singular moment Gaussian distribution?"
      );
    }
    MatrixType sol_xy = chol.solve(mg.coefficients());

    size_t m = mg.head_size();
    size_t n = mg.tail_size();
    resize(m + n);

    eta.segment(0, m) = chol.solve(mg.mean());
    eta.segment(m, n).noalias() = -sol_xy.transpose() * mg.mean();

    lambda.block(0, 0, m, m) = chol.solve(MatrixType::Identity(m, m));
    lambda.block(0, m, m, n) = -sol_xy;
    lambda.block(m, 0, n, m) = -sol_xy.transpose();
    lambda.block(m, m, n, n).noalias() = mg.coef.transpose() * sol_xy;

    lm = mg.lm - (m * std::log(two_pi<T>()) + logdet(chol)
                  + eta.segment(0, m).dot(mg.mean())) / T(2);
  }
