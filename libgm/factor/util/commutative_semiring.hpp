#ifndef LIBGM_COMMUTATIVE_SEMIRING_HPP
#define LIBGM_COMMUTATIVE_SEMIRING_HPP

namespace libgm {

  /**
   * A base class that represents one of pre-defined commutative semirings
   * on factor types.
   */
  template <typename F>
  class commutative_semiring {
  public:
    typedef typename F::domain_type domain_type;

    //! destructor
    virtual ~commutative_semiring() { }

    //! the "cross" operation (e.g., marginal in the sum-product algorithm)
    virtual F collapse(const F& x, const domain_type& retain) const = 0;

    //! the "dot" operation (e.g., multiplication in the sum-product algorithm)
    virtual F combine(const F& x, const F& y) const = 0;

    //! an in-place version of the dot operation
    virtual void combine_in(F& x, const F& y) const = 0;

    //! the initial value for the dot operation (e.g., 1 in sum-product)
    virtual F combine_init() const = 0;

    //! eliminates a variable from the factor
    F collapse_out(const F& x, const domain_type& eliminate) const {
      return collapse(x, x.arguments() - eliminate);
    }
  };

  /**
   * An object representing the sum product commutative semiring
   * \f$([0, \infty), +, \times, 0, 1)\f$.
   * \relates commutative_semiring
   */
  template <typename F>
  class sum_product : public commutative_semiring<F> {
  public:
    typedef typename F::domain_type domain_type;

    F collapse(const F& x, const domain_type& retain) const override {
      return x.marginal(retain);
    }
    F combine(const F& x, const F& y) const override {
      return x * y;
    }
    void combine_in(F& x, const F& y) const override {
      x *= y;
    }
    F combine_init() const override {
      return F(typename F::result_type(1));
    }
  };

  /**
   * An object representing the max product commutative semiring
   * \f$([0, \infty), \max, \times, 0, 1)\f$.
   * \relates commutative_semiring
   */
  template <typename F>
  class max_product : public commutative_semiring<F> {
  public:
    typedef typename F::domain_type domain_type;

    F collapse(const F& x, const domain_type& retain) const override {
      return x.maximum(retain);
    }
    F combine(const F& x, const F& y) const override {
      return x * y;
    }
    void combine_in(F& x, const F& y) const override {
      x *= y;
    }
    F combine_init() const override {
      return F(typename F::result_type(1));
    }
  };

  /**
   * An object representing the min-sum commutative semiring
   * \f$((-\infty, \infty], \min, +, \infty, 0)\f$.
   * \relates commutative_semiring
   */
  template <typename F>
  class min_sum : public commutative_semiring<F> {
  public:
    typedef typename F::domain_type domain_type;

    F collapse(const F& x, const domain_type& retain) const override {
      return x.minimum(retain);
    }
    F combine(const F& x, const F& y) const override {
      return x + y;
    }
    void combine_in(F& x, const F& y) const override {
      x += y;
    }
    F combine_init() const override {
      return F(0);
    }
  };

  /**
   * An object representing the max-sum commutative semiring
   * \f$([-\infty, \infty), \max, +, -\infty, 0)\f$.
   * \relates commutative_semiring
   */
  template <typename F>
  class max_sum : public commutative_semiring<F> {
  public:
    typedef typename F::domain_type domain_type;

    F collapse(const F& x, const domain_type& retain) const override {
      return x.maximum(retain);
    }
    F combine(const F& x, const F& y) const override {
      return x + y;
    }
    void combine_in(F& x, const F& y) const override {
      x += y;
    }
    F combine_init() const override {
      return F(0);
    }
  };

} // namespace libgm

#endif
