#ifndef LIBGM_COMMUTATIVE_SEMIRING_HPP
#define LIBGM_COMMUTATIVE_SEMIRING_HPP

#include <libgm/argument/annotated.hpp>
#include <libgm/datastructure/uint_vector.hpp>

namespace libgm {

  /**
   * A base class that represents one of pre-defined commutative semirings
   * on factor types.
   */
  template <typename Arg, typename F>
  class commutative_semiring {
  public:
    virtual ~commutative_semiring() { }

    /**
     * The "dot" operation that takes two factors along with their domains
     * and returns a new factor with a domain that's a combination of the two.
     */
    annotated<Arg, F> combine(const annotated<Arg, F>& f,
                              const annotated<Arg, F>& g) const {
      annotated<Arg, F> result;
      result.domain = f.domain + g.domain;
      result.factor = combine(f.factor, result.domain.index(f.domain),
                              g.factor, result.domain.index(g.domain));
      return result;
    }

    /**
     * An in-place version of the dot operation, updating the left factor
     * with the right one. This operation may extend the domain of the
     * resulting factor.
     */
    void combine_in(annotated<Arg, F>& f, const annotated<Arg, F>& g) const {
      if (superset(f.domain, g.domain)) {
        combine_in(f.factor, f.domain.index(g.domain), g.factor);
      } else {
        f = combine(f, g);
      }
    }

    /**
     * Combines all annotated factors in a range.
     * \tparam Range a range of annotated<Arg, F> objects
     */
    template <typename Range>
    annotated<Arg, F> combine_all(const Range& factors) const {
      annotated<Arg, F> result{{}, combine_init()};
      for (const annotated<Arg, F>& f : factors) {
        combine_in(result, f);
      }
      return result;
    }

    /**
     * The "cross" operation (e.g., marginal in the sum-product algorithm).
     */
    annotated<Arg, F> collapse(const annotated<Arg, F>& f,
                               const domain<Arg>& retain) const {
      return { retain, collapse(f.factor, f.domain.index(retain)) };
    }

    /**
     * The "cross" operation that eliminates a single argument from a factor.
     */
    annotated<Arg, F> eliminate(const annotated<Arg, F>& f, Arg v) const {
      std::size_t pos = f.domain.index_of(v);
      return { f.domain - v, eliminate(f.factor, pos, argument_arity(v)) };
    }

    /**
     * The "cross" oeration that eliminates a subset of arguments from a factor.
     */
    annotated<Arg, F> eliminate(const annotated<Arg, F>&
                                const domain<Arg>& args) const {
      return { f.domain - v, eliminate(f.factor, f.domain.index(args)) };
    }

    /**
     * Combines two factors along the given indices.
     */
    virtual F combine(const F& f, const uint_vector& fdims,
                      const F& g, const uint_vector& gdims) const = 0;

    /**
     * Combines a factor with another one in place along give indices.
     */
    virtual F combine_in(F& f, const uint_vector& dims, const F& g) const = 0;

    /**
     * The initial value for the dot operation (e.g., 1 in sum-product).
     */
    virtual F combine_init() const = 0;

    /**
     * Collapses a factor, retaining a set of indices.
     */
    virtual F collapse(const F& f, const uint_vector& retain) const = 0;

    /**
     * Eliminates a range of dimensions from a factor.
     */
    virtual F eliminate(const F& f, std::size_t pos, std::size_t n) const = 0;

    /**
     * Eliminates a subset of dimension from a factor.
     */
    virtual F eliminate(const F& f, const uint_vector& dims) const = 0;

  }; // class commutative_semiring

  /**
   * An object representing the sum product commutative semiring
   * \f$([0, \infty), +, \times, 0, 1)\f$.
   * \relates commutative_semiring
   */
  template <typename Arg, typename F>
  class sum_product : public commutative_semiring<Arg, F> {
  public:
    using base = commutative_semiring<Arg, F>;

    using base::combine;
    using base::combine_in;
    using base::collapse;
    using base::eliminate;

    F combine(const F& f, const uint_vector& fdims,
              const F& g, const uint_vector& gdims) const override {
      return f.dims(fdims) * g.dims(gdims);
    }

    void combine_in(F& f, const uint_vector& fdims, const F& g) const override {
      f.dims(fdims) *= g;
    }

    F combine_init() const override {
      return F(typename F::result_type(1));
    }

    F collapse(const F& f, const uint_vector& retain) const override {
      return f.marginal(retain);
    }

    F eliminate(const F& f, std::size_t pos, std::size_t n) const override {
      return f.marginal(pos, n);
    }

    F eliminate(const F& f, const uint_vector& dims) const override {
      return f.dims(dims).sum();
    }
  };

  /**
   * An object representing the max product commutative semiring
   * \f$([0, \infty), \max, \times, 0, 1)\f$.
   * \relates commutative_semiring
   */
  template <typename Arg, typename F>
  class max_product : public commutative_semiring<Arg, F> {
  public:
    using base = commutative_semiring<Arg, F>;

    using base::combine;
    using base::combine_in;
    using base::collapse;
    using base::eliminate;

    F combine(const F& f, const uint_vector& fdims,
              const F& g, const uint_vector& gdims) const override {
      return f.dims(fdims) * g.dims(gdims);
    }

    void combine_in(F& f, const uint_vector& fdims, const F& g) const override {
      f.dims(fdims) *= g;
    }

    F combine_init() const override {
      return F(typename F::result_type(1));
    }

    F collapse(const F& f, const uint_vector& retain) const override {
      return f.maximum(retain);
    }

    F eliminate(const F& f, std::size_t pos, std::size_t n) const override {
      return f.maximum(pos, n);
    }

    F eliminate(const F& f, const uint_vector& dims) const override {
      return f.dims(dims).max();
    }
  };

  /**
   * An object representing the min-sum commutative semiring
   * \f$((-\infty, \infty], \min, +, \infty, 0)\f$.
   * \relates commutative_semiring
   */
  template <typename Arg, typename F>
  class min_sum : public commutative_semiring<Arg, F> {
  public:
    using base = commutative_semiring<Arg, F>;

    using base::combine;
    using base::combine_in;
    using base::collapse;
    using base::eliminate;

    F combine(const F& f, const uint_vector& fdims,
              const F& g, const uint_vector& gdims) const override {
      return f.dims(fdims) + g.dims(gdims);
    }

    void combine_in(F& f, const uint_vector& fdims, const F& g) const override {
      f.dims(fdims) += g;
    }

    F combine_init() const override {
      return F(typename F::result_type(0));
    }

    F collapse(const F& f, const uint_vector& retain) const override {
      return f.minimum(retain);
    }

    F eliminate(const F& f, std::size_t pos, std::size_t n) const override {
      return f.minimum(pos, n);
    }

    F eliminate(const F& f, const uint_vector& dims) const override {
      return f.dims(dims).min();
    }
  };

  /**
   * An object representing the max-sum commutative semiring
   * \f$([-\infty, \infty), \max, +, -\infty, 0)\f$.
   * \relates commutative_semiring
   */
  template <typename Arg, typename F>
  class max_sum : public commutative_semiring<Arg, F> {
  public:
    using base = commutative_semiring<Arg, F>;

    using base::combine;
    using base::combine_in;
    using base::collapse;
    using base::eliminate;

    F combine(const F& f, const uint_vector& fdims,
              const F& g, const uint_vector& gdims) const override {
      return f.dims(fdims) + g.dims(gdims);
    }

    void combine_in(F& f, const uint_vector& fdims, const F& g) const override {
      f.dims(fdims) += g;
    }

    F combine_init() const override {
      return F(typename F::result_type(0));
    }

    F collapse(const F& f, const uint_vector& retain) const override {
      return f.maximum(retain);
    }

    F eliminate(const F& f, std::size_t pos, std::size_t n) const override {
      return f.maximum(pos, n);
    }

    F eliminate(const F& f, const uint_vector& dims) const override {
      return f.dims(dims).max();
    }
  };

} // namespace libgm

#endif
