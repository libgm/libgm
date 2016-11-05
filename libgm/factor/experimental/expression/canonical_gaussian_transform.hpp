#ifndef LIBGM_CANONICAL_GAUSSIAN_TRANSFORM_HPP
#define LIBGM_CANONICAL_GAUSSIAN_TRANSFORM_HPP

#include <libgm/math/param/canonical_gaussian_param.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_base.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/tuple.hpp>

#include <type_traits>

namespace libgm { namespace experimental {

  /**
   * A class that represents an element-wise transform of one or more
   * canonical_gaussian expressions. The expressions must have the same arity.
   *
   * \tparam VectorOp
   *         A unary operation accepting a dense vector or matrix (the natural
   *         parameters) and returning an Eigen expression.
   * \tparam ScalarOp
   *         A unary operation accepting a real type (the log-multiplier) and
   *         returning a real type.
   * \tparam F
   *         A non-empty pack of canonical_gaussian expressions with identical
   *         real_type.
   */
  template <typename VectorOp, typename ScalarOp, typename... F>
  class canonical_gaussian_transform
    : public canonical_gaussian_base<
        std::result_of_t<ScalarOp(typename F::real_type...)>,
        canonical_gaussian_transform<VectorOp, ScalarOp, F...> > {
  public:
    // shortcuts
    using real_type  = std::result_of_t<ScalarOp(typename F::real_type...)>;
    using param_type = canonical_gaussian_param<real_type>;

    canonical_gaussian_transform(VectorOp vector_op,
                                 ScalarOp scalar_op,
                                 const std::tuple<const F&...>& data)
      : vector_op_(vector_op), scalar_op_(scalar_op), data_(data) { }

    std::size_t arity() const {
      return std::get<0>(data_).arity();
    }

    bool alias(const param_type& param) const {
      return false; // transform is always safe
    }

    void eval_to(param_type& result) const {
      auto param = tuple_transform(member_param(), data_);
      result.assign(eta(param), lambda(param), lm(param));
    }

    template <typename UpdateOp>
    void transform_inplace(UpdateOp update_op, param_type& result) const {
      auto param = tuple_transform(member_param(), data_);
      result.update(update_op, eta(param), lamda(param), lm(param));
    }

    template <typename UpdateOp, typename It>
    void join_inplace(UpdateOp update_op, index_range<It> join_dims,
                      param_type& result) const {
      auto param = tuple_transform(member_param(), data_);
      result.update(update_op, join_dims, eta(param), lamda(param), lm(param));
    }

    template <typename UnaryVectorOp, typename UnaryScalarOp>
    canonical_gaussian_transform<compose_t<UnaryVectorOp, VectorOp>,
                                 compose_t<UnaryScalarOp, ScalarOp>,
                                 F...>
    transform(UnaryVectorOp vector_op, UnaryScalarOp scalar_op) const {
      return { compose(vector_op, vector_op_),
               compose(scalar_op, scalar_op_),
               data_ };
    }

    friend VectorOp transform_vector_op(const canonical_gaussian_transform& f) {
      return f.vector_op_;
    }

    friend ScalarOp transform_scalar_op(const canonical_gaussian_transform& f) {
      return f.scalar_op_;
    }

    std::tuple<const F&...> transform_data(const canonical_gaussian_transform& f) {
      return f.data_;
    }

  private:
    template <typename... Param>
    decltype(auto) eta(const std::tuple<Param...>& param) const {
      return tuple_apply(
        vector_op_,
        tuple_transform(std::mem_fn(&param_type::eta), param)
      );
    }

    template <typename... Param>
    decltype(auto) lambda(const std::tuple<Param...>& param) const {
      return tuple_apply(
        vector_op_,
        tuple_transform(std::mem_fn(&param_type::lambda), param)
      );
    }

    template <typename... Param>
    real_type lm(const std::tuple<Param...>& param) const {
      return tuple_apply(
        scalar_op_,
        tuple_transform(std::mem_fn(&param_type::lm), param)
      );
    }

    VectorOp vector_op_; // operator transforming the information vector/matrix
    ScalarOp scalar_op_; // operator transforming the log multiplier
    std::tuple<add_const_reference_if_factor_t<F>...> data_;

  }; // class canonical_gaussian_transform

  /**
   * Constructs a canonical_gaussian_transform object, deducing its type.
   *
   * \relates canonical_gaussian_transform
   */
  template <typename VectorOp, typename ScalarOp, typename... F>
  inline canonical_gaussian_transform<VectorOp, ScalarOp, F...>
  make_canonical_gaussian_transform(VectorOp vector_op,
                                    ScalarOp scalar_op,
                                    const std::tuple<const F&...>& data) {
    return { vector_op, scalar_op, data };
  }

  /**
   * The default vector transform associated with a canonical_gaussian.
   * \relates canonical_gaussian_base
   */
  template <typename RealType, typename Derived>
  inline identity
  transform_vector_op(const canonical_gaussian_base<RealType, Derived>& f) {
    return identity();
  }

  /**
   * The default scalar transform associated with a canonical_gaussian.
   * \relates canonical_gaussian_base
   */
  template <typename RealType, typename Derived>
  inline identity
  transform_scalar_op(const canonical_gaussian_base<RealType, Derived>& f) {
    return identity();
  }

  /**
   * The default data associated with a canonical_gaussian.
   * \relates canonical_gaussian_base
   */
  template <typename RealType, typename Derived>
  inline std::tuple<const Derived&>
  transform_data(const canonical_gaussian_base<RealType, Derived>& f) {
    return std::tie(f.derived());
  }

  /**
   * Transforms two canonical_gaussian expressions with identical RealType.
   * \relates canonical_gaussian_base
   */
  template <typename BinaryOp, typename RealType, typename F, typename G>
  inline auto
  transform(BinaryOp binary_op,
            const canonical_gaussian_base<RealType, F>& f,
            const canonical_gaussian_base<RealType, G>& g) {
    const F& fd = f.derived();
    const G& gd = g.derived();
    auto fdata = transform_data(fd);
    auto gdata = transform_data(gd);
    constexpr std::size_t m = std::tuple_size<decltype(fdata)>::value;
    constexpr std::size_t n = std::tuple_size<decltype(gdata)>::value;
    return make_canonical_gaussian_transform(
      compose<m,n>(binary_op, transform_vector_op(fd), transform_vector_op(gd)),
      compose<m,n>(binary_op, transform_scalar_op(fd), transform_scalar_op(gd)),
      std::tuple_cat(fdata, gdata)
    );
  }

} } // namespace libgm::experimental

#endif
