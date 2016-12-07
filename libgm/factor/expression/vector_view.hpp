#ifndef LIBGM_VECTOR_VIEW_HPP
#define LIBGM_VECTOR_VIEW_HPP

#include <libgm/factor/expression/vector_base.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/functional/tuple.hpp>

namespace libgm { namespace experimental {

  /**
   * A matrix expression that returns a view of other matrix / vector
   * expressions.
   *
   * \tparam Space
   *         A tag denoting the space of the matrix (prob_tag or log_tag).
   * \tparam Op
   *         A functor that given F, returns the Eigen matrix expression.
   * \tparam F
   *         A (possibly empty) parameter pack of matrix or vector expressions.
   */
  template <typename Space, typename Op, typename... F>
  class vector_view
    : public vector_base<
        Space,
        typename std::result_of_t<Op(F...)>::Scalar,
        vector_view<Space, Op, F...> > {
  public:
    // Shortcuts
    using real_type = typename std::result_of_t<Op(F...)>::Scalar;

    vector_view(Op op, const F&... f)
      : op_(op), data_(f...) { }

    bool alias(const dense_vector<real_type>& param) const {
      return tuple_any(make_member_alias(param), data_);
    }

    bool alias(const dense_matrix<real_type>& param) const {
      return tuple_any(make_member_alias(param), data_);
    }

    void eval_to(dense_vector<real_type>& result) const {
      result = param();
    }

    auto param() const {
      return tuple_apply(op_, data_);
    }

  private:
    Op op_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;

  }; // class vector_view

  /**
   * Creates a vector_view object, automatically deducing its type.
   * \relates vector_view
   */
  template <typename Space, typename Op, typename... F>
  inline vector_view<Space, Op, F...>
  make_vector_view(Op op, const F&... f) {
    return { op, f... };
  }

  /**
   * Returns a vector_view expression object representing a view of a raw
   * buffer with known number of rows.
   * \relates vector_view
   */
  template <typename Space, typename RealType>
  inline auto
  vector_raw(const RealType* ptr, std::size_t rows) {
    return make_vector_view<Space>(
      [ptr, rows]() {
        return Eigen::Map<const dense_vector<RealType> >(ptr, rows);
      });
  }

} } // namespace libgm::experimental

#endif
