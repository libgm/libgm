#ifndef LIBGM_MOMENT_GAUSSIAN_FUNCTION_HPP
#define LIBGM_MOMENT_GAUSSIAN_FUNCTION_HPP

#include <libgm/datastructure/compressed.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>
#include <libgm/factor/expression/moment_gaussian_base.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/traits/nth_type.hpp>

namespace libgm { namespace experimental {

  /**
   * A simple moment_gaussian expression that is a function of one or more
   * underlying table expressions.
   *
   * \tparam Workspace
   *         An optional datastructure that is stored in the function and
   *         is mutated by the evaluation function. May be void.
   * \tparam Alias
   *         Determines whether alias checking is performed.
   * \tparam EvalOp
   *         A function accepting all F, workspace (if not void), and
   *         moment_gaussina_param reference.
   * \tparam F
   *         A non-empty pack of moment_gaussian expressions with identical
   *         real_type.
   */
  template <typename Workspace, bool Alias, typename EvalOp, typename... F>
  class moment_gaussian_function
    : public moment_gaussian_base<
        typename nth_type_t<0, F...>::real_type,
        moment_gaussian_function<Workspace, Alias, EvalOp, F...> >,
      compressed_workspace<Workspace> {

  public:
    // Shortcuts
    using real_type  = typename nth_type_t<0, F...>::real_type;
    using param_type = moment_gaussian_param<real_type>;

    moment_gaussian_function(EvalOp eval_op,
                             std::size_t nhead, std::size_t ntail,
                             const F&... f)
      : eval_op_(eval_op), nhead_(nhead), ntail_(ntail), data_(f...) { }

    std::size_t head_arity() const {
      return nhead_;
    }

    std::size_t tail_arity() const {
      return ntail_;
    }

    //! Implementation of alias() that checks for aliasing.
    template <bool B = Alias>
    std::enable_if_t<B, bool> alias(const param_type& param) const {
      return tuple_any(member_alias<param_type>(param), data_);
    }

    //! Implementation fo alias() that does not check for aliasing.
    template <bool B = Alias>
    std::enable_if_t<!B, bool> alias(const param_type& param) const {
      return false;
    }

    //! Implementation of eval_to with workspace.
    template <bool B = std::is_void<Workspace>::value>
    std::enable_if_t<!B, void> eval_to(param_type& result) const {
      tuple_apply(eval_op_, tuple_cat(data_, std::tie(this->ws_, result)));
    }

    //! Implementation of eval_to without workspace.
    template <bool B = std::is_void<Workspace>::value>
    std::enable_if_t<B, void> eval_to(param_type& result) const {
      tuple_apply(eval_op_, tuple_cat(data_, std::tie(result)));
    }

  private:
    EvalOp eval_op_;
    std::size_t nhead_;
    std::size_t ntail_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;
  }; // class moment_function

  /**
   * Creates a moment_gaussian function that checks for aliasing, automatically
   * deducing its type.
   *
   * \relates moment_gaussian_function
   */
  template <typename Workspace, typename EvalOp, typename... F>
  inline moment_gaussian_function<Workspace, true, EvalOp, F...>
  make_moment_gaussian_function(EvalOp eval_op,
                                std::size_t nhead, std::size_t ntail,
                                const F&... f) {
    return { eval_op, nhead, ntail, f... };
  }

  /**
   * Creates a moment_gaussian function that does not check for aliasing,
   * automatically deducing its type.
   *
   * \relates moent_gaussian_function
   */
  template <typename Workspace, typename EvalOp, typename... F>
  inline moment_gaussian_function<Workspace, false, EvalOp, F...>
  make_moment_gaussian_function_noalias(EvalOp eval_op,
                                        std::size_t nhead, std::size_t ntail,
                                        const F&... f) {
    return { eval_op, nhead, ntail, f... };
  }

} } // namespace libgm::experimental

#endif
