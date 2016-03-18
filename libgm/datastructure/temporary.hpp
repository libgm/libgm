#ifndef LIBGM_TEMPORARY_HPP
#define LIBGM_TEMPORARY_HPP

namespace libgm {

  template <typename T, bool Capture = true>
  struct temporary {
    const T& capture(T&& value) {
      t = std::move(value);
      return t;
    }
    T t;
  };

  template <typename T>
  struct temporary<T, false> {
    template <typename Expr>
    decltype(auto) capture(Expr&& expr) {
      return std::forward<Expr>(expr);
    }
  };

} // namespace libgm

#endif
