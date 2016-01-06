#ifndef LIBGM_EXPRESSION_MACROS_HPP
#define LIBGM_EXPRESSION_MACROS_HPP

// Shortcuts for common expressions (used inside expression bases)
//==============================================================================

namespace libgm {
  void transform();
}

#define LIBGM_TRANSFORM_LEFT(function, factor, scalar, op)              \
  friend auto function(const factor##_base<Arg, RealType, Derived>& f,  \
                       scalar x) {                                      \
    return f.derived().transform(op);                                   \
  }                                                                     \
                                                                        \
  friend auto function(factor##_base<Arg, RealType, Derived>&& f,       \
                       scalar x) {                                      \
    return std::move(f).derived().transform(op);                        \
  }

#define LIBGM_TRANSFORM_RIGHT(function, factor, scalar, op)             \
  friend auto function(scalar x,                                        \
                       const factor##_base<Arg, RealType, Derived>& f) { \
    return f.derived().transform(op);                                   \
  }                                                                     \
                                                                        \
  friend auto function(scalar x,                                        \
                       factor##_base<Arg, RealType, Derived>&& f) {     \
    return std::move(f).derived().transform(op);                        \
  }

#define LIBGM_TRANSFORM_LEFT2(function, factor, scalar, op1, op2)       \
  friend auto function(const factor##_base<Arg, RealType, Derived>& f,  \
                       scalar x) {                                      \
    return f.derived().transform(op1, op2);                             \
  }                                                                     \
                                                                        \
  friend auto function(factor##_base<Arg, RealType, Derived>&& f,       \
                       scalar x) {                                      \
    return std::move(f).derived().transform(op1, op2);                  \
  }

#define LIBGM_TRANSFORM_RIGHT2(function, factor, scalar, op1, op2)      \
  friend auto function(scalar x,                                        \
                       const factor##_base<Arg, RealType, Derived>& f) { \
    return f.derived().transform(op1, op2);                             \
  }                                                                     \
                                                                        \
  friend auto function(scalar x,                                        \
                       factor##_base<Arg, RealType, Derived>&& f) {     \
    return std::move(f).derived().transform(op1, op2);                  \
  }

#define LIBGM_TRANSFORM(function, factor, op)                           \
  template <typename Other>                                             \
  friend auto function(const factor##_base<Arg, RealType, Derived>& f,  \
                       const factor##_base<Arg, RealType, Other>& g) {  \
    assert(f.derived().arguments() == g.derived().arguments());         \
    using libgm::transform;                                             \
    return transform(op, f.derived(), g.derived(),                      \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const factor##_base<Arg, RealType, Derived>& f,  \
                       factor##_base<Arg, RealType, Other>&& g) {       \
    assert(f.derived().arguments() == g.derived().arguments());         \
    using libgm::transform;                                             \
    return transform(op, f.derived(), std::move(g).derived(),           \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(factor##_base<Arg, RealType, Derived>&& f,       \
                       const factor##_base<Arg, RealType, Other>& g) {  \
    assert(f.derived().arguments() == g.derived().arguments());         \
    using libgm::transform;                                             \
    return transform(op, std::move(f).derived(), g.derived(),           \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(factor##_base<Arg, RealType, Derived>&& f,       \
                       factor##_base<Arg, RealType, Other>&& g) {       \
    assert(f.derived().arguments() == g.derived().arguments());         \
    using libgm::transform;                                             \
    return transform(op, std::move(f).derived(), std::move(g).derived(),\
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }

#define LIBGM_TRANSFORM_SCALAR(function, factor, scalar, op)            \
  template <typename Other>                                             \
  friend auto function(const factor##_base<Arg, RealType, Derived>& f,  \
                       const factor##_base<Arg, RealType, Other>& g,    \
                       scalar x) {                                      \
    assert(f.derived().arguments() == g.derived().arguments());         \
    using libgm::transform;                                             \
    return transform(op, f.derived(), g.derived(),                      \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const factor##_base<Arg, RealType, Derived>& f,  \
                       factor##_base<Arg, RealType, Other>&& g,         \
                       scalar x) {                                      \
    assert(f.derived().arguments() == g.derived().arguments());         \
    using libgm::transform;                                             \
    return transform(op, f.derived(), std::move(g).derived(),           \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(factor##_base<Arg, RealType, Derived>&& f,       \
                       const factor##_base<Arg, RealType, Other>& g,    \
                       scalar x) {                                      \
    assert(f.derived().arguments() == g.derived().arguments());         \
    using libgm::transform;                                             \
    return transform(op, std::move(f).derived(), g.derived(),           \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(factor##_base<Arg, RealType, Derived>&& f,       \
                       factor##_base<Arg, RealType, Other>&& g,         \
                       scalar x) {                                      \
    assert(f.derived().arguments() == g.derived().arguments());         \
    using libgm::transform;                                             \
    return transform(op, std::move(f).derived(), std::move(g).derived(),\
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }

#define LIBGM_JOIN2(function, factor1, factor2, op)                     \
  template <typename Other>                                             \
  friend auto function(const factor1##_base<Arg, RealType, Derived>& f, \
                       const factor2##_base<Arg, RealType, Other>& g) { \
    return join(op, f.derived(), g.derived(),                           \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const factor1##_base<Arg, RealType, Derived>& f, \
                       factor2##_base<Arg, RealType, Other>&& g) {      \
    return join(op, f.derived(), std::move(g).derived(),                \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(factor1##_base<Arg, RealType, Derived>&& f,      \
                       const factor2##_base<Arg, RealType, Other>& g) { \
    return join(op, std::move(f).derived(), g.derived(),                \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(factor1##_base<Arg, RealType, Derived>&& f,      \
                       factor2##_base<Arg, RealType, Other>&& g) {      \
    return join(op, std::move(f).derived(), std::move(g).derived(),     \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }

#define LIBGM_JOIN2R(function, factor1, factor2, op)                    \
  template <typename Other>                                             \
  friend auto function(const factor1##_base<Arg, RealType, Other>& f,   \
                       const factor2##_base<Arg, RealType, Derived>& g) { \
    return join(op, f.derived(), g.derived(),                           \
                static_cast<Other*>(0), static_cast<Derived*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const factor1##_base<Arg, RealType, Other>& f,   \
                       factor2##_base<Arg, RealType, Derived>&& g) {    \
    return join(op, f.derived(), std::move(g).derived(),                \
                static_cast<Other*>(0), static_cast<Derived*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(factor1##_base<Arg, RealType, Other>&& f,        \
                       const factor2##_base<Arg, RealType, Derived>& g) { \
    return join(op, std::move(f).derived(), g.derived(),                \
                static_cast<Other*>(0), static_cast<Derived*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(factor1##_base<Arg, RealType, Other>&& f,        \
                       factor2##_base<Arg, RealType, Derived>&& g) {    \
    return join(op, std::move(f).derived(), std::move(g).derived(),     \
                static_cast<Other*>(0), static_cast<Derived*>(0));      \
  }

#define LIBGM_JOIN(function, factor, op)        \
  LIBGM_JOIN2(function, factor, factor, op)

#endif
