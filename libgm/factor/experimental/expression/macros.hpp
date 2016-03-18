#ifndef LIBGM_EXPRESSION_MACROS_HPP
#define LIBGM_EXPRESSION_MACROS_HPP

// Macros for common expressions (used inside expression bases)
//==============================================================================

namespace libgm {
  void transform();
}

#define LIBGM_TRANSFORM_RIGHT(function, op, scalar, base, ...)          \
  friend auto function(const base<__VA_ARGS__, Derived>& f, scalar x) { \
    return f.derived().transform(op);                                   \
  }                                                                     \
                                                                        \
  friend auto function(base<__VA_ARGS__, Derived>&& f, scalar x) {      \
    return std::move(f).derived().transform(op);                        \
  }

#define LIBGM_TRANSFORM_LEFT(function, op, scalar, base, ...)           \
  friend auto function(scalar x, const base<__VA_ARGS__, Derived>& f) { \
    return f.derived().transform(op);                                   \
  }                                                                     \
                                                                        \
  friend auto function(scalar x, base<__VA_ARGS__, Derived>&& f) {      \
    return std::move(f).derived().transform(op);                        \
  }

#define LIBGM_TRANSFORM_RIGHT2(function, op1, op2, scalar, base, ...)   \
  friend auto function(const base<__VA_ARGS__, Derived>& f, scalar x) { \
    return f.derived().transform(op1, op2);                             \
  }                                                                     \
                                                                        \
  friend auto function(base<__VA_ARGS__, Derived>&& f, scalar x) {      \
    return std::move(f).derived().transform(op1, op2);                  \
  }

#define LIBGM_TRANSFORM_LEFT2(function, op1, op2, scalar, base, ...)    \
  friend auto function(scalar x, const base<__VA_ARGS__, Derived>& f) { \
    return f.derived().transform(op1, op2);                             \
  }                                                                     \
                                                                        \
  friend auto function(scalar x, base<__VA_ARGS__, Derived>&& f) {      \
    return std::move(f).derived().transform(op1, op2);                  \
  }

#define LIBGM_TRANSFORM(function, op, base, ...)                        \
  template <typename Other>                                             \
  friend auto function(const base<__VA_ARGS__, Derived>& f,             \
                       const base<__VA_ARGS__, Other>& g) {             \
    using libgm::transform;                                             \
    return transform(op, f.derived(), g.derived(),                      \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const base<__VA_ARGS__, Derived>& f,             \
                       base<__VA_ARGS__, Other>&& g) {                  \
    using libgm::transform;                                             \
    return transform(op, f.derived(), std::move(g).derived(),           \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base<__VA_ARGS__, Derived>&& f,                  \
                       const base<__VA_ARGS__, Other>& g) {             \
    using libgm::transform;                                             \
    return transform(op, std::move(f).derived(), g.derived(),           \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base<__VA_ARGS__, Derived>&& f,                  \
                       base<__VA_ARGS__, Other>&& g) {                  \
    using libgm::transform;                                             \
    return transform(op, std::move(f).derived(), std::move(g).derived(),\
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }

#define LIBGM_TRANSFORM_SCALAR(function, op, scalar, base, ...)         \
  template <typename Other>                                             \
  friend auto function(const base<__VA_ARGS__, Derived>& f,             \
                       const base<__VA_ARGS__, Other>& g,               \
                       scalar x) {                                      \
    using libgm::transform;                                             \
    return transform(op, f.derived(), g.derived(),                      \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const base<__VA_ARGS__, Derived>& f,             \
                       base<__VA_ARGS__, Other>&& g,                    \
                       scalar x) {                                      \
    using libgm::transform;                                             \
    return transform(op, f.derived(), std::move(g).derived(),           \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base<__VA_ARGS__, Derived>&& f,                  \
                       const base<__VA_ARGS__, Other>& g,               \
                       scalar x) {                                      \
    using libgm::transform;                                             \
    return transform(op, std::move(f).derived(), g.derived(),           \
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base<__VA_ARGS__, Derived>&& f,                  \
                       base<__VA_ARGS__, Other>&& g,                    \
                       scalar x) {                                      \
    using libgm::transform;                                             \
    return transform(op, std::move(f).derived(), std::move(g).derived(),\
                     static_cast<Derived*>(0), static_cast<Other*>(0)); \
  }

#define LIBGM_OUTER(function, op, base, ...)                            \
  template <typename Other>                                             \
  friend auto function(const base<__VA_ARGS__, Derived>& f,             \
                       const base<__VA_ARGS__, Other>& g) {             \
    return outer(op, f.derived(), g.derived(),                          \
                 static_cast<Derived*>(0), static_cast<Other*>(0));     \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const base<__VA_ARGS__, Derived>& f,             \
                       base<__VA_ARGS__, Other>&& g) {                  \
    return outer(op, f.derived(), std::move(g).derived(),               \
                 static_cast<Derived*>(0), static_cast<Other*>(0));     \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base<__VA_ARGS__, Derived>&& f,                  \
                       const base<__VA_ARGS__, Other>& g) {             \
    return outer(op, std::move(f).derived(), g.derived(),               \
                 static_cast<Derived*>(0), static_cast<Other*>(0));     \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base<__VA_ARGS__, Derived>&& f,                  \
                       base<__VA_ARGS__, Other>&& g) {                  \
    return outer(op, std::move(f).derived(), std::move(g).derived(),    \
                 static_cast<Derived*>(0), static_cast<Other*>(0));     \
  }

#define LIBGM_JOIN(function, op, base, ...)                             \
  template <typename Other>                                             \
  friend auto function(const base<__VA_ARGS__, Derived>& f,             \
                       const base<__VA_ARGS__, Other>& g) {             \
    return join(op, f.derived(), g.derived(),                           \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const base<__VA_ARGS__, Derived>& f,             \
                       base<__VA_ARGS__, Other>&& g) {                  \
    return join(op, f.derived(), std::move(g).derived(),                \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base<__VA_ARGS__, Derived>&& f,                  \
                       const base<__VA_ARGS__, Other>& g) {             \
    return join(op, std::move(f).derived(), g.derived(),                \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base<__VA_ARGS__, Derived>&& f,                  \
                       base<__VA_ARGS__, Other>&& g) {                  \
    return join(op, std::move(f).derived(), std::move(g).derived(),     \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }

#define LIBGM_JOIN_RIGHT(function, op, base, other, ...)                \
  template <typename Other>                                             \
  friend auto function(const other<__VA_ARGS__, Other>& f,              \
                       const base& g) {                                 \
    return join(op, f.derived(), g.derived(),                           \
                static_cast<Other*>(0), static_cast<Derived*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const other<__VA_ARGS__, Other>& f,              \
                       base&& g) {                                      \
    return join(op, f.derived(), std::move(g).derived(),                \
                static_cast<Other*>(0), static_cast<Derived*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(other<__VA_ARGS__, Other>&& f,                   \
                       const base& g) {                                 \
    return join(op, std::move(f).derived(), g.derived(),                \
                static_cast<Other*>(0), static_cast<Derived*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(other<__VA_ARGS__, Other>&& f,                   \
                       base&& g) {                                      \
    return join(op, std::move(f).derived(), std::move(g).derived(),     \
                static_cast<Other*>(0), static_cast<Derived*>(0));      \
  }

#define LIBGM_JOIN_LEFT(function, op, base, other, ...)                 \
  template <typename Other>                                             \
  friend auto function(const base& f,                                   \
                       const other<__VA_ARGS__, Other>& g) {            \
    return join(op, f.derived(), g.derived(),                           \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(const base& f,                                   \
                       other<__VA_ARGS__, Other>&& g) {                 \
    return join(op, f.derived(), std::move(g).derived(),                \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base&& f,                                        \
                       const other<__VA_ARGS__, Other>& g) {            \
    return join(op, std::move(f).derived(), g.derived(),                \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }                                                                     \
                                                                        \
  template <typename Other>                                             \
  friend auto function(base&& f,                                        \
                       other<__VA_ARGS__, Other>&& g) {                 \
    return join(op, std::move(f).derived(), std::move(g).derived(),     \
                static_cast<Derived*>(0), static_cast<Other*>(0));      \
  }

#define LIBGM_AGGREGATE(function, domain, ...)                  \
  auto function(domain retain) const& {                         \
    return derived().aggregate(__VA_ARGS__, retain);            \
  }                                                             \
                                                                \
  auto function(domain retain) && {                             \
    return std::move(derived()).aggregate(__VA_ARGS__, retain); \
  }

#define LIBGM_ELIMINATE(function, ...)                          \
  auto function() const& {                                      \
    return this->derived().eliminate(__VA_ARGS__);              \
  }                                                             \
                                                                \
  auto function() && {                                          \
    return std::move(this->derived()).eliminate(__VA_ARGS__);   \
  }

#define LIBGM_BLOCK(function, vector, vals, expr, ...)          \
  expr<__VA_ARGS__, const Derived&>                             \
  function(vector vals) const& {                                \
    return { derived(), vals };                                 \
  }                                                             \
                                                                \
  expr<__VA_ARGS__, Derived>                                    \
  function(vector vals) && {                                    \
    return { std::move(derived()), vals };                      \
  }

#define LIBGM_RESTRICT(domain, dims, vector, vals, expr, ...)   \
  expr<__VA_ARGS__, const Derived&>                             \
  restrict(domain dims, vector vals) const& {                   \
    return { derived(), dims, vals };                           \
  }                                                             \
                                                                \
  expr<__VA_ARGS__, Derived>                                    \
  restrict(domain dims, vector vals) && {                       \
    return { std::move(derived()), dims, vals };                \
  }

#define LIBGM_SELECT0(function, selector, ...)                  \
  selector<__VA_ARGS__, Derived&>                               \
  function() & {                                                \
    return { derived() };                                       \
  }                                                             \
                                                                \
  selector<__VA_ARGS__, const Derived&>                         \
  function() const& {                                           \
    return { derived() };                                       \
  }                                                             \
                                                                \
  selector<__VA_ARGS__, Derived>                                \
  function() && {                                               \
    return { std::move(derived()) };                            \
  }

#define LIBGM_SELECT1(function, domain, dims, selector, ...)    \
  selector<__VA_ARGS__, Derived&>                               \
  function(domain dims) & {                                     \
    return { derived(), dims };                                 \
  }                                                             \
                                                                \
  selector<__VA_ARGS__, const Derived&>                         \
  function(domain dims) const& {                                \
    return { derived(), dims };                                 \
  }                                                             \
                                                                \
  selector<__VA_ARGS__, Derived>                                \
  function(domain dims) && {                                    \
    return { std::move(derived()), dims };                      \
  }

/*
#define LIBGM_MATRIX_CONDITIONAL(division_op)                     \
  auto conditional(const unary_domain<Arg>& tail) const& {        \
    return make_matrix_conditional(derived(), tail, division_op); \
  }                                                               \
                                                                  \
  auto conditional(const unary_domain<Arg>& tail) && {            \
    return make_matrix_conditional(std::move(derived()), tail, division_op); \
  }
*/

/*
#define LIBGM_TABLE_REORDER() \
  auto reorder(const uint_vector& order) const {  \
    return ...;                                  \
  }                                              \
                                                 \
  auto reorder(const uint_vector& order) && {     \
    return ...;                                  \
  }
*/

#endif
