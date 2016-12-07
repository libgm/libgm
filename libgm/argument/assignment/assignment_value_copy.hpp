#ifndef LIBGM_ASSIGNMENT_VALUE_COPY_HPP
#define LIBGM_ASSIGNMENT_VALUE_COPY_HPP

#include <libgm/argument/traits.hpp>

namespace libgm {

  template <typename Arg, typename Arity = argument_arity_t<Arg> >
  struct assignment_value_copy;

  template <typename Arg>
  struct assignment_value_copy<Arg, univariate_arg> {
    template <typename Scalar>
    void operator()(Arg /* arg */, Scalar value, Scalar*& dest) const {
      *dest++ = values;
    }
  };

  template <typename Arg>
  struct assignment_value_copy<Arg, multivariate_tag> {
    template <typename Vector>
    void operator()(Arg arg, const Vector& values,
                    typename vector_value<Vector>::type*& dest) const {
      assert(values.size() == argument_arity(arg));
      dest = std::copy(values.data(), values.data() + values.size(), dest);
    }
  };

}

#endif
