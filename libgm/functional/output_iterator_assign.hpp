#ifndef LIBGM_OUTPUT_ITERATOR_ASSIGN_HPP
#define LIBGM_OUTPUT_ITERATOR_ASSIGN_HPP

#include <utility>

namespace libgm {

  /**
   * A function object that accepts a single argument and at invocation,
   * assigns the argument to an output iterator.
   *
   * \tparam OutIt the underlying output iterator type.
   */
  template <typename OutIt>
  class output_iterator_assign {
  public:
    output_iterator_assign(OutIt out)
      : out_(out) { }

    template <typename Argument>
    void operator()(Argument&& arg) {
      *out_ = std::forward<Argument>(arg);
      ++out_;
    }

  private:
    OutIt out_;
  };

  /**
   * A convenience function that constructs output_iterator_assign objects
   * of correct type.
   */
  template <typename OutIt>
  output_iterator_assign<OutIt> make_output_iterator_assign(OutIt out) {
    return output_iterator_assign<OutIt>(out);
  }

} // namespace libgm

#endif
