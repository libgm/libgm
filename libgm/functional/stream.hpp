#ifndef LIBGM_FUNCTIONAL_STREAM_HPP
#define LIBGM_FUNCTIONAL_STREAM_HPP

namespace libgm {

  /**
   * A unary operator that, upon invocation, outputs its argument
   * to an archive / stream via operator<<.
   */
  template <typename Stream>
  struct stream_out {
    Stream* out;
    explicit stream_out(Stream& out) : out(&out) { }
    template <typename T>
    void operator()(const T& value) const {
      (*out) << value;
    }
  };

  /**
   * A unary operator that, upon invocation, inputs its argument
   * from am archive / stream via operator>>.
   */
  template <typename Stream>
  struct stream_in {
    Stream* in;
    explicit stream_in(Stream& in) : in(&in) { }
    template <typename T>
    void operator()(T& value) const {
      (*in) >> value;
    }
  };

} // namespace libgm

#endif
