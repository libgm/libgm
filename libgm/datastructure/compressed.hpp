#ifndef LIBGM_COMPRESSED_HPP
#define LIBGM_COMPRESSED_HPP

namespace libgm {

  template <typename T>
  struct compressed_workspace { protected: mutable T ws_; };

  template <>
  struct compressed_workspace<void> { };

} // namespace libgm

#endif
