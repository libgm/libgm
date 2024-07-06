#pragma once

#include <libgm/argument/values.hpp>

#include <tuple>

namespace libgm {

class Assignment {
public:
  /// ?
  Assignment(const Domain& domain, const ShapeMap& map);

  Values values(Arg arg) const;
  Values values(const Domain& domain) const;

  void advance();

private:
  struct Index {
    uint32_t start : 32;
    uint32_t size : 24;
    uin8_t type : 8;
  };

  std::tuple<size_t*, double*, float*> data_;
  std::array<uint32_t, 3> sizes;
  ankerl::nordered_dense::map<Arg, Index> indices_;
};

} // namespace libgm
