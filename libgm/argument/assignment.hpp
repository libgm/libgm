#ifndef LIBGM_ASSIGNMENT_HPP
#define LIBGM_ASSIGNMENT_HPP

#include <variant>

#include <libgm/config.hpp>

namespace libgm {

class Assignment {
public:
  struct Value {
    uint32_t start;
    uint16_t size;
    uint16_t data;
  };

  using Pointer = std::variant<uint32_t*, float*, double*>;

  struct Data {
    Pointer ptr;
    size_t size;
  };

  Value& operator[](Arg arg) {
    return values_[arg];
  }

  Value at(Arg arg) const {
    return values_.at(arg);
  }

  void advance();

  void append(Data data);

private:
  ankerl::nordered_dense::map<Arg, Value> values_;
};


} // namespace libgm

#endif
