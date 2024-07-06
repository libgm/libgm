#include <iostream>

namespace libgm {

struct Arg {
  char label : 8 = 0;
  uint32_t id : 24 = 0;
  uint32_t index : 32 = 0;
};

std::ostream& operator<<(std::ostream& out, Arg arg);

}  // namespace libgm
