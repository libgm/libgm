#include "argument.hpp"

namespace libgm {

std::ostream& operator<<(std::ostream& out, Arg arg) {
  if (arg) {
    arg.get().print(out);
  } else {
    out << "null";
  }
  return out;
}

bool operator<(Arg a, Arg b) {
  if (a == b) {
    return false;
  }

  if (!a || !b) {
    return a.ptr() < b.ptr();
  }

  assert(typeid(a.get()) == typeid(b.get()));
  return a.get().less(b.get());
}

} // namespace libgm
