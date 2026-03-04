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

  const Argument* a_ptr = a.ptr();
  const Argument* b_ptr = b.ptr();
  assert(typeid(*a_ptr) == typeid(*b_ptr));
  return a_ptr->less(*b_ptr);
}

} // namespace libgm
