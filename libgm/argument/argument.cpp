#include "argument.hpp"

namespace libgm {

std::ostream& operator<<(std::ostream& out, Arg arg) {
  if (arg.label) out << arg.label; else out << "null";
  if (arg.id) out << arg.id;
  if (arg.index) out << '(' << arg.index << ')';
  return out;
}

}
