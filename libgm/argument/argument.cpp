#include "argument.hpp"

namespace libgm {

std::unique_ptr<int[]> Arg::priority;

void Arg::set_ordering(const std::string& ordering) {
  // Allocate the array of priorities
  if (!priority) {
    priority.reset(new int[256]);
  }

  // Populate the given labels with priority [0; ordering.size())
  std::fill_n(priority.get(), 256, -1);
  for (size_t i = 0; i < ordering.size(); ++i) {
    if (priority[ordering[i]] == -1) {
      priority[ordering[i]] = i;
    } else {
      throw std::invalid_argument("Duplicate label '" + std::string(1, ordering[i]) + "'");
    }
  }

  // Populate the remaining labels with priority [ordering.size(); 256)
  for (size_t i = 0, value = ordering.size(); i < 256; ++i) {
    if (priority[i] == -1) {
      priority[i] = value++;
    }
  }
}

void Arg::reset_ordering() {
  priority.reset();
}

std::ostream& operator<<(std::ostream& out, Arg arg) {
  if (arg.label) out << arg.label; else out << "null";
  if (arg.id) out << arg.id;
  if (arg.index) out << '(' << arg.index << ')';
  return out;
}

}
