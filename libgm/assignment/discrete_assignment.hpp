#pragma once

#include <vector>
#include <libgm/assignment/interfaces/assignment.hpp>

namespace libgm {

/**
 * An assignment to discrete variables.
 *
 * Each discrete variable can have only one value (of type size_t) assigned to it.
 * So the assignment can be efficiently represented as a map from Arg to size_t.
 */
class DiscreteAssignment
  : public Object,
    public AssignmentInterface<DiscreteAssignment, std::vector<size_t>> {
public:
  struct Impl;

  static const VTable vtable;
};

} // namepsace libgm
