#pragma once

#include <libgm/assignment/real_values.hpp>
#include <libgm/assignment/interfaces/assignment.hpp>

namespace libgm {

/**
 * An assignment to vector variables.
 *
 * Each vector variable can have a value (of given element type) assigned to it.
 * The assignment can be efficiently represented as a map from Arg to Vector<T>.
 */
template <typename T>
class VectorAssignment
  : public Object,
    public AssignmentInterface<VectorAssignment<T>, RealValues<T>> {
public:
  struct Impl;

  static const vtables::Assignment<VectorAssignment<T>, RealValues<T>> vtable;
};

} // namepsace libgm
