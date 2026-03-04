#pragma once

#include <libgm/math/eigen/dense.hpp>
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
    public AssignmentInterface<VectorAssignment<T>, Vector<T>> {
public:
  struct Impl;

  static const vtables::Assignment<VectorAssignment<T>, Vector<T>> vtable;
};

} // namepsace libgm
