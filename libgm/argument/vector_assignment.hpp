#ifndef LIBGM_VECTOR_ASSIGNMENT_HPP
#define LIBGM_VECTOR_ASSIGNMENT_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/variable.hpp>
#include <libgm/math/eigen/dynamic.hpp>

#include <stdexcept>
#include <unordered_map>

namespace libgm {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment to vector variables.
   * Each vector variable is mapped to an Eigen vector.
   */
  template <typename T = double, typename Var = variable>
  using vector_assignment = std::unordered_map<Var, dynamic_vector<T> >;
  
  /**
   * Returns the aggregate size of all vector variables in the assignment.
   * \relates vector_assignment
   */
  template <typename T, typename Var>
  size_t vector_size(const vector_assignment<T, Var>& a) {
    size_t size = 0;
    for (const auto& p : a) {
      size += p.first.size();
    }
    return size;
  }

  /**
   * Returns the concatenation of (a subset of) vectors in an assignment
   * in the order specified by the given domain.
   * \relates vector_assignment
   */
  template <typename T, typename Var>
  dynamic_vector<T>
  extract(const vector_assignment<T, Var>& a,
          const basic_domain<Var>& dom) {
    dynamic_vector<T> result(vector_size(dom));
    size_t i = 0;
    for (Var v : dom) {
      result.segment(i, v.size()) = a.at(v);
      i += v.size();
    }
    return result;
  }

  //! @}

} // namespace libgm

#endif
