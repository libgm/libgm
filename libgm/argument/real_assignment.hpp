#ifndef LIBGM_VECTOR_ASSIGNMENT_HPP
#define LIBGM_VECTOR_ASSIGNMENT_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/variable.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/parser/range_io.hpp>

#include <stdexcept>
#include <unordered_map>

namespace libgm {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment to vector variables.
   * Each vector variable is mapped to an Eigen dynamically-allocated vector.
   *
   * \tparam Var a type that satisfies the ContinuousArgument concept
   */
  template <typename T = double, typename Var = variable>
  using real_assignment =
    std::unordered_map<Var, real_vector<T>, typename argument_traits<Var>::hasher>;

  /**
   * Prints the assignment to an output stream.
   * \relates uint_assignment
   */
  template <typename T, typename Var>
  std::ostream& operator<<(std::ostream& out, const real_assignment<T, Var>& a){
    out << '{';
    bool first = true;
    for (const auto& p : a) {
      if (first) { first = false; } else { out << ','; }
      argument_traits<Var>::print(out, p.first);
      out << ':';
      print_range(out, p.second.data(), p.second.data() + p.second.size(),
                  '[', ' ', ']');
    }
    out << '}';
    return out;
  }

  /**
   * Returns the aggregate size of all vector variables in the assignment.
   * \relates real_assignment
   */
  template <typename T, typename Var>
  std::size_t num_dimensions(const real_assignment<T, Var>& a) {
    std::size_t size = 0;
    for (const auto& p : a) {
      size += argument_traits<Var>::num_dimensions(p.first);
    }
    return size;
  }

  /**
   * Returns the concatenation of (a subset of) vectors in an assignment
   * in the order specified by the given domain.
   * \relates real_assignment
   */
  template <typename T, typename Var>
  real_vector<T>
  extract(const real_assignment<T, Var>& a,
          const basic_domain<Var>& dom) {
    real_vector<T> result(num_dimensions(dom));
    std::size_t i = 0;
    for (Var v : dom) {
      std::size_t n = argument_traits<Var>::num_dimensions(v);
      result.segment(i, n) = a.at(v);
      i += n;
    }
    return result;
  }

  //! @}

} // namespace libgm

#endif
