#ifndef LIBGM_FINITE_ASSIGNMENT_HPP
#define LIBGM_FINITE_ASSIGNMENT_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/variable.hpp>
#include <libgm/datastructure/uint_vector.hpp>

#include <unordered_map>

namespace libgm {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment to finite variables.
   * Each variable is mapped to a finite value.
   *
   * \tparam Var a type that satisfies the DiscreteArgument concept
   */
  template <typename Var = variable>
  using uint_assignment =
    std::unordered_map<Var, std::size_t, typename argument_traits<Var>::hasher>;

  /**
   * Prints the assignment to an output stream.
   * \relates uint_assignment
   */
  template <typename Var>
  std::ostream& operator<<(std::ostream& out, const uint_assignment<Var>& a) {
    out << '{';
    bool first = true;
    for (const auto& p : a) {
      if (first) { first = false; } else { out << ','; }
      argument_traits<Var>::print(out, p.first);
      out << ':' << p.second;
    }
    out << '}';
    return out;
  }

  /**
   * Returns the finite values in the assignment for a subset of arguments
   * in the order specified by the given domain.
   * \relates uint_assignment
   */
  template <typename Var>
  uint_vector extract(const uint_assignment<Var>& a,
                      const basic_domain<Var>& dom,
                      std::size_t start = 0) {
    assert(start <= dom.size());
    uint_vector result(dom.size() - start);
    for (std::size_t i = start; i < dom.size(); ++i) {
      result[i - start] = a.at(dom[i]);
    }
    return result;
  }

  //! @}

} // namespace libgm

#endif
