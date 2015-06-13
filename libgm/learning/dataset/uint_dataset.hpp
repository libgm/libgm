#ifndef LIBGM_UINT_DATASET_HPP
#define LIBGM_UINT_DATASET_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/learning/dataset/basic_dataset.hpp>

#include <limits>

namespace libgm {

  /**
   * The traits for a dataset that stores observations for discrete variables.
   *
   * \tparam T the type representing the weights
   * \tparam Var a type that models the DiscreteArgument concept
   */
  template <typename T, typename Var>
  struct uint_data_traits {
    typedef Var                  variable_type;
    typedef basic_domain<Var>    domain_type;
    typedef uint_assignment<Var> assignment_type;
    typedef uint_vector          data_type;
    typedef std::size_t          element_type;
    typedef T                    weight_type;

    //! Returns the number of columns occupied by a variable (always 1).
    static constexpr std::size_t ncols(Var v) {
      return 1;
    }

    //! Returns the number of columns occupied by a domain.
    static std::size_t ncols(const domain_type& dom) {
      return dom.size();
    }

    //! Returns the special "missing" value.
    static constexpr std::size_t missing() {
      return std::numeric_limits<std::size_t>::max();
    }

    //! Extracts the value for a single variable.
    static void copy(std::size_t* const* ptrs,
                     std::size_t ncols,
                     std::size_t row,
                     std::size_t& val) {
      val = ptrs[0][row];
    }

    //! Copies the value for a single variable.
    static void copy(std::size_t val, std::size_t ncols, std::size_t* dest) {
      *dest = val;
    }

  }; // struct uint_data_traits

  /**
   * A dense dataset that stores observations for discrete variables
   * in memory. The observations are stored in the column-major
   * format, i.e., we first store the values for the first variable
   * for all time steps, then for the second variable etc.
   *
   * \tparam T the type representing the weights
   * \tparam Var a type that models the DiscreteArgument concept
   * \see Dataset
   */
  template <typename T = double, typename Var = variable>
  using uint_dataset = basic_dataset<uint_data_traits<T, Var> >;

} // namespace libgm

#endif
