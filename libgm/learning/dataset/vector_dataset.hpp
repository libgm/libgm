#ifndef LIBGM_VECTOR_DATASET_HPP
#define LIBGM_VECTOR_DATASET_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/vector_assignment.hpp>
#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/learning/dataset/basic_dataset.hpp>

#include <algorithm>
#include <limits>

namespace libgm {

  /**
   * The traits for a dataset that stores observations for vector variables.
   * \tparam T the type representing the value and weights
   */
  template <typename T, typename Var = variable>
  struct vector_data_traits {
    typedef Var                       variable_type;
    typedef basic_domain<Var>         domain_type;
    typedef vector_assignment<T, Var> assignment_type;
    typedef dynamic_vector<T>         data_type;
    typedef T                         element_type;
    typedef T                         weight_type;

    //! Returns the number of columns occupied by a variable.
    static std::size_t ncols(Var v) {
      return v.size();
    }

    //! Returns the number of columns occupied by a domain.
    static std::size_t ncols(const domain_type& dom) {
      return vector_size(dom);
    }

    //! Returns the special "missing" value.
    static constexpr T missing() {
      return std::numeric_limits<T>::quiet_NaN();
    }

    //! Extracts the values for a single variable occupying ncol columns.
    static void copy(T* const* ptrs, std::size_t ncols, std::size_t row,
                     dynamic_vector<T>& val) {
      val.resize(ncols);
      for (std::size_t i = 0; i < ncols; ++i) {
        val[i] = ptrs[i][row];
      }
    }

    //! Copies the values for a single variable occupying ncol columns.
    static void copy(const dynamic_vector<T>& val, std::size_t ncols, T* dest) {
      assert(val.size() == ncols);
      std::copy(val.data(), val.data() + ncols, dest);
    }

  }; // struct vector_data_traits

  /**
   * A dense dataset that stores observations for vector variables
   * in memory. The observations are stored in the column-major
   * format, i.e., for each variable, we first store the index-0
   * values for all time steps, then index-1 values, etc.
   *
   * \tparam T the type representing the values and weights
   * \see Dataset
   */
  template <typename T = double, typename Var = variable>
  using vector_dataset = basic_dataset<vector_data_traits<T, Var> >;

} // namespace libgm

#endif
