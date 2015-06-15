#ifndef LIBGM_REAL_SEQUENCE_DATASET_HPP
#define LIBGM_REAL_SEQUENCE_DATASET_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/real_assignment.hpp>
#include <libgm/argument/process.hpp>
#include <libgm/learning/dataset/basic_sequence_dataset.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/eigen/matrix_index.hpp>
#include <libgm/math/eigen/submatrix.hpp>

#include <unordered_map>

namespace libgm {

  /**
   * The traits for a dataset that stores observations for continuous-valued
   * discrete-time processes.
   *
   * \tparam T the type representing the values and weights
   * \tparam Var a variable type that models the ContinuousArgument concept
   */
  template <typename T, typename Var>
  struct real_sequence_traits {
    typedef process<std::size_t, Var>    process_type;
    typedef Var                          variable_type;
    typedef basic_domain<process_type>   proc_domain_type;
    typedef basic_domain<variable_type>  var_domain_type;
    typedef real_matrix<T>               proc_data_type;
    typedef real_vector<T>               var_data_type;
    typedef real_assignment<T, Var>      assignment_type;
    typedef T                            weight_type;
    typedef matrix_index                 index_type;
    typedef std::pair<real_matrix<T>, T> proc_value_type;
    typedef std::pair<real_vector<T>, T> var_value_type;
    typedef std::unordered_map<process_type, std::size_t>  column_map_type;
    typedef std::unordered_map<variable_type, std::size_t> offset_map_type;

    typedef argument_traits<process_type> process_traits;
    typedef argument_traits<variable_type> variable_traits;

    //! Computes the column indices for the given process domain.
    static void initialize(const proc_domain_type& procs,
                           column_map_type& cols) {
      std::size_t col = 0;
      for (process_type proc : procs) {
        cols.emplace(proc, col);
        col += process_traits::num_dimensions(proc);
      }
    }

    //! Computes the argument set and offsets for a step view.
    static matrix_index initialize(const proc_domain_type& procs,
                                   std::size_t first, std::size_t length,
                                   var_domain_type& variables,
                                   offset_map_type& offsets) {
      std::size_t ndims = num_dimensions(procs);
      std::size_t offset = ndims * first;
      for (std::size_t t = 0; t < length; ++t) {
        for (process_type proc : procs) {
          variable_type v = proc(t);
          variables.push_back(v);
          offsets.emplace(v, offset);
          offset += variable_traits::num_dimensions(v);
        }
      }
      return matrix_index(ndims * first, ndims * length);
    }

    //! Computes the column index for the given domain.
    static index_type
    index(const proc_domain_type& procs, const column_map_type& columns) {
      index_type index;
      for (process_type proc : procs) {
        index.append(columns.at(proc), process_traits::num_dimensions(proc));
      }
      return index;
    }

    //! Computes the linear index for the given domain.
    static index_type
    index(const var_domain_type& vars, const offset_map_type& offsets) {
      index_type index;
      for (variable_type var : vars) {
        index.append(offsets.at(var), variable_traits::num_dimensions(var));
      }
      return index;
    }

    //! Loads the data for a subset of arguments.
    static void load(const proc_value_type& from,
                     const index_type& index,
                     proc_value_type& to) {
      std::size_t nsteps = from.first.cols();
      set(to.first, submat(from.first, index, matrix_index(0, nsteps)));
      to.second = from.second;
    }

    //! Saves the data for a subset of arguments.
    static void save(const proc_value_type& from,
                     const index_type& index,
                     proc_value_type& to) {
      std::size_t nsteps = from.first.cols();
      set(submat(to.first, index, matrix_index(0, nsteps)), from.first);
      to.second = from.second;
    }

    //! Extracts the data for a subset of arguments into an assignment.
    static void extract(const proc_value_type& from,
                        const proc_domain_type& procs,
                        const column_map_type& colmap,
                        std::pair<assignment_type, T>& to) {
      std::size_t nsteps = from.first.cols();
      assignment_type& a = to.first;
      a.clear();
      a.reserve(procs.size() * nsteps);
      for (process_type proc : procs) {
        std::size_t row = colmap.at(proc);
        for (std::size_t t = 0; t < nsteps; ++t) {
          std::size_t n = process_traits::num_dimensions(proc);
          a[proc(t)] = from.first.block(row, t, n, 1);
        }
      }
      to.second = from.second;
    }

    //! Extracts the data for a subset of variables.
    static void extract(const proc_value_type& from,
                        const index_type& index,
                        std::size_t tstart,
                        var_value_type& to) {
      std::size_t offset = tstart * from.first.rows();
      to.first.resize(index.size());
      if (index.contiguous()) {
        const T* begin = from.first.data() + index.start() + offset;
        std::copy(begin, begin + index.size(), to.first.data());
      } else {
        for (std::size_t i = 0; i < to.first.size(); ++i) {
          to.first[i] = from.first(index[i] + offset);
        }
      }
      to.second = from.second;
    }

    //! Extracts the data for a subset of variables into an assignment.
    static void extract(const proc_value_type& from,
                        const var_domain_type& vars,
                        const offset_map_type& offsets,
                        std::size_t tstart,
                        std::pair<assignment_type, T>& to) {
      std::size_t offset = tstart * from.first.rows();
      assignment_type& a = to.first;
      a.clear();
      a.reserve(vars.size());
      for (variable_type var : vars) {
        real_vector<T>& vec = a[var];
        vec.resize(variable_traits::num_dimensions(var));
        const T* begin = from.first.data() + offsets.at(var) + offset;
        std::copy(begin, begin + vec.size(), vec.data());
      }
      to.second = from.second;
    }

    //! Returns an empty sequecnce for the given processes.
    static proc_data_type empty(const proc_domain_type& procs) {
      return proc_data_type(num_dimensions(procs), 0);
    }

    //! Checks if the given value has size compatible with given processes.
    static bool compatible(const proc_data_type& data,
                           const proc_domain_type& procs) {
      return data.rows() == num_dimensions(procs);
    }

  }; // struct real_sequence_traits

  /**
   * A dense dataset that stores observations for continuous-valued,
   * discrete-time processes in memory. Each sample is a matrix with rows
   * corresponding to processes and columns corresponding to the time steps.
   * The samples are stored in an std::vector.
   *
   * \tparam T the type representing the weights
   * \tparam Var a variable type that models the ContinuousArgument concept
   * \see Dataset
   */
  template <typename T = double, typename Var = variable>
  using real_sequence_dataset =
    basic_sequence_dataset<real_sequence_traits<T, Var> >;

} // namespace libgm

#endif
