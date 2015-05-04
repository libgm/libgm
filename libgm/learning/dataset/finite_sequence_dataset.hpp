#ifndef LIBGM_FINITE_SEQUENCE_DATASET_HPP
#define LIBGM_FINITE_SEQUENCE_DATASET_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/finite_assignment.hpp>
#include <libgm/argument/process.hpp>
#include <libgm/learning/dataset/basic_sequence_dataset.hpp>
#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/math/eigen/matrix_index.hpp>
#include <libgm/math/eigen/submatrix.hpp>

#include <unordered_map>

namespace libgm {

  /**
   * The traits for a dataset that stores observations for finite
   * discrete processes.
   *
   * \tparam T the type representing the weights
   */
  template <typename T, typename Var>
  struct finite_sequence_traits {
    typedef process<size_t, Var>         process_type;
    typedef Var                          variable_type;
    typedef basic_domain<process_type>   proc_domain_type;
    typedef basic_domain<variable_type>  var_domain_type;
    typedef dynamic_matrix<size_t>       proc_data_type;
    typedef finite_index                 var_data_type;
    typedef finite_assignment<Var>       assignment_type;
    typedef T                            weight_type;
    typedef matrix_index                 index_type;
    typedef std::pair<proc_data_type, T> proc_value_type;
    typedef std::pair<var_data_type, T>  var_value_type;
    typedef std::unordered_map<process_type, size_t>  column_map_type;
    typedef std::unordered_map<variable_type, size_t> offset_map_type;

    //! Computes the column indices for the given process domain.
    static void initialize(const proc_domain_type& procs, column_map_type& cols) {
      size_t col = 0;
      for (process_type proc : procs) {
        cols.emplace(proc, col);
        ++col;
      }
    }

    //! Computes the argument set and offsets for a step view.
    static matrix_index initialize(const proc_domain_type& procs,
                                   size_t first, size_t length,
                                   var_domain_type& variables,
                                   offset_map_type& offsets) {
      size_t offset = procs.size() * first;
      for (size_t t = 0; t < length; ++t) {
        for (process_type proc : procs) {
          variable_type v = proc(t);
          variables.push_back(v);
          offsets.emplace(v, offset++);
        }
      }
      return matrix_index(procs.size() * first, procs.size() * length);
    }

    //! Computes the column index for the given domain.
    static index_type
    index(const proc_domain_type& procs, const column_map_type& columns) {
      index_type index;
      for (process_type proc : procs) {
        index.append(columns.at(proc), 1);
      }
      return index;
    }

    //! Computes the linear index for the given domain.
    static index_type
    index(const var_domain_type& vars, const offset_map_type& offsets) {
      index_type index;
      for (variable_type var : vars) {
        index.append(offsets.at(var), 1);
      }
      return index;
    }

    //! Loads the data for a subset of arguments.
    static void load(const proc_value_type& from,
                     const index_type& index,
                     proc_value_type& to) {
      size_t nsteps = from.first.cols();
      set(to.first, submat(from.first, index, matrix_index(0, nsteps)));
      to.second = from.second;
    }

    //! Saves the data for a subset of arguments.
    static void save(const proc_value_type& from,
                     const index_type& index,
                     proc_value_type& to) {
      size_t nsteps = from.first.cols();
      set(submat(to.first, index, matrix_index(0, nsteps)), from.first);
      to.second = from.second;
    }

    //! Extracts the data for a subset of arguments into an assignment.
    static void extract(const proc_value_type& from,
                        const proc_domain_type& procs,
                        const column_map_type& colmap,
                        std::pair<assignment_type, T>& to) {
      size_t nsteps = from.first.cols();
      assignment_type& a = to.first;
      a.clear();
      a.reserve(procs.size() * nsteps);
      for (process_type proc : procs) {
        size_t row = colmap.at(proc);
        for (size_t t = 0; t < nsteps; ++t) {
          a[proc(t)] = from.first(row, t);
        }
      }
      to.second = from.second;
    }

    //! Extracts the data for a subset of variables.
    static void extract(const proc_value_type& from,
                        const index_type& index,
                        size_t tstart,
                        var_value_type& to) {
      size_t offset = tstart * from.first.rows();
      to.first.resize(index.size());
      if (index.contiguous()) {
        const size_t* begin = from.first.data() + index.start() + offset;
        std::copy(begin, begin + index.size(), to.first.data());
      } else {
        for (size_t i = 0; i < to.first.size(); ++i) {
          to.first[i] = from.first(index[i] + offset);
        }
      }
      to.second = from.second;
    }

    //! Extracts the data for a subset of variable into an assignment.
    static void extract(const proc_value_type& from,
                        const var_domain_type& vars,
                        const offset_map_type& offsets,
                        size_t tstart,
                        std::pair<assignment_type, T>& to) {
      size_t offset = tstart * from.first.rows();
      assignment_type& a = to.first;
      a.clear();
      a.reserve(vars.size());
      for (variable_type var : vars) {
        a[var] = from.first(offsets.at(var) + offset);
      }
      to.second = from.second;
    }

    //! Returns an empty sequence for the given processes.
    static proc_data_type empty(const proc_domain_type& procs) {
      return proc_data_type(procs.size(), 0);
    }

    //! Checks if the given value has size compatible with given arguments.
    static bool compatible(const proc_data_type& data,
                           const proc_domain_type& procs) {
      return data.rows() == procs.size();
    }

  }; // struct finite_sequence_traits

  /**
   * A dense dataset that stores observations for finite discrete processes
   * in memory. Each observation is a matrix with rows being the processes
   * and columns being the time steps. The observations are stored in an
   * std::vector.
   *
   * \tparam T the type representing the weights
   * \see Dataset
   */
  template <typename T = double, typename Var = variable>
  using finite_sequence_dataset =
    basic_sequence_dataset<finite_sequence_traits<T, Var> >;
  
} // namespace libgm

#endif
