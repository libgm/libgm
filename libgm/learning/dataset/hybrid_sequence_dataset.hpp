#ifndef LIBGM_HYBRID_SEQUENCE_DATASET_HPP
#define LIBGM_HYBRID_SEQUENCE_DATASET_HPP

#include <libgm/argument/hybrid_assignment.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/argument/process.hpp>
#include <libgm/datastructure/hybrid_matrix.hpp>
#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/math/eigen/matrix_index.hpp>
#include <libgm/math/eigen/submatrix.hpp>

#include <unordered_map>

namespace libgm {

  /**
   * The traits for a dataset that stores observations for hybrid
   * discrete processes.
   * \tparam T the type representing the values and weights
   */
  template <typename T, typename Var>
  struct hybrid_sequence_traits {
    typedef process<size_t, Var>        process_type;
    typedef variable                    variable_type;
    typedef hybrid_domain<process_type> proc_domain_type;
    typedef hybrid_domain<...>          var_domain_type;
    typedef hybrid_matrix<T>            proc_data_type;
    typedef hybrid_vector<T>            var_data_type;
    typedef hybrid_assignment<T>        assignment_type;
    typedef T                           weight_type;
    typedef std::pair<hybrid_matrix<T>, T> proc_value_type;
    typedef std::pair<hybrid_vector<T>, T> var_value_type;
    typedef std::unordered_map<process_type, size_t>  column_map_type;
    typedef std::unordered_map<variable_type, size_t> offset_map_type;
    struct index_type {
      matrix_index finite;
      matrix_index vector;
    };

    //! Computes the column indices for the given domain.
    static void initialize(const proc_domain_type& args, col_map_type& cols) {
      size_t fcol = 0;
      for (process_type proc : args.finite()) {
        cols.emplace(proc, fcol);
        ++fcol;
      }
      size_t vcol = 0;
      for (process_type proc : arg.vector()) {
        cols.emplace(proc, vcol);
        vol += proc->size();
      }
    }

    //! Computes the column index for the given domain.
    static index_type
    index(const proc_domain_type& procs, const column_map_type& columns) {
      index_type index;
      for (process_type proc : procs.finite()) {
        index.finite.append(columns.at(proc), 1);
      }
      for (process_type proc : procs.vector()) {
        index.vector.append(columns.at(proc), proc->size());
      }
      return index;
    }

    //! Loads the data for a subset of arguments.
    static void load(const proc_value_type& from,
                     const index_type& index,
                     proc_value_type& to) {
      matrix_index all(0, from.first.cols());
      set(to.first.finite(), submat(from.first.finite(), index.finite, all));
      set(to.first.vector(), submat(from.first.vector(), index.vector, all));
      to.second = from.second;
    }

    //! Saves the data for a subset of arguments.
    static void save(const proc_value_type& from,
                     const index_type& index,
                     proc_value_type& to) {
      matrix_index all(0, from.first.cols());
      set(submat(to.first.finite(), index.finite, all), from.first.finite());
      set(submat(to.first.vector(), index.vector, all), from.first.vector());
      to.second = from.second;
    }

    //! Etracts the data for a subset of arguments into an assignment.
    static void extract(const proc_value_type& from,
                        const proc_domain_type& procs,
                        const column_map_type& colmap,
                        std::pair<assignment_type, T>& to) {
      size_t nsteps = value.first.cols();
      assignment_type& a = to.first;
      a.clear();
      a.reserve(args.size() * nsteps);
      for (process_type proc : procs.finite()) {
        size_t row = colmap.at(proc);
        for (size_t t = 0; t < nsteps; ++t) {
          a[proc->at(t)] = from.first.finite()(row, t);
        }
      }
      for (process_type proc : procs.finite()) {
        size_t row = colmap.at(proc);
        for (size_t t = 0; t < nsteps; ++t) {
          a[proc->at(t)] = from.first.vector().block(row, t, proc->size(), 1);
        }
      }
      to.second = from.second;
    }

    //! Checks if the given value has size compatible with given arguments
    static bool compatible(const data_type& data, const domain_type& args) {
      return
        data.finite().rows() == args.finite_size() &&
        data.vector().rows() == args.vector_size();
    }

  }; // struct hybrid_sequence_traits

  /**
   * A dense dataset that stores observations for hybrid discrete processes
   * in memory. Each observation is a matrix with rows being the processes
   * and columns being the time steps. The observations are stored in an
   * std::vector.
   *
   * \tparam T the type representing the weights
   * \see Dataset
   */
  template <typename T, typename Var>
  using hybrid_sequence_dataset =
    basic_sequence_dataset<hybrid_sequence_traits<T, Var> >;
  
} // namespace libgm

#endif
