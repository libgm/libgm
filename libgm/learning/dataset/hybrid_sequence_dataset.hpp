#ifndef LIBGM_HYBRID_SEQUENCE_DATASET_HPP
#define LIBGM_HYBRID_SEQUENCE_DATASET_HPP

#include <libgm/argument/hybrid_assignment.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/argument/process.hpp>
#include <libgm/datastructure/hybrid_matrix.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/eigen/matrix_index.hpp>
#include <libgm/math/eigen/submatrix.hpp>

#include <unordered_map>

namespace libgm {

  /**
   * The traits for a dataset that stores observations for discrete- and
   * continuous-valued, discrete-time processes.
   *
   * \tparam T the type representing the values and weights
   * \tparam Var a variable type that models the MixedArgument concept
   */
  template <typename T, typename Var>
  struct hybrid_sequence_traits {
    typedef process<std::size_t, Var>      process_type;
    typedef variable                       variable_type;
    typedef hybrid_domain<process_type>    proc_domain_type;
    typedef hybrid_domain<Var>             var_domain_type;
    typedef hybrid_matrix<T>               proc_data_type;
    typedef hybrid_vector<T>               var_data_type;
    typedef hybrid_assignment<T>           assignment_type;
    typedef T                              weight_type;
    typedef std::pair<hybrid_matrix<T>, T> proc_value_type;
    typedef std::pair<hybrid_vector<T>, T> var_value_type;
    typedef std::unordered_map<process_type, std::size_t>  column_map_type;
    typedef std::unordered_map<variable_type, std::size_t> offset_map_type;
    struct index_type {
      matrix_index uint;
      matrix_index real;
    };

    //! Computes the column indices for the given domain.
    static void initialize(const proc_domain_type& args, col_map_type& cols) {
      std::size_t ucol = 0;
      for (process_type proc : args.discrete()) {
        cols.emplace(proc, ucol);
        ++ucol;
      }
      std::size_t rcol = 0;
      for (process_type proc : arg.continuous()) {
        cols.emplace(proc, rcol);
        rcol += proc->size();
      }
    }

    //! Computes the column index for the given domain.
    static index_type
    index(const proc_domain_type& procs, const column_map_type& columns) {
      index_type index;
      for (process_type proc : procs.discrete()) {
        index.uint.append(columns.at(proc), 1);
      }
      for (process_type proc : procs.continuous()) {
        index.real.append(columns.at(proc), num_dimensions(proc));
      }
      return index;
    }

    //! Loads the data for a subset of arguments.
    static void load(const proc_value_type& from,
                     const index_type& index,
                     proc_value_type& to) {
      matrix_index all(0, from.first.cols());
      set(to.first.uint(), submat(from.first.uint(), index.uint, all));
      set(to.first.real(), submat(from.first.real(), index.real, all));
      to.second = from.second;
    }

    //! Saves the data for a subset of arguments.
    static void save(const proc_value_type& from,
                     const index_type& index,
                     proc_value_type& to) {
      matrix_index all(0, from.first.cols());
      set(submat(to.first.uint(), index.uint, all), from.first.uint());
      set(submat(to.first.real(), index.real, all), from.first.real());
      to.second = from.second;
    }

    //! Etracts the data for a subset of arguments into an assignment.
    static void extract(const proc_value_type& from,
                        const proc_domain_type& procs,
                        const column_map_type& colmap,
                        std::pair<assignment_type, T>& to) {
      std::size_t nsteps = value.first.cols();
      assignment_type& a = to.first;
      a.clear();
      a.reserve(args.size() * nsteps);
      for (process_type proc : procs.discrete()) {
        std::size_t row = colmap.at(proc);
        for (std::size_t t = 0; t < nsteps; ++t) {
          a[proc(t)] = from.first.uint()(row, t);
        }
      }
      for (process_type proc : procs.discrete()) {
        std::size_t row = colmap.at(proc);
        for (std::size_t t = 0; t < nsteps; ++t) {
          a[proc(t)] = from.first.real().block(row, t, num_dimensions(proc), 1);
        }
      }
      to.second = from.second;
    }

    //! Checks if the given value has size compatible with given arguments
    static bool compatible(const data_type& data, const domain_type& args) {
      return
        data.uint().rows() == args.num_values() &&
        data.real().rows() == args.num_dimensions();
    }

  }; // struct hybrid_sequence_traits

  /**
   * A dense dataset that stores observations for discrete- and contiuous-valued
   * discrete-time processes in memory. Each sample is a matrix with rows being
   * the processes and columns being the time steps. The samples are stored in
   * an std::vector.
   *
   * \tparam T the type representing the weights
   * \tparam Var a variable type that models the MixedArgument concept
   * \see Dataset
   */
  template <typename T, typename Var>
  using hybrid_sequence_dataset =
    basic_sequence_dataset<hybrid_sequence_traits<T, Var> >;

} // namespace libgm

#endif
