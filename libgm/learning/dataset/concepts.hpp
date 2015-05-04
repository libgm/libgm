#ifndef LIBGM_DATASET_CONCEPTS_HPP
#define LIBGM_DATASET_CONCEPTS_HPP

namespace libgm {

  /**
   * Represents a sequence of dense datapoints, where each column
   * is a variable of the specified type. Supports efficient 
   * iteration over all the data for a specified subset of the columns,
   * in the form of the records of the given type. The record type
   * is variable_type-dependent and provides weight and values in the
   * native (flat) representation of the extracted data, so that many 
   * factor operations, such as sampling or log-likelihood can be
   * performed without additional data lookups/transforms or allocs.
   *
   * \ingroup dataset_concepts
   * \see finite_dataset, vector_dataset, hybrid_dataset
   */
  template <typename DS>
  struct Dataset {

    //! The variable/process type of each column.
    typedef typename DS::argument_type argument_type;

    //! The set of column variables/processes (e.g., finite_domain).
    typedef typename DS::domain_type domain_type;

    //! A sequence of variables/processes (e.g., finite_var_vector).
    typedef typename DS::var_vector_type var_vector_type;

    //! A type that maps variable_type to values (e.g., finite_assignment).
    typedef typename DS::assignment_type assignment_type;

    //! The record that stores the extracted data (e.g., finite_record_old).
    typedef typename DS::record_type record_type;

    //! An iterator that provides mutable access to the data in this dataset.
    typedef typename DS::record_iterator record_iterator;

    //! An iterator that provides const access to the data in this dataset.
    typedef typename DS::const_record_iterator const_record_iterator;

    //! Returns the number of datapoints in the dataset.
    size_t size() const;

    //! Returns true if the dataset has no datapoints.
    bool empty() const;

    //! Returns the variables in this dataset (may return a const-reference).
    const domain_type arguments() const;

    //! Returns the variables in this dataset in the dataset's internal order.
    const var_vector_type& arg_vector() const;

    //! Returns a single datapoint in the dataset's internal variable order.
    //! This function may not be implemented efficiently.
    record_type record(size_t row) const;

    //! Returns a single datapoint for a subset of columns in the given order.
    //! This function may not be implemented efficiently.
    record_type record(size_t row, const var_vector_type& vars) const;

    //! Provides mutable access to a subset of columns in the specified order.
    std::pair<record_iterator, record_iterator>
    records(const var_vector_type& vars);

    //! Provides const access to a subset of columns in the specified order.
    std::pair<const_record_iterator, const_record_iterator>
    records(const var_vector_type& vars) const;

    //! Returns a random datapoint drawn from this dataset.
    template <typename RandomNumberGenerator>
    record_type sample(const var_vector_type& vars, RandomNumberGenerator& rng) const;
    
  };

  /**
   * Dataset that permits the insertion of new datapoints.
   * \see finite_memory_dataset, vector_memory_dataset
   * \ingroup dataset_concepts
   */
  template <typename DS>
  struct InsertableDataset : public Dataset<DS> {

    //! Inserts a new datapoint. The values in the record must be in the
    //! internal ordering of the dataset.
    void insert(const record_type& record);

    //! Inserts a new datapoint. The values in the assignment must include
    //! all the arguments (columns) of this datset.
    void insert(const assignment_type& a, typename record_type::weight_type w);

    //! Inserts the given number of datapoints with unit weights and
    //! variable_type-specific special undefined values.
    void insert(size_t nrows);

  };

} // namespace libgm

#endif
