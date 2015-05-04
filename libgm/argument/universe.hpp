#ifndef LIBGM_UNIVERSE_HPP
#define LIBGM_UNIVERSE_HPP

#include <libgm/global.hpp>
#include <libgm/argument/process.hpp>
#include <libgm/argument/variable.hpp>

#include <string>
#include <vector>

namespace libgm {

  /**
   * A class that creates and stores basic variables and processes.
   * Conceptually, this is just a vector of variables and processes.
   */
  class universe {
  public:
    //! Default constructor. Creates an empty universe.
    universe() { }

    //! Destructor. Deletes the stored variables and processes.
    ~universe() {
      for (argument_object* obj : objects_) { delete obj; }
    }

    /**
     * Creates a new finite variable with given name and cardinality.
     */
    variable new_finite_variable(const std::string& name, size_t card) {
      objects_.push_back(argument_object::finite(name, card));
      return variable(objects_.back());
    }

    /**
     * Creates a collection of new finite variables with given base name
     * and cardinality. Each variable is called <base>id with id in
     * [0, ..., n).
     */
    domain new_finite_variables(size_t n, const std::string& base, size_t card) {
      domain result(n);
      for (size_t i = 0; i < n; ++i) {
        result[i] = new_finite_variable(base + std::to_string(i), card);
      }
      return result;
    }

    /**
     * Creates a new vector variable with given name and cardinality.
     */
    variable new_vector_variable(const std::string& name, size_t dim) {
      objects_.push_back(argument_object::vector(name, dim));
      return variable(objects_.back());
    }

    /**
     * Creates a collection of new vector variables with given base name
     * and cardinality. Each variable is called <base>id with id in
     * [0, ..., n).
     */
    domain new_vector_variables(size_t n, const std::string& base, size_t dim) {
      domain result(n);
      for (size_t i = 0; i < n; ++i) {
        result[i] = new_vector_variable(base + std::to_string(i), dim);
      }
      return result;
    }

    /**
     * Creates a new finite discrete process with given name and cardinality.
     */
    dprocess new_finite_dprocess(const std::string& name, size_t card) {
      objects_.push_back(argument_object::finite(name, card));
      return dprocess(objects_.back());
    }

    /**
     * Creates a collection of new finite discrete process with given base name
     * and cardinality. Each discrete process is called <base>id with id in
     * [0, ..., n).
     */
    dprocess_domain
    new_finite_dprocesss(size_t n, const std::string& base, size_t card) {
      dprocess_domain result(n);
      for (size_t i = 0; i < n; ++i) {
        result[i] = new_finite_dprocess(base + std::to_string(i), card);
      }
      return result;
    }

    /**
     * Creates a new vector discrete process with given name and cardinality.
     */
    dprocess new_vector_dprocess(const std::string& name, size_t dim) {
      objects_.push_back(argument_object::vector(name, dim));
      return dprocess(objects_.back());
    }

    /**
     * Creates a collection of new vector discrete proces with given base name
     * and cardinality. Each discrete process is called <base>id with id in
     * [0, ..., n).
     */
    dprocess_domain
    new_vector_dprocesss(size_t n, const std::string& base, size_t dim) {
      dprocess_domain result(n);
      for (size_t i = 0; i < n; ++i) {
        result[i] = new_vector_dprocess(base + std::to_string(i), dim);
      }
      return result;
    }
    
  private:
    //! The vector of allocated argument objects.
    std::vector<argument_object*> objects_;

  }; // class universe

} // namespace libgm

#endif
