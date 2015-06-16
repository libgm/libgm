#ifndef LIBGM_UNIVERSE_HPP
#define LIBGM_UNIVERSE_HPP

#include <libgm/argument/process.hpp>
#include <libgm/argument/variable.hpp>

#include <string>
#include <vector>

namespace libgm {

  /**
   * A class that creates and stores basic variables and processes.
   * The lifetime of this class must extend past the lifetime of any
   * object that uses the variables and processes created by this class.
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
     * Creates a new discrete variable with given name and cardinality.
     */
    variable new_discrete_variable(const std::string& name, std::size_t card) {
      objects_.push_back(argument_object::discrete(name, card));
      return variable(objects_.back());
    }

    /**
     * Creates a new discrete variable with given name and levels.
     */
    variable new_discrete_variable(const std::string& name,
                                   const std::vector<std::string>& levels) {
      objects_.push_back(argument_object::discrete(name, levels));
      return variable(objects_.back());
    }

    /**
     * Creates a collection of new discrete variables with given base name
     * and cardinality. Each variable is called <base>id with id in
     * [0, ..., n).
     */
    domain new_discrete_variables(std::size_t n, const std::string& base,
                                  std::size_t card) {
      domain result(n);
      for (std::size_t i = 0; i < n; ++i) {
        result[i] = new_discrete_variable(base + std::to_string(i), card);
      }
      return result;
    }

    /**
     * Creates a collection of new discrete variables with given base name
     * and levels. Each variable is called <base>id with id in [0, ..., n).
     */
    domain new_discrete_variables(std::size_t n, const std::string& base,
                                  const std::vector<std::string>& levels) {
      domain result(n);
      for (std::size_t i = 0; i < n; ++i) {
        result[i] = new_discrete_variable(base + std::to_string(i), levels);
      }
      return result;
    }

    /**
     * Creates a new continuous variable with given name and cardinality.
     */
    variable new_continuous_variable(const std::string& name, std::size_t dim) {
      objects_.push_back(argument_object::continuous(name, dim));
      return variable(objects_.back());
    }

    /**
     * Creates a collection of new continuous variables with given base name
     * and cardinality. Each variable is called <base>id with id in
     * [0, ..., n).
     */
    domain new_continuous_variables(std::size_t n, const std::string& base,
                                    std::size_t dim) {
      domain result(n);
      for (std::size_t i = 0; i < n; ++i) {
        result[i] = new_continuous_variable(base + std::to_string(i), dim);
      }
      return result;
    }

    /**
     * Creates a new discrete-value, discrete-time process with given name and
     * cardinality.
     */
    dprocess new_discrete_dprocess(const std::string& name, std::size_t card) {
      objects_.push_back(argument_object::discrete(name, card));
      return dprocess(objects_.back());
    }

    /**
     * Creates a new discrete-value, discrete-time process with given name and
     * levels.
     */
    dprocess new_discrete_dprocess(const std::string& name,
                                   const std::vector<std::string>& levels) {
      objects_.push_back(argument_object::discrete(name, levels));
      return dprocess(objects_.back());
    }

    /**
     * Creates a collection of new discrete-value, discrete-time process with
     * given base name and cardinality. Each discrete process is called
     * <base>id with id in [0, ..., n).
     */
    dprocess_domain
    new_discrete_dprocesses(std::size_t n, const std::string& base,
                            std::size_t card) {
      dprocess_domain result(n);
      for (std::size_t i = 0; i < n; ++i) {
        result[i] = new_discrete_dprocess(base + std::to_string(i), card);
      }
      return result;
    }

    /**
     * Creates a collection of new discrete-value, discrete-time process with
     * given base name and levels. Each discrete process is called <base>id
     * with id in [0, ..., n).
     */
    dprocess_domain
    new_discrete_dprocesses(std::size_t n, const std::string& base,
                            std::vector<std::string>& levels) {
      dprocess_domain result(n);
      for (std::size_t i = 0; i < n; ++i) {
        result[i] = new_discrete_dprocess(base + std::to_string(i), levels);
      }
      return result;
    }

    /**
     * Creates a new continuous-value, discrete-time process with given name
     * and number of dimensions.
     */
    dprocess new_continuous_dprocess(const std::string& name, std::size_t dim) {
      objects_.push_back(argument_object::continuous(name, dim));
      return dprocess(objects_.back());
    }

    /**
     * Creates a collection of new continuous-value, discrete-time proces with
     * given base name and cardinality. Each discrete process is called
     * <base>id with id in [0, ..., n).
     */
    dprocess_domain
    new_continuous_dprocesses(std::size_t n,
                              const std::string& base, std::size_t dim) {
      dprocess_domain result(n);
      for (std::size_t i = 0; i < n; ++i) {
        result[i] = new_continuous_dprocess(base + std::to_string(i), dim);
      }
      return result;
    }

  private:
    //! The continuous of allocated argument objects.
    std::vector<argument_object*> objects_;

  }; // class universe

} // namespace libgm

#endif
