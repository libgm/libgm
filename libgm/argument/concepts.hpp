#ifndef LIBGM_BASE_CONCEPTS_HPP
#define LIBGM_BASE_CONCEPTS_HPP

#include <libgm/global.hpp>

#include <string>

namespace libgm {

  /**
   * The concept that represents a variable.
   *
   * \ingroup argument_concepts
   */
  template <class V>
  struct Variable {

    //! Returns true iff the variables have equivalent domains
    friend bool compatible(Variable x, Variable y) const;

    concept_usage(Variable) {
      libgm::same_type(var1->compatible(var2), bool_value);
    }

  };
  
  /** 
   * The concept that represents a process.
   * Conceptually, a process is a collection of variables, indexed by some type.
   * Typically, distinct processes have disjoint collections of variables.
   * The variables are created as needed, and are deleted when the 
   * process is destroyed.
   * 
   * \ingroup base_concepts
   */
  template <typename P>
  struct Process {
    
    //! The type of variables used by this process
    typedef typename P::variable_type variable_type;

    //! The type used to index the variables of this process
    typedef typename P::index_type index_type;

    //! Returns the variable for the given index
    variable_type operator()(const index_type& index) const;

    concept_usage(Process) {
      libgm::same_type(proc(index_type()), var);
    }

  private:
    variable_type var;
    P proc;
  };

} // namespace libgm

#endif
