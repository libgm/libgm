#ifndef LIBGM_DYNAMIC_BAYESIAN_NETWORK_HPP
#define LIBGM_DYNAMIC_BAYESIAN_NETWORK_HPP

#include <libgm/global.hpp>
#include <libgm/base/discrete_process.hpp>
#include <libgm/graph/directed_graph.hpp>
#include <libgm/model/bayesian_network.hpp>

#include <iosfwd>

namespace libgm {

  /**
   * A dynamic Bayesian network.
   * @tparam F a type that models the DistributionFactor concept
   *
   * \ingroup model
   */
  template <typename F>
  class dynamic_bayesian_network {

    // Public type declarations
    // =========================================================================
  public:

    typedef typename F::domain_type domain_type;

    typedef typename F::variable_type variable_type;

    //! The type of processes used in this network
    typedef discrete_process<variable_type> process_type;
   
    // Private data members
    // =========================================================================
  private:
    //! The arguments of this DBN
    std::set<process_type> processes_;

    //! The prior model
    bayesian_network<F> prior;

    //! The transition model
    //! \todo This really should be a two_step_bayesian_network
    bayesian_network<F> transition;

    // Constructors and conversion operators
    // =========================================================================
  public:
    //! Creates an empty DBN with a Markov network prior
    dynamic_bayesian_network()  { }

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }

    // Accessors
    // =========================================================================
    //! Returns the processes in this DBN
    const std::set<process_type>& processes() const {
      return processes_;
    }

    //! Returns the prior model
    const bayesian_network<F>& prior_model() const {
      return prior;
    }

    //! Returns the transition model
    const bayesian_network<F>& transition_model() const {
      return transition;
    }

    //! Returns the CPD of the transition model for a given process
    const F& operator[](process_type p) const {
      return transition[p->next()];
    }

    //! Returns the CPD of the transition model for a given time-t+1 variable
    const F& operator[](variable_type v) const {
      return transition[v];
    }
    
    // parents, graph information etc.?

    // Queries
    // =========================================================================
    /**
     * Returns the ancestors of the t+1-time variables in the transition model.
     * \param procs The set of processes whose ancestors are being sought
     */
    domain_type ancestors(const std::set<process_type>& procs) const {
      return transition.ancestors(variables(procs, next_step));
    }

    /**
     * Unrolls the dynamic Bayesian network over steps 0, ..., n.
     * The unrolled network contains 
     */
    bayesian_network<F> unroll(size_t n) const {
      // Initialize the prior
      std::map<variable_type, variable_type> prior_var_map
        = make_process_var_map(processes(), current_step, 0);
      bayesian_network<F> bn;
      for (process_type p : processes()) {
        F factor = prior.factor(p->current());
        factor.subst_args(prior_var_map);
        bn.add_factor(p->at(0), factor);
      }
      
      // Add the n transition models
      for(size_t t = 0; t < n; t++) {
        std::map<variable_type, variable_type> var_map
          = map_union(make_process_var_map(processes(), current_step, t),
                      make_process_var_map(processes(), next_step, t+1));
        for (process_type p : processes()) {
          F cpd = transition[p->next()];
          cpd.subst_args(var_map);
          bn.add_factor(p->at(t+1), cpd);
        }
      }
      return bn;
    }

    //! Throws an assertion violation if the DBN is not valid
    void check_valid() const {
      // The variables at the current and next time step
      domain_type vars_t  = variables(processes(), current_step);
      domain_type vars_t1 = variables(processes(), next_step);

      // Check the prior and the transition model
      assert(prior.arguments() == vars_t);
      assert(is_superset(transition.arguments(),vars_t1));
      assert(is_subset(transition.arguments(), set_union(vars_t, vars_t1)));
    }
    
    // Modifiers
    // =========================================================================
    /**
     * Adds a new CPD to the transition model.
     * @param factor
     *        A factor that represents the conditional probability distribution.
     *        The arguments of this factor must be either time-t or time-t+1
     *        variables of one or more timed processes.
     * @param p
     *        The process, for which the CPD is being added.  The argument 
     *        factor must contain the t+1-step variable of process p. 
     */
    void add_factor(process_type p, const F& factor) {
      assert(factor.arguments().count(p->next()) > 0);
      for (variable_type v : factor.arguments()) {
        int t = boost::any_cast<int>(v.index());
        assert(t == current_step || t == next_step);
      }
      processes_.insert(p);
      transition.add_factor(p->next(), factor);
    }

    /**
     * Adds a new factor to the prior distribution.
     * @param factor
     *        A factor that represents the conditional probability distribution.
     *        The arguments of this factor must be time-t variables of one or
     *        more timed processes.
     * @param v
     *        The head of the conditional probability distribution.
     */
    void add_factor(variable_type head, const F& factor) {
      assert(factor.arguments().count(head) > 0);
      check_index(factor.arguments(), current_step);
      prior.add_factor(head, factor);
    }

    /**
     * Removes all processes and factors from this DBN.
     */
    void clear() {
      prior.clear();
      transition.clear();
      processes_.clear();
    }
       

  }; // class dynamic_bayesian_network

  //! \relates dynamic_bayesian_network
  template <typename F>
  std::ostream& operator<<(std::ostream& out,
                           const dynamic_bayesian_network<F>& dbn) {
    using std::endl;
    out << "#DBN(" << dbn.processes() << ")" << endl;
    out << "Prior:" << endl;
    out << dbn.prior_model();
    out << "Transition model:" << endl;
    out << dbn.transition_model();
    return out;
  }
  
}

#endif
