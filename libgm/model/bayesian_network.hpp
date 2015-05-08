#ifndef LIBGM_BAYESIAN_NETWORK_HPP
#define LIBGM_BAYESIAN_NETWORK_HPP

#include <libgm/graph/algorithm/graph_traversal.hpp>
#include <libgm/graph/algorithm/make_clique.hpp>
#include <libgm/graph/directed_graph.hpp>
#include <libgm/graph/property_fn.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/math/logarithmic.hpp>

#include <boost/iterator/transform_iterator.hpp>

#include <random>

namespace libgm {

  /**
   * A Bayesian network with CPDs for each variable.
   *
   * \ingroup model
   */
  template <typename F>
  class bayesian_network
    : public directed_graph<typename F::variable_type, F> {

    typedef directed_graph<typename F::variable_type, F> base;

    // Public type declarations
    //==========================================================================
  public:
    // FactorizedModel types
    typedef typename F::real_type       real_type;
    typedef logarithmic<real_type>      result_type;
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef F                           value_type;

    typedef boost::transform_iterator<
      vertex_property_fn<base>, typename base::vertex_iterator
    > iterator;

    typedef boost::transform_iterator<
      vertex_property_fn<const base>, typename base::vertex_iterator
    > const_iterator;

    // Shortcuts
    typedef typename base::vertex_type vertex_type;

    // Constructors
    //==========================================================================
  public:
    //! Default constructor. Creates an empty Bayesian network.
    bayesian_network() { }

    //! Constructs a Bayesian network with the given variables and no edges.
    explicit bayesian_network(const domain_type& variables) {
      for (variable_type v : variables) {
        this->add_vertex(v);
      }
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this model (the range of all the vertices).
    iterator_range<typename base::vertex_iterator> arguments() const {
      return this->vertices();
    }

    //! Returns the arguments of the factor associated with a variable.
    const domain_type& arguments(variable_type v) const {
      return (*this)[v].arguments();
    }

    //! Returns the iterator to the first factor.
    iterator begin() {
      return iterator(this->vertices().begin(),
                      vertex_property_fn<base>(this));
    }

    //! Returns the iterator to the first factor.
    const_iterator begin() const {
      return const_iterator(this->vertices().begin(),
                            vertex_property_fn<const base>(this));
    }

    //! Returns the iterator past the last factor.
    iterator end() {
      return iterator(this->vertices().end(),
                      vertex_property_fn<base>(this));
    }

    //! Returns the iterator past the last factor.
    const_iterator end() const {
      return const_iterator(this->vertices().end(),
                            vertex_property_fn<const base>(this));
    }

    // Queries
    //==========================================================================

    /**
     * Returns the likelihood of the given assignment.
     * The assignment must include all the arguments of this Bayesian network.
     */
    result_type operator()(const assignment_type& a) const {
      return result_type(log(a), log_tag());
    }

    /**
     * Returns the log-likelihood of the given assignment.
     * The assignment must include all the arguments of this Bayesian network.
     */
    real_type log(const assignment_type& a) const {
      real_type result(0);
      for (const F& f : *this) { result += f.log(a);  }
      return result;
    }

    /**
     * Draws a single sample from a Bayesian network.
     * \tparam Generator a type that models UniformRandomNumberGenerator
     */
    template <typename Generator>
    void sample(Generator& rng, assignment_type& a) const {
      partial_order_traversal(*this, [&](vertex_type v) {
          (*this)[v].sample(rng, {v}, a);
        });
    }

    /**
     * Computes a minimal Markov graph capturing dependencies in this model.
     */
    void markov_graph(undirected_graph<variable_type>& mg) const {
      for (vertex_type v : this->vertices()) {
        mg.add_vertex(v);
        make_clique(mg, arguments(v));
      }
    }

    /**
     * Returns true if the domain of the factor at each vertex includes the
     * vertex itself and its parents.
     */
    bool valid() const {
      for (vertex_type v : this->vertices()) {
        domain_type args(this->parents(v).begin(), this->parents(v).end());
        args.push_back(v);
        if (!equivalent(args, arguments(v))) { return false; }
      }
      return true;
    }

    // Modifiers
    //==========================================================================

    /**
     * Adds a factor representing the conditional distribution p(v | rest) to
     * the graphical model and creates the necessary vertices and edges.
     * If another factor for this vertex already exists, it is overwritten.
     *
     * Note: It is the responsibility of the caller to ensure that the
     * graph remains a DAG.
     */
    void add_factor(variable_type v, const F& f) {
      if (this->contains(v)) {
        this->remove_vertex(v);
      }
      assert(f.arguments().count(v));
      this->add_vertex(v, f);
      for (variable_type u : f.arguments()) {
        if (u != v) {
          this->add_edge(u, v);
        }
      }
    }

    /**
     * Conditions this model on the values in the given assignment and
     * returns the likelihood of the evidence. If the head of the factor
     * is restricted, its tail must be, too.
     */
    result_type condition(const assignment_type& a) {
      // condition each factor, collecting the vertices from the assignment
      real_type ll = 0;
      std::vector<variable_type> removed;
      for (variable_type v : this->vertices()) {
        F& factor = (*this)[v];
        // count the number of arguments restricted
        std::size_t n = 0;
        for (variable_type u : factor.arguments()) {
          n += a.count(u);
        }
        if (n == 0) { // nothing to do
          continue;
        } else if (n == factor.arguments().size()) { // restricted all
          ll += factor.log(a);
          removed.push_back(v);
        } else if (!a.count(v)) { // partial restrict
          factor = factor.restrict(a);
        } else {
          throw std::runtime_error("Unsupported operation");
        }
      }

      // remove the vertices (this will drop the edges as well)
      for (variable_type v : removed) {
        this->remove_vertex(v);
      }
      return result_type(ll, log_tag());
    }

  }; // class bayesian_network

} // namespace libgm

namespace boost {

  //! A traits class that lets bayesian_network work in BGL algorithms
  template <typename F>
  struct graph_traits< libgm::bayesian_network<F> >
    : public graph_traits<libgm::directed_graph<typename F::variable_type, F> >
  { };

} // namespace boost

#endif
