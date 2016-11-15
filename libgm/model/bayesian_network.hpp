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
   * A Bayesian network with CPDs for each argument.
   *
   * \ingroup model
   */
  template <typename Arg, typename F>
  class bayesian_network
    : public directed_graph<Arg, std::pair<domain<Arg>, F> > {

    using base = directed_graph<Arg, std::pair<const domain<Arg>, F> >;

    // Public type declarations
    //--------------------------------------------------------------------------
  public:
    // Argument types
    using argument_type     = Arg;
    using argument_hasher   = typename argument_traits<Arg>::hasher;
    using argument_iterator = typename graph_type::vertex_iterator;

    // Factor types
    using real_type      = typename F::real_type;
    using result_type    = typename F::result_type;
    using factor_type    = F;
    using value_type     = std::pair<const domain<Arg>, F>;
    using iterator       = boost::transform_iterator<
      vertex_property_fn<base>, typename base::vertex_iterator>;
    using const_iterator = boost::transform_iterator<
      vertex_property_fn<const base>, typename base::vertex_iterator>;

    //typedef typename F::assignment_type assignment_type;

    // Shortcuts
    //typedef typename base::vertex_type vertex_type;

    // Constructors
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty Bayesian network.
    bayesian_network() { }

    //! Constructs a Bayesian network with the given arguments and no edges.
    explicit bayesian_network(const domain<Arg>& args) {
      for (Arg arg : args) {
        this->add_vertex(arg);
      }
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the number of arguments (the same as the number of vertices).
    std::size_t num_arguments() const {
      return this->num_vertices();
    }

    //! Returns the arguments of this model (the range of all the vertices).
    iterator_range<argument_iterator> arguments() const {
      return this->vertices();
    }

    //! Returns the arguments of the factor with the given head.
    const domain<Arg>& arguments(Arg head) const {
      return (*this)[arg].first;
    }

    //! Returns the iterator to the first factor.
    iterator begin() {
      return { this->vertices().begin(), vertex_property_fn<base>(this) };
    }

    //! Returns the iterator to the first factor.
    const_iterator begin() const {
      return { this->vertices().begin(), vertex_property_fn<const base>(this) };
    }

    //! Returns the iterator past the last factor.
    iterator end() {
      return { this->vertices().end(), vertex_property_fn<base>(this) };
    }

    //! Returns the iterator past the last factor.
    const_iterator end() const {
      return { this->vertices().end(), vertex_property_fn<const base>(this) };
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
    void markov_graph(undirected_graph<Arg>& mg) const {
      for (Arg v : this->vertices()) {
        mg.add_vertex(v);
        make_clique(mg, arguments(v));
      }
    }

    /**
     * Returns true if the domain of the factor at each vertex includes the
     * vertex itself and its parents.
     */
    bool valid() const {
      for (Arg v : this->vertices()) {
        domain<Arg> args(this->parents(v));
        args.push_back(v);
        if (!equivalent(args, arguments(v))) { return false; }
      }
      return true;
    }

    // Modifiers
    //--------------------------------------------------------------------------

    /**
     * Adds a factor representing the conditional distribution p(v | rest) to
     * the graphical model and creates the necessary vertices and edges.
     * If another factor for this vertex already exists, it is overwritten.
     *
     * Note: It is the responsibility of the caller to ensure that the
     * graph remains a DAG.
     */
    void add_factor(const domain<Arg> args, const F& f) {
      Arg head = args.front();
      if (this->contains(head)) {
        this->remove_vertex(head);
      }
      this->add_vertex(head, value_type(args, f));
      for (Arg arg : args) {
        if (arg != head) {
          this->add_edge(head, arg);
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
      std::vector<argument_type> removed;
      for (argument_type v : this->vertices()) {
        F& factor = (*this)[v];
        // count the number of arguments restricted
        std::size_t n = 0;
        for (argument_type u : factor.arguments()) {
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
      for (argument_type v : removed) {
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
    : public graph_traits<libgm::directed_graph<typename F::argument_type, F> >
  { };

} // namespace boost

#endif
