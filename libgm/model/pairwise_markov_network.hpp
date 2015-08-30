#ifndef LIBGM_PAIRWISE_MARKOV_NETWORK_HPP
#define LIBGM_PAIRWISE_MARKOV_NETWORK_HPP

#include <libgm/graph/property_fn.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/iterator/join_iterator.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/range/transformed.hpp>
#include <libgm/traits/is_range.hpp>
#include <libgm/traits/pairwise_compatible.hpp>

namespace libgm {

  /**
   * Implements a Markov network with pairwise potentials.
   *
   * \tparam NodeF The type of factors associated with vertices.
   * \tparam EdgeF The type of factors associated with edges.
   *               Must be pairwise_compatible with NodeF.
   *
   * \ingroup model
   */
  template <typename NodeF, typename EdgeF = NodeF>
  class pairwise_markov_network
    : public undirected_graph<typename NodeF::argument_type, NodeF, EdgeF> {
    static_assert(pairwise_compatible<NodeF, EdgeF>::value,
                  "The node and edge factors are not pairwise compatible");

    typedef undirected_graph<typename NodeF::argument_type, NodeF, EdgeF> base;

    // Public type declarations
    //==========================================================================
  public:
    // FactorizedModel types
    typedef typename NodeF::real_type       real_type;
    typedef logarithmic<real_type>          result_type;
    typedef typename NodeF::argument_type   argument_type;
    typedef typename NodeF::domain_type     node_domain_type;
    typedef typename EdgeF::domain_type     edge_domain_type;
    typedef typename NodeF::assignment_type assignment_type;

    // Additional types
    typedef boost::transform_iterator<
      vertex_property_fn<const base>, typename base::vertex_iterator
    > node_factor_iterator;
    typedef boost::transform_iterator<
      edge_property_fn<const base>, typename base::edge_iterator
    > edge_factor_iterator;
    typedef NodeF value_type;

    // Shortcuts
    typedef typename base::vertex_type vertex_type;
    typedef typename base::edge_type edge_type;

    // Constructors
    //==========================================================================
  public:
    //! Default constructor. Creates an empty pairwise Markov network.
    pairwise_markov_network() { }

    /**
     * Constructs a pairwise Markov network with the given vertices.
     * \tparam Range A forward range over elements convertible to argument_type
     */
    template <typename Range>
    explicit pairwise_markov_network(
        const Range& vertices,
        typename std::enable_if<is_range<Range, vertex_type>::value>::type* = 0) {
      for (vertex_type v : vertices) {
        this->add_vertex(v);
      }
    }

    /**
     * Constructs a pairwise Markov network from a collection of factors.
     * \tparam Range A forward range over elements convertible to NodeF or EdgeF
     */
    template <typename Range>
    explicit pairwise_markov_network(
        const Range& factors,
        typename std::enable_if<is_range<Range, NodeF>::value ||
                                is_range<Range, EdgeF>::value>::type* = 0) {
      for (const auto& factor : factors) {
        add_factor(factor);
      }
    }

    /**
     * Converts a pairwise Markov network from one type of factors to another.
     * The factors must have the same argument type as this class..
     */
    template <typename OtherNodeF, typename OtherEdgeF>
    explicit pairwise_markov_network(
        const pairwise_markov_network<OtherNodeF, OtherEdgeF>& g) {
      for (argument_type v : g.vertices()) {
        this->add_vertex(v, NodeF(g[v]));
      }
      for (edge_type e : g.edges()) {
        this->add_edge(e.source(), e.target(), EdgeF(g[e]));
      }
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this model (i.e., the range of all vertices).
    iterator_range<typename base::vertex_iterator> arguments() const {
      return this->vertices();
    }

    //! Returns the arguments of the factor associated with a vertex.
    const node_domain_type& arguments(argument_type v) const {
      return (*this)[v].arguments();
    }

    //! Returns the arguments of the factor associated with an edge.
    const edge_domain_type& arguments(const edge_type& e) const {
      return (*this)[e].arguments();
    }

    //! Returns the factors associated with vertices.
    iterator_range<node_factor_iterator>
    node_factors() const {
      return make_transformed(this->vertices(),
                              vertex_property_fn<const base>(this));
    }

    //! Returns the factors associated with the edges.
    iterator_range<edge_factor_iterator>
    edge_factors() const {
      return make_transformed(this->edges(),
                              edge_property_fn<const base>(this));
    }

    /**
     * Returns iterator to the first factor in the model.
     * This function is only available if NodeF and EdgeF is the same type.
     */
    template <bool B = true>
    join_iterator<node_factor_iterator, edge_factor_iterator>
    begin() const {
      auto nf = node_factors();
      auto ef = edge_factors();
      return make_join_iterator(nf.begin(), nf.end(), ef.begin());
    }

    /**
     * Returns iterator to past the last factor in the model.
     * This function is only available if NodeF and EdgeF is the same type.
     */
    template <bool B = true>
    join_iterator<node_factor_iterator, edge_factor_iterator>
    end() const {
      auto nf = node_factors();
      auto ef = edge_factors();
      return make_join_iterator(nf.end(), nf.end(), ef.end());
    }

    // Queries
    //==========================================================================

    /**
     * Returns the unnormalized likelihood of the given assignment.
     * The assignment must include all the arguments of this Markov network.
     */
    result_type operator()(const assignment_type& a) const {
      return result_type(log(a), log_tag());
    }

    /**
     * Returns the unnormalized log-likelihood of the given assignment.
     * The assignment must include all the arguments of this Markov network.
     */
    real_type log(const assignment_type& a) const {
      real_type result(0);
      for (const NodeF& f : node_factors()) {
        result += f.log(a);
      }
      for (const EdgeF& f : edge_factors()) {
        result += f.log(a);
      }
      return result;
    }

    /**
     * Computes a minimal Markov graph capturing dependencies in this model.
     */
    void markov_graph(undirected_graph<argument_type>& mg) const {
      for (vertex_type v : this->vertices()) {
        mg.add_vertex(v);
      }
      for (edge_type e : this->edges()) {
        mg.add_edge(e.source(), e.target());
      }
    }

    /**
     * Returns true if the domain of the factor matches the vertex / vertices.
     */
    bool valid() const {
      for (vertex_type v : this->vertices()) {
        if (!equivalent(arguments(v), node_domain_type({v}))) {
          return false;
        }
      }
      for (edge_type e : this->edges()) {
        edge_domain_type dom({e.source(), e.target()});
        if (!equivalent(arguments(e), dom)) {
          return false;
        }
      }
      return true;
    }

    // Mutators
    //==========================================================================
    /**
     * Initializes the node and edge potentials with the given functors.
     * This is performed by invoking the functors on the arguments given
     * by each vertex and edge and assigning the result to the model.
     */
    void initialize(std::function<NodeF(const node_domain_type&)> nodefn,
                    std::function<EdgeF(const edge_domain_type&)> edgefn) {
      typename NodeF::result_type one(1);
      for (vertex_type v : this->vertices()) {
        (*this)[v] = nodefn ? nodefn({v}) : NodeF({v}, one);
      }
      for (edge_type e : this->edges()) {
        edge_domain_type dom({e.source(), e.target()});
        (*this)[e] = edgefn ? edgefn(dom) : EdgeF(dom, one);
      }
    }

    /**
     * Adds a factor to the graphical model and creates the vertices and edges.
     * If the corresponding vertex/edge already exists, does nothing.
     * Only supported when NodeF and EdgeF refere to the same type.
     *
     * \return bool indicating whether the factor was inserted
     * \throw std::invalid_argument if the factor is not unary or binary
     */
    template <bool B = std::is_same<NodeF, EdgeF>::value>
    typename std::enable_if<B, bool>::type
    add_factor(const NodeF& factor) {
      const node_domain_type& args = factor.arguments();
      switch (args.size()) {
        case 1:
	  return this->add_vertex(*args.begin(), factor);
        case 2:
	  return this->add_edge(*args.begin(), *++args.begin(), factor).second;
        default:
          throw std::invalid_argument("Unsupported factor arity " +
                                      std::to_string(args.size()));
      }
    }

    /**
     * Conditions the model on an assignment. This restricts any edge
     * factor whose argument is contained in a and multiplies it to
     * the adjacent node factor. The normalizing constant is not preserved.
     */
    void condition(const assignment_type& a) {
      for (const auto& p : a) {
        argument_type u = p.first;
        if (this->contains(u)) {
          for (edge_type e : this->out_edges(u)) {
            if (!a.count(e.target())) {
              (*this)[e.target()] *= (*this)[e].restrict(a);
            }
          }
          this->remove_vertex(u);
        }
      }
    }

  }; // class pairwise_markov_network

} // namespace libgm

#endif
