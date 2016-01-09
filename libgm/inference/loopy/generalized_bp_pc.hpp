#ifndef LIBGM_GBP_PC_HPP
#define LIBGM_GBP_PC_HPP

#include <libgm/factor/invalid_operation.hpp>
#include <libgm/factor/util/diff_fn.hpp>
#include <libgm/graph/region_graph.hpp>

#include <algorithm>
#include <unordered_map>

namespace libgm {

  /**
   * A class that implements the parent-to-child generalized belief
   * propagation algorithm.
   *
   * @tparam F A type that implements the Factor concept.
   *
   * \ingroup inference
   */
  template <typename F>
  class generalized_bp_pc {

    // Public types
    //==========================================================================
  public:
    // FactorizedInference types
    typedef typename F::real_type        real_type;
    typedef typename F::result_type      result_type;
    typedef typename F::argument_type    argument_type;
    typedef typename F::domain_type      domain_type;
    typedef typename F::assignment_type  assignment_type;
    typedef F                            factor_type;
    typedef region_graph<domain_type, F> graph_type;

    // Vertex and edge types
    typedef typename graph_type::vertex_type vertex_type;
    typedef typename graph_type::edge_type edge_type;

    // Constructors and initialization
    //==========================================================================
  public:
    //! Constructs the algorithm to the given graph and difference fn.
    generalized_bp_pc(const region_graph<domain_type, F>& graph,
                      diff_fn<F> diff)
      : graph_(graph), diff_(std::move(diff)) {
      compute_belief_edges();
      compute_message_edges();
      initialize_messages();
    }

    // Constructs the algorithm to the given graph structure and difference fn.
    generalized_bp_pc(const region_graph<domain_type>& graph, diff_fn<F> diff)
      : diff_(std::move(diff)) {
      graph_.structure_from(graph);
      compute_belief_edges();
      compute_message_edges();
      initialize_messages();
      initialize_factors();
    }

    //! Destructor.
    virtual ~generalized_bp_pc() { }

    //! Initializes the messages to unity.
    void initialize_messages() {
      for (edge_type e : graph_.edges()) {
        message_[e] = F(graph_.cluster(e.target()), result_type(1));
      }
    }

    //! Initializes the factors to unity.
    void initialize_factors() {
      for (id_t v : graph_.vertices()) {
        graph_[v] = F(graph_.cluster(v), result_type(1));
      }
    }

    //! Initializes the factors to those in a range.
    template <typename Range>
    void initialize_factors(const Range& factors) {
      initialize_factors();
      for (const F& factor : factors) {
        id_t v = graph_.find_root_cover(factor.arguments());
        assert(v);
        graph_[v] *= factor;
      }
    }

    //! Performs one iteration
    virtual double iterate(double eta) = 0;

    //! Returns the underlying region graph.
    const graph_type& graph() const {
      return graph_;
    }

    //! Returns a belief for a region
    F belief(id_t v) const {
      F result = graph_[v];
      for (edge_type e : belief_edges_.at(v)) {
        result *= message_.at(e);
      }
      return result.normalize();
    }

    //! Returns the marginal over a set of variables
    F belief(const domain_type& vars) const {
      id_t v = graph_.find_cover(vars);
      assert(v);
      return belief(v).marginal(vars);
    }

    // Implementation
    //==========================================================================
  protected:
    //! Precomputes which messages contribute to the belief.
    void compute_belief_edges() {
      for (id_t u : graph_.vertices()) {
        edge_vector& edges = belief_edges_[u];
        std::unordered_set<id_t> descendants = graph_.descendants(u);
        descendants.insert(u);
        // messages from external sources to regions in down+(u)
        for (id_t v : descendants) {
          for (edge_type in : graph_.in_edges(v)) {
            if (!descendants.count(in.source())) {
              edges.push_back(in);
            }
          }
        }
      }
    }

    //! Precomputes which messages contribute to a message.
    void compute_message_edges() {
      for (edge_type e : graph_.edges()) {
        id_t u = e.source();
        id_t v = e.target();
        std::unordered_set<id_t> descendants_u = graph_.descendants(u);
        std::unordered_set<id_t> descendants_v = graph_.descendants(v);
        descendants_u.insert(u);
        descendants_v.insert(v);

        // numerator: edges from sources external to u that are outside
        // the scope of influence of v. \todo verify this
        edge_vector& numerator_edges = numerator_edges_[e];
        for (id_t w : descendants_u) {
          if (!descendants_v.count(w)) {
            for (edge_type in : graph_.in_edges(w)) {
              if (!descendants_u.count(in.source())) {
                numerator_edges.push_back(in);
              }
            }
          }
        }

        // denominator: information passed from u to regions below v indirectly
        // \todo verify this
        edge_vector& denominator_edges = denominator_edges_[e];
        for (id_t w : descendants_v) {
          for (edge_type in : graph_.in_edges(w)) {
            if (in != e &&
                descendants_u.count(in.source()) &&
                !descendants_v.count(in.source())) {
              denominator_edges.push_back(in);
            }
          }
        }
      }
    }

    //! Passes a message along an edge
    real_type pass_message(edge_type e, real_type eta) {
      F new_msg = graph_[e.source()];
      for (edge_type in : numerator_edges_.at(e)) {
        new_msg *= message_.at(in);
      }
      for (edge_type in : denominator_edges_.at(e)) {
        new_msg /= message_.at(in);
      }
      try {
        new_msg = new_msg.marginal(graph_.cluster(e.target())).normalize();
      } catch (invalid_operation&) {
        std::cerr << ".";
        new_msg = F(graph_.cluster(e.target()), result_type(1));
      }

      // compute the residual and update the message
      F& msg = message_[e];
      real_type residual = diff_(new_msg, msg);
      msg = (eta == 1) ? new_msg : weighted_update(msg, new_msg, eta);
      return residual;
    }

    // Protected members
    //==========================================================================
  protected:
    //! A vector of edges.
    typedef std::vector<edge_type> edge_vector;

    //! The underlying region graph.
    region_graph<domain_type, F> graph_;

    //! The function computing difference between the parameters of two factors.
    diff_fn<F> diff_;

    //! The messages.
    std::unordered_map<edge_type, F> message_;

    //! The edges that are used to compute the belief over a region.
    std::unordered_map<id_t, edge_vector> belief_edges_;

    //! The edges in the numerator of the message.
    std::unordered_map<edge_type, edge_vector> numerator_edges_;

    //! The edges in the denominator of the message.
    std::unordered_map<edge_type, edge_vector> denominator_edges_;

  }; // class generalized_bp_pc


  /**
   * GBP engine that updates the messages in a topological order.
   * according to the natural order of messages.
   *
   * \ingroup inference
   */
  template <typename F>
  class asynchronous_generalized_bp_pc : public generalized_bp_pc<F> {
    typedef generalized_bp_pc<F> base;

  public:
    // bring in some types that are already in the base
    typedef typename F::real_type    real_type;
    typedef typename F::domain_type  domain_type;
    typedef typename base::edge_type edge_type;

    asynchronous_generalized_bp_pc(const region_graph<domain_type, F>& graph,
                                   diff_fn<F> diff)
      : base(graph, std::move(diff)) { }


    asynchronous_generalized_bp_pc(const region_graph<domain_type>& graph,
                                   diff_fn<F> diff)
      : base(graph, std::move(diff)) { }

    real_type iterate(real_type eta) override {
      real_type residual(0);
      // pass the messages downwards \todo is this correct?
      for (edge_type e : this->graph_.edges()) {
        residual = std::max(residual, this->pass_message(e, eta));
      }
      return residual;
    }

  }; // class asynchronous_generalized_bp_pc

} // namespace libgm

#endif
