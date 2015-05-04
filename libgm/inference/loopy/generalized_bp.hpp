#ifndef LIBGM_GENERALIZED_BP_HPP
#define LIBGM_GENERALIZED_BP_HPP

#include <libgm/factor/invalid_operation.hpp>
#include <libgm/factor/util/diff_fn.hpp>
#include <libgm/functional/hash.hpp>
#include <libgm/functional/output_iterator_assign.hpp>
#include <libgm/graph/algorithm/graph_traversal.hpp>
#include <libgm/graph/region_graph.hpp>
#include <libgm/range/reversed.hpp>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace libgm {

  /**
   * A class that implements generalized synchronous generalized belief
   * propagation.
   *
   * @tparam F A type that implements the Factor concept.
   *
   * \ingroup inference
   */
  template <typename F>
  class generalized_bp {

    // Public types
    //==========================================================================
  public:
    // FactorizedInference types
    typedef typename F::real_type        real_type;
    typedef typename F::result_type      result_type;
    typedef typename F::variable_type    variable_type;
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
    generalized_bp(const region_graph<domain_type, F>& graph, diff_fn<F> diff) 
      : graph_(graph), diff_(std::move(diff)) {
      initialize_messages();
    }

    // Constructs the algorithm to the given graph structure and difference fn.
    generalized_bp(const region_graph<domain_type>& graph, diff_fn<F> diff)
      : diff_(std::move(diff)) {
      graph_.structure_from(graph);
      initialize_messages();
      initialize_factors();
    }

    //! Destructor.
    virtual ~generalized_bp() { }

    //! Initializes the messages to unity.
    void initialize_messages() {
      for (edge_type e : graph_.edges()) {
        size_t u = e.source();
        size_t v = e.target();
        const domain_type& args = graph_.separator(e);
        pseudo_message(u, v) = F(args, result_type(1)); 
        pseudo_message(v, u) = F(args, result_type(1));
        message(u, v) = F(args, result_type(1));
        message(v, u) = F(args, result_type(1));
      }
    }

    //! Initializes the factors to unity.
    void initialize_factors() {
      for (size_t v : graph_.vertices()) {
        graph_[v] = F(graph_.cluster(v), result_type(1));
      }
    }

    //! Initializes the factors to those in a range.
    template <typename Range>
    void initialize_factors(const Range& factors) {
      initialize_factors();
      for (const F& factor : factors) {
        size_t v = graph_.find_root_cover(factor.arguments());
        assert(v);
        graph_[v] *= factor;
      }
    }

    // Running the algorithm
    //==========================================================================

    //! Performs one iteration
    virtual real_type iterate(real_type eta) = 0;

    //! Returns the underlying region graph.
    const graph_type& graph() const {
      return graph_;
    }

    //! Returns the belief for a region.
    F belief(size_t v) const {
      F result = pow(graph_[v], graph_.counting(v));
      for (size_t u : graph_.parents(v)) {
        result *= message(u, v);
      }
      for (size_t u : graph_.children(v)) {
        result *= message(u, v);
      }
      return result.normalize();
    }

    //! Returns the marginal over a set of variables
    F belief(const domain_type& vars) const {
      size_t v = graph_.find_cover(vars); 
      assert(v);
      return belief(v).marginal(vars);
    }

    // Implementation
    //==========================================================================
  protected:
    //! Returns a pseudo-message from region u to region v
    F& pseudo_message(size_t u, size_t v) {
      return pseudo_message_[std::make_pair(u, v)];
    }

    //! Returns the message from region u to region v
    F& message(size_t u, size_t v) {
      return message_[std::make_pair(u, v)];
    }

    //! Returns the message from region u to region v
    const F& message(size_t u, size_t v) const {
      return message_.at(std::make_pair(u, v));
    }

    /**
     * Passes a message from region u to region v.
     * u and v must be adjacent.
     */
    real_type pass_message(size_t u, size_t v, real_type eta) {
      bool down = graph_.contains(u, v); /* true means the edge is u->v */
      real_type br = down ? beta(v) : beta(u);
      const domain_type& sep =
        down ? graph_.separator(u, v) : graph_.separator(v, u);

      // compute the pseudo message (this is m0 in the Yedidia paper)
      F m0 = pow(graph_[u], graph_.counting(u));
      for (size_t w : graph_.parents(u)) {
        if (w != v) m0 *= message(w, u);
      }
      for (size_t w : graph_.children(u)) {
        if (w != v) m0 *= message(w, u);
      }
      try {
        pseudo_message(u, v) = m0.marginal(sep).normalize();
      } catch(std::invalid_argument& e) {
        std::ostringstream out;
        out << "Encountered invalid argument" << std::endl;
        out << "c_r = " << graph_.counting(u) << std::endl;
        out << "beta = " << br << std::endl;
        out << m0 << std::endl;
        out << pow(graph_[u], graph_.counting(u)) << std::endl;
        for (size_t w : graph_.parents(u)) {
          if (w != v) out << message(w, u);
        }
        for (size_t w : graph_.children(u)) {
          if (w != v) out << message(w, u);
        }
        throw std::runtime_error(out.str());
      } catch(invalid_operation& exc) {
        std::cerr << ".";
        pseudo_message(u, v) = F(sep, result_type(1));
      }
      
      // compute the true message
      F new_msg;
      new_msg  = pow(pseudo_message(u, v), br);
      new_msg *= pow(pseudo_message(v, u), br - real_type(1));
      new_msg.normalize();

      // update the message and compute the residual
      F& msg = message(u, v);
      new_msg = weighted_update(msg, new_msg, eta);
      swap(msg, new_msg);
      return diff_(msg, new_msg);
    }

    //! Returns the beta coefficient for vertex v.
    real_type beta(size_t v) const {
      if (graph_.in_degree(v) > 0) {
        real_type qr = real_type(1 - graph_.counting(v)) / graph_.in_degree(v);
        real_type br = real_type(1) / (real_type(2) - qr);
        assert(qr != 2);
        return br;
      } else
        return real_type(1);
    }

    // Protected members
    //==========================================================================
  protected:
    //! A map type that holds messages
    typedef std::unordered_map<std::pair<size_t, size_t>, F,
                               pair_hash<size_t, size_t> > message_map_type;

    //! The underlying region graph.
    region_graph<domain_type, F> graph_;

    //! The function computing difference between the parameters of two factors.
    diff_fn<F> diff_;

    //! The pseudo message (does not account for counting numbers).
    message_map_type pseudo_message_;

    //! The message
    message_map_type message_;

  }; // class generalized_bp


  /**
   * GBP engine that updates the messages in a topological order.
   * according to the natural order of messages.
   *
   * \ingroup inference
   */
  template <typename F>
  class asynchronous_generalized_bp : public generalized_bp<F> {
    typedef generalized_bp<F> base;

  public:
    // bring in some types that are already in the base
    typedef typename F::real_type   real_type;
    typedef typename F::domain_type domain_type;

    asynchronous_generalized_bp(const region_graph<domain_type, F>& graph,
                                diff_fn<F> diff) 
      : base(graph, std::move(diff)) {
      auto out = std::back_inserter(order_);
      partial_order_traversal(graph, make_output_iterator_assign(out));
    }

    // Constructs the algorithm to the given graph structure and difference fn.
    asynchronous_generalized_bp(const region_graph<domain_type>& graph,
                                diff_fn<F> diff)
      : base(graph, std::move(diff)) {
      auto out = std::back_inserter(order_);
      partial_order_traversal(graph, make_output_iterator_assign(out));
    }      

    real_type iterate(real_type eta) override {
      // pass the messages downwards
      real_type residual(0);
      for (size_t v : order_) {
        for (size_t u : this->graph_.parents(v)) {
          residual = std::max(residual, this->pass_message(u, v, eta));
        }
      }
      // pass the messages upwards
      for (size_t v : make_reversed(order_)) {
        for (size_t u : this->graph_.parents(v)) {
          residual = std::max(residual, this->pass_message(v, u, eta));
        }
      }
      return residual;
    }
    
  private:
    //! A partial order over the graph vertices.
    std::vector<size_t> order_;

  }; // class asynchronous_generalized_bp

} // namespace libgm

#endif
