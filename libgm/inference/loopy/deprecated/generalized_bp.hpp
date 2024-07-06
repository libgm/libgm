#ifndef LIBGM_GENERALIZED_BP_HPP
#define LIBGM_GENERALIZED_BP_HPP

#include <libgm/factor/invalid_operation.hpp>
#include <libgm/factor/utility/diff_fn.hpp>
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
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam F
   *         A type representing the factors. The type must support
   *         multiplication and marginalization operations.
   *
   * \ingroup inference
   */
  template <typename Arg, typename F>
  class generalized_bp {

    // Public types
    //--------------------------------------------------------------------------
  public:
    // Factor types
    using real_type   = typename F::real_type;
    using result_type = typename F::result_type;


    typedef typename std::pair<Arg, Arg> vertex_pair;

    // Constructors and initialization
    //==========================================================================
  public:
    //! Constructs the algorithm to the given graph and difference fn.
    generalized_bp(const region_graph<Arg, F>& graph, real_binary_fn<F> diff)
      : graph_(graph), diff_(std::move(diff)) {
      initialize_messages();
    }

    // Constructs the algorithm to the given graph structure and difference fn.
    generalized_bp(const region_graph<Arg>& graph, diff_fn<F> diff)
      : diff_(std::move(diff)) {
      graph_.structure_from(graph);
      initialize_messages();
      initialize_factors();
    }

    //! Destructor.
    virtual ~generalized_bp() { }

    //! Initializes the messages to unity.
    void initialize_messages() {
      for (directed_edge<Arg> e : graph_.edges()) {
        id_t u = e.source();
        id_t v = e.target();
        auto shape = F::shape(graph_.separator(e));
        pseudo_message(u, v) = F(shape, result_type(1));
        pseudo_message(v, u) = F(shape, result_type(1));
        message(u, v) = F(shape, result_type(1));
        message(v, u) = F(shape, result_type(1));
      }
    }

    //! Initializes the factors to unity.
    void initialize_factors() {
      for (id_t v : graph_.vertices()) {
        graph_[v] = F(F::shape(graph_.cluster(v)), result_type(1));
      }
    }

    //! Initializes the factors to those in a range.
    template <typename Range>
    void initialize_factors(const Range& factors) {
      initialize_factors();
      for (const F& factor : factors) {
        id_t v = graph_.find_root_cover(factor.first);
        assert(v);
        graph_[v].dims(graph_.index(v, factor.first)) *= factor.second;
      }
    }

    // Running the algorithm
    //--------------------------------------------------------------------------

    //! Performs one iteration
    virtual real_type iterate(real_type eta) = 0;

    //! Returns the underlying region graph.
    const cluster_graph<Arg, F>& graph() const {
      return graph_;
    }

    //! Returns the belief for a region.
    F belief(id_t v) const {
      F result = pow(graph_[v], graph_.counting(v));
      for (id_t u : graph_.parents(v)) {
        result *= message(u, v); // indexing??
      }
      for (id_t u : graph_.children(v)) {
        result *= message(u, v); // indexing??
      }
      return result.normalize();
    }

    //! Returns the marginal over a set of variables
    F belief(const domain<Arg>& args) const {
      id_t v = graph_.find_cover(args);
      assert(v);
      return belief(v).marginal(graph_.index(v, args));
    }

    // Implementation
    //--------------------------------------------------------------------------
  protected:
    //! Returns a pseudo-message from region u to region v
    F& pseudo_message(id_t u, id_t v) {
      return pseudo_message_[std::make_pair(u, v)];
    }

    //! Returns the message from region u to region v
    F& message(id_t u, id_t v) {
      return message_[std::make_pair(u, v)];
    }

    //! Returns the message from region u to region v
    const F& message(id_t u, id_t v) const {
      return message_.at(std::make_pair(u, v));
    }

    /**
     * Passes a message from region u to region v.
     * u and v must be adjacent.
     */
    real_type pass_message(id_t u, id_t v, real_type eta) {
      bool down = graph_.contains(u, v); /* true means the edge is u->v */
      real_type br = down ? beta(v) : beta(u);
      const domain_type& sep = //?
        down ? graph_.separator(u, v) : graph_.separator(v, u);

      // compute the pseudo message (this is m0 in the Yedidia paper)
      F m0 = pow(graph_[u], graph_.counting(u));
      for (id_t w : graph_.parents(u)) {
        if (w != v) m0 *= message(w, u); //?
      }
      for (id_t w : graph_.children(u)) {
        if (w != v) m0 *= message(w, u); //?
      }
      try {
        pseudo_message(u, v) = m0.marginal(sep).normalize();
      } catch (std::invalid_argument&) {
        std::ostringstream out;
        out << "Encountered invalid argument" << std::endl;
        out << "c_r = " << graph_.counting(u) << std::endl;
        out << "beta = " << br << std::endl;
        out << m0 << std::endl;
        out << pow(graph_[u], graph_.counting(u)) << std::endl;
        for (id_t w : graph_.parents(u)) {
          if (w != v) out << message(w, u);
        }
        for (id_t w : graph_.children(u)) {
          if (w != v) out << message(w, u);
        }
        throw std::runtime_error(out.str());
      } catch (invalid_operation&) {
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
    real_type beta(id_t v) const {
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
    typedef std::unordered_map<
      vertex_pair, F, pair_hash<vertex_type, vertex_type>
    > message_map_type;

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
      for (id_t v : order_) {
        for (id_t u : this->graph_.parents(v)) {
          residual = std::max(residual, this->pass_message(u, v, eta));
        }
      }
      // pass the messages upwards
      for (id_t v : make_reversed(order_)) {
        for (id_t u : this->graph_.parents(v)) {
          residual = std::max(residual, this->pass_message(v, u, eta));
        }
      }
      return residual;
    }

  private:
    //! A partial order over the graph vertices.
    std::vector<id_t> order_;

  }; // class asynchronous_generalized_bp

} // namespace libgm

#endif
