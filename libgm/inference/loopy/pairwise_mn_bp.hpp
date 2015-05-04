#ifndef LIBGM_PAIRWISE_MN_BP_HPP
#define LIBGM_PAIRWISE_MN_BP_HPP

#include <libgm/datastructure/mutable_queue.hpp>
#include <libgm/factor/util/diff_fn.hpp>
#include <libgm/model/pairwise_markov_network.hpp>
#include <libgm/traits/pairwise_compatible.hpp>

#include <algorithm>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libgm {

  /**
   * An engine that performs loopy belief propagation.
   * If the underlying markov network changes, the results are undefined.
   * The lifetime of the Markov network object must extend past the lifetime
   * of this object.
   *
   * \tparam NodeF the factor type associated with nodes of the Markov network
   * \tparam EdgeF the factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename NodeF, typename EdgeF = NodeF>
  class pairwise_mn_bp {
    static_assert(pairwise_compatible<NodeF, EdgeF>::value,
                  "The node and edge factors are not pairwise compatible");

    // Public types
    //==========================================================================
  public:
    // FactorizedInference types
    typedef typename NodeF::real_type       real_type;
    typedef typename NodeF::result_type     result_type;
    typedef typename NodeF::variable_type   variable_type;
    typedef typename NodeF::domain_type     domain_type;
    typedef typename NodeF::assignment_type assignment_type;
    typedef pairwise_markov_network<NodeF, EdgeF> graph_type;

    // Vertex and edge types
    typedef typename graph_type::vertex_type vertex_type;
    typedef typename graph_type::edge_type edge_type;
    typedef std::pair<vertex_type, vertex_type> vertex_pair;

    // Constructors and initialization
    //==========================================================================
  public:

    /**
     * Constructs a loopy bp engine for the given graph and difference function.
     */
    pairwise_mn_bp(const graph_type* graph, diff_fn<NodeF> diff)
      : graph_(graph), diff_(std::move(diff)), nupdates_(0) {
      reset();
    }

    //! Destructor.
    virtual ~pairwise_mn_bp() { }

    /**
     * Resets all the messages using the given generator or uniformly
     * if the generator is null.
     */
    virtual void reset(std::function<NodeF(const domain_type&)> gen = nullptr) {
      for (vertex_type v : graph_->vertices()) {
        for (edge_type e : graph_->in_edges(v)) {
          if (gen) {
            message(e) = gen({v});
          } else {
            message(e) = NodeF({v}, result_type(1));
          }
        }
      }
    }

    // Iteration and queries
    //==========================================================================

    //! Performs a single iteration of BP.
    virtual real_type iterate(real_type eta) = 0;

    //! Returns the network that we perform inference over.
    const graph_type& graph() const { return *graph_; }

    //! The number of updates performed so far
    size_t num_updates() const { return nupdates_; }

    //! Computes the node belief.
    NodeF belief(vertex_type u) const {
      NodeF f = (*graph_)[u];
      for (vertex_type v : graph_->neighbors(u)) {
        f *= message(v, u);
      }
      return std::move(f.normalize());
    }

    //! Computes the edge belief.
    EdgeF belief(const edge_type& e) const {
      vertex_type u = e.source();
      vertex_type v = e.target();
      NodeF fu = (*graph_)[u];
      NodeF fv = (*graph_)[v];
      for (vertex_type w : graph_->neighbors(u)) {
        if (w != v) { fu *= message(w, u); }
      }
      for (vertex_type w : graph_->neighbors(v)) {
        if (w != u) { fv *= message(w, v); }
      }
      EdgeF result = (*graph_)[e];
      result *= fu;
      result *= fv;
      result.normalize();
      return result;
    }

    /**
     * Computes the expected residual raised to n + 1.
     * \param alpha if 0, amounts to the average residual.
     */
    real_type expected_residual(real_type n = real_type(0)) const {
      real_type numer(0);
      real_type denom(0);
      for (edge_type e : graph_->edges()) {
        real_type r = residual(e);
        numer += std::pow(r, n + 1); denom += std::pow(r, n);
        real_type s = residual(e.reverse());
        numer += std::pow(s, n + 1); denom += std::pow(s, n);
      }
      return numer / denom;
    }

    /**
     * Computes the maximum residual.
     */
    real_type maximum_residual() const {
      real_type result(0);
      for (edge_type e : graph_->edges()) {
        result = std::max(result, residual(e));
        result = std::max(result, residual(e.reverse()));
      }
      return result;
    }

    // Implementation
    //==========================================================================
  protected:

    //! Returns a message, default-initialized if not already present.
    NodeF& message(vertex_type from, vertex_type to) {
      return message_[vertex_pair(from, to)];
    }

    //! Returns a message. Throws std::out_of_range if not already present.
    const NodeF& message(vertex_type from, vertex_type to) const {
      return message_.at(vertex_pair(from, to));
    }

    //! Returns a message, default-initialized if not already present.
    NodeF& message(const edge_type& e) {
      return message_[e.pair()];
    }

    //! Returns a message. Throws std::out_of_range if not already present.
    const NodeF& message(const edge_type& e) const {
      return message_.at(e.pair());
    }

    //! Computes the message along an edge.
    NodeF compute_message(const edge_type& e) const {
      vertex_type u = e.source();
      vertex_type v = e.target();
      NodeF incoming = (*graph_)[u];
      for (vertex_type w : graph_->neighbors(u)) {
        if (w != v) { incoming *= message(w, u); }
      }
      return (incoming * (*graph_)[e]).marginal({v}).normalize();
    }

    //! Updates the message along an edge and returns the residual.
    real_type update_message(const edge_type& e, real_type eta = real_type(1)) {
      NodeF new_message = compute_message(e);
      real_type residual = diff_(message(e), new_message);
      if (eta == real_type(1)) {
        message(e) = std::move(new_message);
      } else {
        message(e) = weighted_update(message(e), new_message, eta);
      }
      ++nupdates_;
      return residual;
    }

    //! Returns the residual for the message along an edge.
    virtual real_type residual(const edge_type& e) const {
      return diff_(compute_message(e), message(e));
    }

    // Protected data members
    //==========================================================================
  protected:
    //! A map that stores the messages.
    typedef std::unordered_map<
      vertex_pair, NodeF, pair_hash<vertex_type, vertex_type>
    > message_map;

    //! A pointer to the Markov network used in the computations.
    const graph_type* graph_;

    //! The norm used to evaluate the change in messages.
    diff_fn<NodeF> diff_;

    //! A map that stores the messages.
    message_map message_;

    //! The total number of updates applied (possibly fewer than computed).
    size_t nupdates_;

  }; // class pairwise_mn_bp


  //============================================================================


  /**
   * Loopy BP engine that updates the messages synchronously.
   * 
   * \tparam NodeF the factor type associated with nodes of the Markov network
   * \tparam EdgeF the factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename NodeF, typename EdgeF = NodeF>
  class synchronous_pairwise_mn_bp : public pairwise_mn_bp<NodeF, EdgeF> {
    typedef pairwise_mn_bp<NodeF, EdgeF> base;
  public:
    // Bring some typedefs from the base class
    typedef pairwise_markov_network<NodeF, EdgeF> graph_type;
    typedef typename NodeF::real_type real_type;
    typedef typename graph_type::edge_type edge_type;

    /**
     * Constructs a synchronous loopy bp engine for the given graph
     * and difference function.
     *
     * \param graph A markov network to run inference on. The vertices and
     *              edges may not change during the execution of the
     *              algorithm; however, the node/edge factors may.
     * \param diff A function for computing differences between two
     *             factors, such as sum_diff_fn<NodeF>().
     */
    synchronous_pairwise_mn_bp(const graph_type* graph, diff_fn<NodeF> diff)
      : base(graph, std::move(diff)) { }

    real_type iterate(real_type eta) override {
      real_type residual(0);
      for (edge_type e : this->graph_->edges()) {
        residual = std::max(residual, update_message(e, eta));
        residual = std::max(residual, update_message(e.reverse(), eta));
      }
      swap(this->message_, new_message_);
      return residual;
    }

  protected:
    //! Updates the message along an edge and returns the residual.
    real_type update_message(const edge_type& e, real_type eta = real_type(1)) {
      NodeF new_message = this->compute_message(e);
      real_type residual = this->diff_(this->message(e), new_message);
      if (eta == real_type(1)) {
        new_message_[e.pair()] = std::move(new_message);
      } else {
        new_message_[e.pair()] = weighted_update(this->message(e), new_message, eta);
      }
      ++this->nupdates_;
      return residual;
    }

    //! The new messages.
    typename base::message_map new_message_;

  }; // class synchronous_pairwse_mn_bp


  //============================================================================


  /**
   * Loopy BP engine that updates the messages in a round-robin manner
   * 
   * \tparam NodeF the factor type associated with nodes of the Markov network
   * \tparam EdgeF the factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename NodeF, typename EdgeF = NodeF>
  class asynchronous_pairwise_mn_bp : public pairwise_mn_bp<NodeF, EdgeF> {
    typedef pairwise_mn_bp<NodeF, EdgeF> base;

  public:
    // Bring some typedefs from the base class
    typedef pairwise_markov_network<NodeF, EdgeF> graph_type;
    typedef typename NodeF::real_type real_type;

    /**
     * Constructs an asynchronous loopy bp engine for the given graph
     * and difference function.
     *
     * \param graph A markov network to run inference on. The vertices and
     *              edges may not change during the execution of the
     *              algorithm; however, the node/edge factors may.
     * \param diff A function for computing differences between two
     *             factors, such as sum_diff_fn<NodeF>().
     */
    asynchronous_pairwise_mn_bp(const graph_type* graph, diff_fn<NodeF> diff)
      : base(graph, std::move(diff)) { }

    real_type iterate(real_type eta) override {
      real_type residual(0);
      for (auto e : this->graph_->edges()) {
        residual = std::max(residual, this->update_message(e, eta));
        residual = std::max(residual, this->update_message(e.reverse(), eta));
      }
      return residual;
    }

  }; // class asynhcronous_pairwise_mn_bp


  //============================================================================


  /**
   * Loopy BP engine that updates the messages greedily based on the one
   * with the largest current residual.
   *
   * \tparam NodeF the factor type associated with nodes of the Markov network
   * \tparam EdgeF the factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename NodeF, typename EdgeF = NodeF>
  class residual_pairwise_mn_bp : public pairwise_mn_bp<NodeF, EdgeF> {
    typedef pairwise_mn_bp<NodeF, EdgeF> base;

  public:
    // Bring some typedefs from the base class
    typedef pairwise_markov_network<NodeF, EdgeF> graph_type;
    typedef typename NodeF::real_type real_type;
    typedef typename NodeF::domain_type domain_type;
    typedef typename graph_type::vertex_type vertex_type;
    typedef typename graph_type::edge_type edge_type;

  public:
    /**
     * Constructs a residual loopy bp engine for the given graph
     * and difference function.
     *
     * \param graph A markov network to run inference on. The vertices and
     *              edges may not change during the execution of the
     *              algorithm; however, the node/edge factors may.
     * \param diff A function for computing differences between two
     *             factors, such as sum_diff_fn<NodeF>().
     */
    residual_pairwise_mn_bp(const graph_type* graph, diff_fn<NodeF> diff)
      : base(graph, std::move(diff)) {
      initialize_residuals(); // base::reset() was already called
    }

    void reset(std::function<NodeF(const domain_type&)> gen = nullptr) override {
      base::reset(gen);
      initialize_residuals();
    }

    real_type iterate(real_type eta) override {
      if (!q_.empty()) {
        // extract the leading candidate edge
        vertex_type u, v;
        std::tie(u, v) = q_.pop().first;
        edge_type e = this->graph_->edge(u, v);

        // update message(e) and recompute residual for dependent messages
        real_type residual = this->update_message(e, eta);
        for (edge_type out : this->graph_->out_edges(v)) {
          if (out.target() != u) { update_residual(out); }
        }
        if (eta < real_type(1)) { update_residual(e); }
        return residual;
      } else {
        return real_type(0);
      }
    }

  protected:
    real_type residual(const edge_type& e) const override {
      try {
        return q_.get(e.pair());
      } catch(std::out_of_range& exc) {
        return real_type(0);
      }
    }

    //! Updates the residuals for an edge.
    void update_residual(const edge_type& e) {
      double r = this->diff_(this->message(e), this->compute_message(e));
      if (!q_.contains(e.pair())) {
        q_.push(e.pair(), r);
      } else {
        q_.update(e.pair(), r);
      }
    }

    // Initializes the messages.
    void initialize_residuals() {
      // Pass the flow along each directed edge
      for (edge_type e : this->graph_->edges()) {
        this->update_message(e);
        this->update_message(e.reverse());
      }
      // Compute the residuals
      for (edge_type e : this->graph_->edges()) {
        update_residual(e);
        update_residual(e.reverse());
      }
    }

    // A queue of residuals.
    mutable_queue<std::pair<vertex_type, vertex_type>, real_type> q_;
  };


  //============================================================================


  /**
   * Loopy BP engine that updates the messages according to a time delay
   * drawn from exponential random distribution with lambda = residual.
   * This algorithm can be viewed as an approximation of residual belief
   * propagation which is particularly useful in distributed systems where
   * it is infeasible to maintain a global queue.
   * 
   * \tparam NodeF the factor type associated with nodes of the Markov network
   * \tparam EdgeF the factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename NodeF, typename EdgeF = NodeF>
  class exponential_pairwise_mn_bp : public pairwise_mn_bp<NodeF, EdgeF> {
    typedef pairwise_mn_bp<NodeF, EdgeF> base;

  public:
    // Bring some typedefs from the base class
    typedef pairwise_markov_network<NodeF, EdgeF> graph_type;
    typedef typename NodeF::real_type real_type;
    typedef typename NodeF::domain_type domain_type;
    typedef typename graph_type::vertex_type vertex_type;
    typedef typename graph_type::edge_type edge_type;

    /**
     * Constructs a residual loopy bp engine for the given graph
     * and difference function.
     *
     * \param graph A markov network to run inference on. The vertices and
     *              edges may not change during the execution of the
     *              algorithm; however, the node/edge factors may.
     * \param diff A function for computing differences between two
     *             factors, such as sum_diff_fn<NodeF>().
     */
    exponential_pairwise_mn_bp(const graph_type* graph, diff_fn<NodeF> diff,
                               real_type exponent = real_type(1))
      : base(graph, std::move(diff)), exponent_(exponent), time_(0) {
      initialize_priorities();
    }

    void reset(std::function<NodeF(const domain_type&)> gen = nullptr) override {
      base::reset(gen);
      initialize_priorities();
    }

    real_type iterate(real_type eta) override {
      if (!q_.empty()) {
        // extract the leading candidate edge
        vertex_type u, v;
        std::tie(u, v) = q_.top().first;
        time_ = real_type(1) / q_.pop().second;
        edge_type e = this->graph_->edge(u, v);

        // update message(e) and update the priorities for dependent messages
        real_type residual = this->update_message(e, eta);
        for (edge_type out : this->graph_->out_edges(v)) {
          if (out.target() != u) { update_priority(out); }
        }
        if (eta < real_type(1)) { update_priority(e); }
        return residual;
      } else {
        return real_type(0);
      }
    }

  protected:
    //! Updates the priority for an edge.
    void update_priority(const edge_type& e) {
      real_type r = this->diff_(this->message(e), this->compute_message(e));
      if (r > real_type(0)) {
        std::exponential_distribution<real_type> exp(std::pow(r, exponent_));
        real_type t = time_ + exp(rng_);
        if (!q_.contains(e.pair())) {
          q_.push(e.pair(), real_type(1) / t);
        } else {
          q_.update(e.pair(), real_type(1) / t);
        }
      }
    }

    //! Initializes the priorities for all edges.
    void initialize_priorities() {
      for (edge_type e : this->graph_->edges()) {
        update_priority(e);
        update_priority(e.reverse());
      }
    }

    //! The exponent of the residual that affects how close we get to the max.
    real_type exponent_;

    //! The queue of with priority = 1 / time.
    mutable_queue<std::pair<vertex_type, vertex_type>, real_type> q_;

    //! The time of the latest updated message.
    real_type time_;

    //! A random number generator.
    std::mt19937 rng_;

  }; // class exponential_pairwise_mn_bp

} // namespace libgm

#endif
