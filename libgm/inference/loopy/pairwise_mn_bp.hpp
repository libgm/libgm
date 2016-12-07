#ifndef LIBGM_PAIRWISE_MN_BP_HPP
#define LIBGM_PAIRWISE_MN_BP_HPP

#include <libgm/factor/utility/traits.hpp>
#include <libgm/datastructure/mutable_queue.hpp>
#include <libgm/factor/utility/diff_fn.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

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
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam NodeF
   *         The factor type associated with nodes of the Markov network
   * \tparam EdgeF
   *         The factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename Arg, typename NodeF, typename EdgeF = NodeF>
  class pairwise_mn_bp {
    static_assert(are_pairwise_compatible<NodeF, EdgeF>::value,
                  "The node and edge factors are not pairwise compatible");

  public:
    // Shortcuts
    using graph_type  = pairwise_markov_network<Arg, NodeF, EdgeF>;
    using real_type   = typename NodeF::real_type;
    using result_type = typename NodeF::result_type;
    using message_map =
      std::unordered_map<std::pair<Arg, Arg>, NodeF, pair_hash<Arg, Arg> >;

    // Constructors and initialization
    //--------------------------------------------------------------------------

    /**
     * Constructs a loopy bp engine for the given graph and difference function.
     */
    pairwise_mn_bp(const graph_type* graph, real_binary_fn<NodeF> diff)
      : graph_(graph), diff_(std::move(diff)), nupdates_(0) {
      reset();
    }

    //! Destructor.
    virtual ~pairwise_mn_bp() { }

    /**
     * Resets all the messages using the given generator or uniformly
     * if the generator is null.
     */
    virtual void reset(std::function<NodeF(Arg)> gen = nullptr) {
      for (vertex_type v : graph().vertices()) {
        for (edge_type e : graph().in_edges(v)) {
          if (gen) {
            message(e) = gen(v);
          } else {
            message(e) = NodeF(NodeF::shape(v), result_type(1));
          }
        }
      }
    }

    // Iteration and queries
    //--------------------------------------------------------------------------

    //! Performs a single iteration of BP.
    virtual real_type iterate(real_type eta) = 0;

    //! Returns the network that we perform inference over.
    const graph_type& graph() const { return *graph_; }

    //! The number of updates performed so far
    std::size_t num_updates() const { return nupdates_; }

    //! Computes the node belief.
    NodeF belief(Arg u) const {
      NodeF f = graph().factor(u);
      for (Arg v : graph().neighbors(u)) {
        f *= message(v, u);
      }
      f.normalize();
      return f;
    }

    //! Computes the edge belief.
    EdgeF belief(undirected_edge<Arg> e) const {
      Arg u = e.source();
      Arg v = e.target();
      NodeF fu = graph().factor(u);
      NodeF fv = graph().factor(v);
      for (Arg w : graph().neighbors(u)) {
        if (w != v) { fu *= message(w, u); }
      }
      for (Arg w : graph().neighbors(v)) {
        if (w != u) { fv *= message(w, v); }
      }
      EdgeF result = graph().factor(e) * outer_prod(fu, fv);
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
      for (edge_type e : graph().edges()) {
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
      for (edge_type e : graph().edges()) {
        result = std::max(result, residual(e));
        result = std::max(result, residual(e.reverse()));
      }
      return result;
    }

    // Implementation visible to base classes
    //--------------------------------------------------------------------------
  protected:

    //! Returns the difference between two node factors.
    real_type diff(const NodeF& f, const NodeF& g) const {
      return diff_(f, g);
    }

    //! Returns a writable message, default-initialized if not present.
    virtual NodeF& message(Arg from, Arg to) {
      return message_[std::make_pair(from, to)];
    }

    //! Returns a message. Throws std::out_of_range if not already present.
    const NodeF& message(Arg from, Arg to) const {
      return message_.at(std::make_pair(from, to));
    }

    //! Returns a message, default-initialized if not already present.
    virtual NodeF& message(undirected_edge<Arg> e) {
      return message_[e.pair()];
    }

    //! Returns a message. Throws std::out_of_range if not already present.
    const NodeF& message(undirected_edge<Arg> e) const {
      return message_.at(e.pair());
    }

    //! Computes the message along an edge.
    NodeF compute_message(undirected_edge<Arg> e) const {
      Arg u = e.source();
      Arg v = e.target();
      NodeF incoming = graph()[u];
      for (Arg w : graph().neighbors(u)) {
        if (w != v) { incoming *= message(w, u); }
      }
      EdgeF result;
      if (e.forward()) {
        result = (graph()[e].head() * incoming).sum();
      } else {
        result = (graph()[e].tail() * incoming).sum();
      }
      result.normalize();
      return result;
    }

    //! Updates the message along an edge and returns the residual.
    real_type update_message(undirected_edge<Arg> e,
                             real_type eta = real_type(1)) {
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
    virtual real_type residual(undirected_edge<Arg> e) const {
      return diff_(compute_message(e), message(e));
    }

    // Private data
    //--------------------------------------------------------------------------
  protected:
    //! A pointer to the Markov network used in the computations.
    const graph_type* graph_;

    //! The norm used to evaluate the change in messages.
    real_binary_fn<NodeF> diff_;

    //! A map that stores the messages.
    message_map message_;

    //! The total number of updates applied (possibly fewer than computed).
    std::size_t nupdates_;

  }; // class pairwise_mn_bp


  //============================================================================


  /**
   * Loopy BP engine that updates the messages synchronously.
   *
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam NodeF
   *         The factor type associated with nodes of the Markov network
   * \tparam EdgeF
   *         The factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename Arg, typename NodeF, typename EdgeF = NodeF>
  class synchronous_pairwise_mn_bp
    : public pairwise_mn_bp<NodeF, EdgeF> {
    using base = pairwise_mn_bp<Arg, NodeF, EdgeF>;

  public:
    // Shortcuts
    using graph_type  = pairwise_markov_network<Arg, NodeF, EdgeF>;
    using real_type   = typename NodeF::real_type;
    using result_type = typename NodeF::result_type;

    /**
     * Constructs a synchronous loopy bp engine for the given graph
     * and difference function.
     *
     * \param graph
     *        A markov network to run inference on. The vertices and edges may
     *        not change during the execution of the algorithm; however, the
     *        node/edge factors may.
     * \param diff
     *        A function for computing differences between two factors.
     */
    synchronous_pairwise_mn_bp(const graph_type* graph,
                               real_binary_fn<NodeF> diff)
      : base(graph, std::move(diff)) { }

    void reset(std::function<NodeF(Arg)> gen = nullptr) {
      base::reset(gen);
      base::swap(new_message_);
    }

    real_type iterate(real_type eta) override {
      real_type residual(0);
      for (undirected_edge<Arg> e : this->graph().edges()) {
        residual = std::max(residual, this->update_message(e, eta));
        residual = std::max(residual, this->update_message(e.reverse(), eta));
      }
      base::swap(new_message_);
      return residual;
    }

  protected:
    NodeF& message(Arg u, Arg v) override {
      return new_message_[std::make_pair(from, to)];
    }

    Node& message(undirected_edge<Arg> e) override {
      return new_message_[e.pair()];
    }

    //! The new messages.
    typename base::message_map new_message_;

  }; // class synchronous_pairwse_mn_bp


  //============================================================================


  /**
   * Loopy BP engine that updates the messages in a round-robin manner
   *
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam NodeF
   *         The factor type associated with nodes of the Markov network
   * \tparam EdgeF
   *         The factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename Arg, typename NodeF, typename EdgeF = NodeF>
  class asynchronous_pairwise_mn_bp
    : public pairwise_mn_bp<Arg, NodeF, EdgeF> {
    using base = pairwise_mn_bp<Arg, NodeF, EdgeF>;

  public:
    // Shortcuts
    using graph_type  = pairwise_markov_network<Arg, NodeF, EdgeF>;
    using real_type   = typename NodeF::real_type;
    using result_type = typename NodeF::result_type;

    /**
     * Constructs an asynchronous loopy bp engine for the given graph
     * and difference function.
     *
     * \param graph
     *        A markov network to run inference on. The vertices and edges may
     *        not change during the execution of the algorithm; however, the
     *        node/edge factors may.
     * \param diff
     *        A function for computing differences between two factors.
     */
    asynchronous_pairwise_mn_bp(const graph_type* graph,
                                real_binary_fn<NodeF> diff)
      : base(graph, std::move(diff)) { }

    real_type iterate(real_type eta) override {
      real_type residual(0);
      for (auto e : this->graph().edges()) {
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
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam NodeF
   *         The factor type associated with nodes of the Markov network
   * \tparam EdgeF
   *         The factor type associated with edges of the Markov network
   *
   * \ingroup inference
   */
  template <typename Arg, typename NodeF, typename EdgeF = NodeF>
  class residual_pairwise_mn_bp
    : public pairwise_mn_bp<Arg, NodeF, EdgeF> {
    using base = pairwise_mn_bp<NodeF, EdgeF>;

  public:
    // Shortcuts
    using graph_type  = pairwise_markov_network<Arg, NodeF, EdgeF>;
    using real_type   = typename NodeF::real_type;
    using result_type = typename NodeF::result_type;

    /**
     * Constructs a residual loopy bp engine for the given graph
     * and difference function.
     *
     * \param graph A markov network to run inference on. The vertices and
     *              edges may not change during the execution of the
     *              algorithm; however, the node/edge factors may.
     * \param diff A function for computing differences between two
     *             factors, such as sum_real_binary_fn<NodeF>().
     */
    residual_pairwise_mn_bp(const graph_type* graph, real_binary_fn<NodeF> diff)
      : base(graph, std::move(diff)) {
      initialize_residuals(); // base::reset() was already called
    }

    void reset(std::function<NodeF(Arg)> gen = nullptr) override {
      base::reset(gen);
      initialize_residuals();
    }

    real_type iterate(real_type eta) override {
      if (!q_.empty()) {
        // extract the leading candidate edge
        undirected_edge<Arg> e = q_.pop().first;

        // update message(e) and recompute residual for dependent messages
        real_type residual = this->update_message(e, eta);
        for (edge_type out : this->graph().out_edges(e.target())) {
          if (out.target() != e.source()) {
            update_residual(out);
          }
        }
        if (eta < real_type(1)) {
          update_residual(e);
        }
        return residual;
      } else {
        return real_type(0);
      }
    }

  protected:
    real_type residual(undirected_edge<Arg> e) const override {
      try {
        return q_.get(e));
      } catch (std::out_of_range&) {
        return real_type(0);
      }
    }

    //! Updates the residuals for an edge.
    void update_residual(undirected_edge<Arg> e) {
      double r = this->diff(this->message(e), this->compute_message(e));
      if (!q_.contains(e)) {
        q_.push(e, r);
      } else {
        q_.update(e, r);
      }
    }

    // Initializes the messages.
    void initialize_residuals() {
      // Pass the flow along each directed edge
      for (undirected_edge<Arg> e : this->graph().edges()) {
        this->update_message(e);
        this->update_message(e.reverse());
      }
      // Compute the residuals
      for (edge_type e : this->graph().edges()) {
        update_residual(e);
        update_residual(e.reverse());
      }
    }

    // A queue of residuals.
    mutable_queue<std::pair<Arg, Arg>, real_type> q_;
  };

} // namespace libgm

#endif
