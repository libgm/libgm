#ifndef LIBGM_DYNAMIC_BAYESIAN_NETWORK_HPP
#define LIBGM_DYNAMIC_BAYESIAN_NETWORK_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/argument/indexed.hpp>
#include <libgm/graph/bayesian_network.hpp>

#include <iosfwd>
#include <unordered_set>

namespace libgm {

  /**
   * A dynamic Bayesian network.
   *
   * \tparam Arg
   *         An indexable argument that represents the discrete process.
   * \tparam VertexProperty
   *         A type that represents the properties stored at each vertex.
   *
   * \ingroup model
   */
  template <typename Arg, typename VertexProperty>
  class dynamic_bayesian_network {

    // Public type
    //--------------------------------------------------------------------------
  public:
    // Underlying graph type
    using graph_type =
      bayesian_network<indexed<Arg, std::size_t>, VertexProperty>;

    // Vertex types, edge type, and properties
    using argument_type   = Arg;
    using vertex_type     = indexed<Arg>;
    using edge_type       = directed_edge<vertex_type>;
    using vertex_property = VertexProperty;

    // Iterators
    using argument_iterator = typename unordered_set<Arg>::const_iterator;
    using vertex_iterator   = typename graph_type::vertex_iterator;
    using parent_iterator   = typename graph_type::parent_iterator;
    using child_iterator    = typename graph_type::child_iterator;
    using in_edge_iterator  = typename graph_type::in_edge_iterator;
    using out_edge_iterator = typename graph_type::out_edge_iterator;
    using edge_iterator     = typename graph_type::edge_iterator;

  public:

    // Constructors and conversion operators
    // =========================================================================
  public:
    //! Creates an empty DBN with a Markov network prior
    dynamic_bayesian_network()  { }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the prior model
    const graph_type& prior() const {
      return prior_;
    }

    //! Returns the transition model
    const graph_type& cpd() const {
      return cpd_;
    }

    //! Returns the arguments of this DBN
    iterator_range<argument_iterator>
    arguments() const {
      return { arguments_.begin(), arguments_.end() };
    }

    //! Returns the range of vertices in the given time slice.
    iterator_range<vertex_iterator>
    vertices(std::size_t t) const {
      return { { arguments_.begin(), t }, { arguments_.end(), t } };
    }

    //! Returns all edges leading into the given time slice.
    iterator_range<edge_iterator>
    edges(std::size_t t) const {
      /// TODO
    }

    //! Returns the parents of u.
    iterator_range<parent_iterator>
    parents(indexed<Arg> u) const {
      std::size_t t = u.index();
      if (t == 0) {
        return make_offset_range(prior_.parents(u), 0);
      } else {
        return make_offset_range(cpd_.parents(u.with_index(1), t - 1));
      }
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(indexed<Arg> u) const {
      iterator_range<parent_iterator> range = parents(u);
      return { { range.begin(), u }, { range.end(), u } };
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(indexed<Arg> u) const {
      if (u.index() == 0) {
        return prior_.contains(u);
      } else {
        return cpd_.contains(u.with_index(1));
      }
    }

    //! Returns true if the graph contains a directed edge (u, v).
    bool contains(indexed<Arg> u, indexed<Arg> v) const {
      if (u.index() != v.index() && u.index() + 1 != v.index()) {
        return false;
      } else if (u.index() == 0 && v.index() == 0) {
        return prior_.contains(u, v);
      } else {
        std::size_t t = std::min(u.index(), v.index());
        return cpd_.contains(u.offset(-t), v.offset(-t));
      }
    }

    //! Returns true if the graph contains a directed edge.
    bool contains(const directed_edge<indexed<Arg> > e) const {
      return contains(e.source(), e.target());
    }

    //! Returns the number of incoming edges to a vertex.
    std::size_t in_degree(indexed<Arg> u) const {
      if (u.index() == 0) {
        return prior_.in_degree(u);
      } else {
        return cpd_.in_degree(u.with_index(1));
      }
    }

    //! Returns the number of arguments (processes).
    std::size_t num_arguments() const {
      return arguments_.size();
    }

    //! Returns the number of vertices in each slice.
    std::size_t num_vertices() const {
      return arguments_.size();
    }

    //! Returns the property associated with a vertex.
    const VertexProperty& operator[](indexed<Arg> u) const {
      if (u.index() == 0) {
        return prior_[u];
      } else {
        return cpd_.[u.with_index(1)];
      }
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator[](indexed<Arg> u) {
      if (u.index() == 0) {
        return prior_[u];
      } else {
        return cpd_.[u.with_index(1)];
      }
    }

    //! Returns the arguments of a factor.
    domain<Arg> arguments(indexed<Arg> u) const {
      std::size_t t = u.index();
      if (t == 0) {
        return prior_.arguments(u);
      } else {
        return cpd_.arguments(u.with_index(1)).offset(t - 1);
      }
    }

    //! Compares two dynamic Bayesian networks.
    bool operator==(const dynamic_bayesian_network& other) const {
      return prior_ == other.prior_ && cpd_ == other.cpd_;
    }

    //! Compares two dynamic Bayesian networks.
    bool operator!=(const dynamic_bayesian_network& other) const {
      return prior_ != other.prior_ || cpd_ != other.cpd_;
    }

    //! Prints the dynamic Bayesian network to an output stream.
    std::ostream&
    operator<<(std::ostream& out, const dynamic_bayesian_network& dbn) {
      out << "#DBN(" << dbn.arguments() << ")" << std::endl;
      out << "Prior:" << std::endl;
      out << dbn.prior();
      out << "Transition model:" << std::endl;
      out << dbn.cpd() << std::endl;
      return out;
    }

    // Queries
    //--------------------------------------------------------------------------

    /**
     * Returns the ancestors of the t+1-time arguments in the transition model.
     * \param procs The set of processes whose ancestors are being sought
     */
    domain_type ancestors(const std::set<process_type>& procs) const {
      return transition.ancestors(arguments(procs, next_step));
    }

    // Modifiers
    //--------------------------------------------------------------------------

    /**
     * Adds a new vertex to the prior model.
     */
    void add_prior(const domain<indexed<Arg> >& args,
                   const VertexProperty& p) {
      assert(!args.empty());
      for (indexed<Arg> arg : args) {
        assert(arg.index() == 0);
        arguments_.insert(arg.process());
      }
      prior_.add_vertex(args, p);
    }

    /**
     * Adds a new vertex to the transition model.
     */
    void add_transition(const domain<indexed<Arg> >& args,
                        const VertexProperty& p) {
      assert(!args.empty() && args.front().index() == 1);
      for (indexed<Arg> arg : args) {
        assert(arg.index() <= 1);
        arguments_.insert(arg.process());
      }
      cpd_.add_vertex(args, p);
    }

    /**
     * Removes all processes and factors from this DBN.
     */
    void clear() {
      prior_.clear();
      cpd_.clear();
      arguments_.clear();
    }

    // Private data members
    //--------------------------------------------------------------------------
  private:
    //! The arguments of this DBN
    std::unordered_set<Arg> arguments_;

    //! The prior model
    graph_type prior_;

    //! The transition model
    graph_type cpd_;

  }; // class dynamic_bayesian_network

} // namespace libgm

#endif
