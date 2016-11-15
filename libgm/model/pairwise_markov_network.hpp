#ifndef LIBGM_PAIRWISE_MARKOV_NETWORK_HPP
#define LIBGM_PAIRWISE_MARKOV_NETWORK_HPP

#include <libgm/enable_if.hpp>
#include <libgm/iterator/uint_vector_iterator.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/graph/property_fn.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/iterator/join_iterator.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/range/transformed.hpp>
#include <libgm/traits/is_range.hpp>

namespace libgm {

  // Forward declaration
  template <typename RealType> class probability_table;

  /**
   * Implements a Markov network with pairwise potentials.
   *
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam NodeF
   *         A type of factors associated with vertices.
   * \tparam EdgeF
   *         A type of factors associated with edges.
   *         Must be pairwise_compatible with NodeF.
   *
   * \ingroup model
   */
  template <typename Arg, typename NodeF, typename EdgeF = NodeF>
  class pairwise_markov_network
    : public undirected_graph<Arg, NodeF, EdgeF> {
    static_assert(are_pairwise_compatible<NodeF, EdgeF>::value,
                  "The node and edge factors are not pairwise compatible");

    using base = undirected_graph<typename NodeF::argument_type, NodeF, EdgeF>;

    /**
     * Indicates whether the model uses the same type for node and edge
     * potentials.
     */
    static const bool is_simple = std::is_same<NodeF, EdgeF>::value;

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
    using factor_type    = std::conditional_t<is_simple, NodeF, void>;
    using value_type     = std::conditional_t<
      is_simple, std::pair<const domain<Arg>, NodeF>, void>;
    class node_factor_iterator;
    class edge_factor_iterator;
    using iterator       = std::condtional_t<
      join_iterator<node_factor_iterator, edge_factor_iterator>, void>;

    // Shortcuts
    using vertex_type = typename base::vertex_type;
    using edge_type   = typename base::edge_type;

    // Constructors
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty pairwise Markov network.
    pairwise_markov_network() { }

    /**
     * Constructs a pairwise Markov network with the given vertices.
     */
    explicit pairwise_markov_network(const vector<Arg>& vertices) {
      for (Arg v : vertices) {
        this->add_vertex(v);
      }
    }

    /**
     * Constructs a pairwise Markov network with the given vertices.
     */
    template <typename It>
    explicit pairwise_markov_network(iterator_range<It> vertices) {
      for (Arg v : vertices) {
        this->add_vertex(v);
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

    //! Returns the arguments of the factor associated with a vertex.
    domain<Arg> arguments(vertex_type v) const {
      return { v };
    }

    //! Returns the arguments of the factor associated with an edge.
    domain<Arg> arguments(edge_type e) const {
      std::pair<Arg, Arg> unordered = e.unordered_pair();
      return { unordered.first, unordered.second };
    }

    //! Returns the number of factors (nodes + edges).
    std::size_t num_factors() const {
      return this->num_vertices() + this->num_edges();
    }

    //! Returns the factors associated with vertices.
    iterator_range<node_factor_iterator>
    node_factors() const {
      return { this->vertices(), vertex_property_fn<const base>(this) };
    }

    //! Returns the factors associated with the edges.
    iterator_range<edge_factor_iterator>
    edge_factors() const {
      return { this->edges(), edge_property_fn<const base>(this) };
    }

    /**
     * Returns iterator to the first factor in the model.
     * This function is only provided if the model is simple (NodeF=EdgeF).
     */
    LIBGM_ENABLE_IF(is_simple)
    iterator begin() const {
      auto nf = node_factors();
      auto ef = edge_factors();
      return { nf.begin(), nf.end(), ef.begin() };
    }

    /**
     * Returns iterator to past the last factor in the model.
     * This function is only provided if the model is simple (NodeF=EdgeF).
     */
    LIBGM_ENABLE_IF(is_simple)
    iterator end() const {
      auto nf = node_factors();
      auto ef = edge_factors();
      return { nf.end(), nf.end(), ef.end() };
    }

    // Queries
    //--------------------------------------------------------------------------

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
    void markov_graph(undirected_graph<Arg>& mg) const {
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
        if ((*this)[v].shape() != NodeF::param_shape(arguments(v))) {
          return false;
        }
      }
      for (edge_type e : this->edges()) {
        if ((*this)[e].shape() != EdgeF::param_shape(arguments(e))) {
          return false;
        }
      }
      return true;
    }

    // Mutators
    //--------------------------------------------------------------------------

    /**
     * Initializes the node and edge potentials with the given functors.
     * This is performed by invoking the functors on the arguments given
     * by each vertex and edge and assigning the result to the model.
     */
    void initialize(std::function<NodeF(vertex_type)> nodefn,
                    std::function<EdgeF(edge_type)> edgefn) {
      for (vertex_type v : this->vertices()) {
        (*this)[v] = nodefn(v);
      }
      for (edge_type e : this->edges()) {
        (*this)[e] = edgefn(e);
      }
    }

    /**
     * Adds a factor to the graphical model and creates the vertices and edges.
     * If the corresponding vertex/edge already exists, does nothing.
     * This function is only provided if the model is simple (NodeF=EdgeF).
     *
     * \throw std::invalid_argument if the factor is not unary or binary
     */
    LIBGM_ENABLE_IF(is_simple)
    bool add_factor(const domain<Arg>& args, const NodeF& factor) {
      switch (args.size()) {
      case 1: {
        bool added = this->add_vertex(args.front());
        (*this)[args.front()] = factor;
        return added;
      }
      case 2: {
        auto added = this->add_edge(args.front(), args.back());
        (*this)[added.first] = factor;
        return added.second;
      }
      default:
        throw std::invalid_argument("Unsupported factor arity " +
                                    std::to_string(args.size()));
      }
    }

    /**
     * Adds an n-ary factor to the graphical model (with n >= 1), unrolling the
     * factor and introducing determininistic pairwise potentials as needed.
     * Only supported when NodeF and EdgeF are both probability_table factors.
     * \param factor the factor to be added
     * \param a factory returning a new argument with the given arity
     * \return the newly created argument or null vertex if the factor
     *         was added as is
     */
    LIBGM_ENABLE_IF((std::is_same<probability_table<real_type>, NodeF>::value &&
                     std::is_same<probability_table<real_type>, EdgeF>::value))
    Arg add_nary(const probability_table<real_type>& f,
                 const std::function<Arg(std::size_t)>& arg_gen) {
      if (f.arity() <= 2) {
        add_factor(f);
        return base::null_vertex();
      } else {
        // create a factor with all the arguments collapsed
        Arg new_arg = arg_gen(f.size());
        add_factor(f.reshape({f.size()}));

        // initialize indicator potentials linking the arguments of f to new_arg
        std::vector<probability_table<real_type> > potentials;
        for (Arg arg : f.arguments()) {
          uint_vector shape = num_dims(arg);
          potentials.emplace_back(shape, real_type(0));
        }

        // populate the potentials
        std::size_t index = 0;
        for (const uint_vector& values : uint_vectors(f.param().shape())) {
          for (std::size_t i = 0; i < f.arity(); ++i) {
            potentials[i][index + values[i] * f.size()] = real_type(1);
          }
          ++index;
        }

        // add the potentials
        for (std::size_t i = 0; i < potentials.size(); ++i) {
          add_factor(potentials[i]); // FIXME
        }

        return new_arg;
      }
    }

    /**
     * Conditions the model on an assignment. This restricts any edge
     * factor whose argument is contained in a and multiplies it to
     * the adjacent node factor. The normalizing constant is not preserved.
     */
    void condition(const assignment_type& a) { // FIXME
      for (const auto& p : a) {
        Arg u = p.first;
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
