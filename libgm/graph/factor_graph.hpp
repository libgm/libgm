#ifndef LIBGM_FACTOR_GRAPH_HPP
#define LIBGM_FACTOR_GRAPH_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/annotated.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/graph/bipartite_edge.hpp>
#include <libgm/graph/util/sampling.hpp>
#include <libgm/graph/util/void.hpp>
#include <libgm/iteraotr/join_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/iterator/map_value_property_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <algorithm>
#include <iterator>
#include <map>
#include <unordered_map>
#include <vector>

namespace libgm {

  /**
   * A class that represents a factor graph. A factor graph is a special kind of
   * bipartite graph, consisting of two types of vertices: argument (type-1)
   * vertices and factor (type-2) vertices. Both types of vertices can be
   * associated with a property, and each factor is associated with a domain,
   * which implicitly defines its edges to the argument vertices.
   * There is no data associated with edges.
   *
   * \tparam Arg
   *         The type that represents an argument.
   * \tparam ArgumentProperty
   *         The property associated with each argument.
   *         Must be DefaultConstructible and CopyConstructible.
   * \tparam FactorProperty
   *         The property associated with each factor.
   *         Must be DefaultConstructible and CopyConstructible.
   * \ingroup graph_types
   */
  template <typename Arg,
            typename ArgumentProperty = void_,
            typename FactorProperty = void_>
  class factor_graph {
    // Private types
    //--------------------------------------------------------------------------
  private:
    struct vertex1_data {
      annotated<Arg, ArgumentProperty> property;
      std::vector<id_t> neighbors; // sorted
      bool operator==(const vertex1_data& other) const {
        return property == other.property; // intentionally omitting neihgbors
      }
      bool contains(id_t v) const {
        return std::binary_search(neighbors.begin(), neighbors.end(), v);
      }
      void erase(id_t v) {
        auto it = std::lower_bound(neighbors.begin(), neighbors.end(), v);
        assert(it != neighbors.end() && *it == v);
        neighbors.erase(it);
      }
    };

    struct vertex2_data {
      annotated<Arg, FactorProperty> property;
      bool operator==(const vertex2_data& other) const {
        return property == other.property;
      }
    };

    using vertex1_data_map = std::unordered_map<Arg, vertex1_data>;
    using vertex2_data_map = std::unordered_map<id_t, vertex2_data>;

    // Public types
    //--------------------------------------------------------------------------
  public:
    // Vertex types, edge type, and properties
    using argument_type    = Arg;
    using vertex1_type     = Arg;
    using vertex2_type     = id_t;
    using edge_type        = bipartite_edge<Arg, id_t>;
    using vertex1_property = Property;
    using vertex2_property = Property;
    using edge_property    = void;

    // Iterators
    using argument_iterator  = map_key_iterator<vertex1_data_map>;
    using vertex1_iterator   = map_key_iterator<vertex1_data_map>;
    using vertex2_iterator   = map_key_iterator<vertex2_data_map>;
    using neighbor1_iterator = typename std::vector<id_t>::const_iterator;
    using neighbor2_iterator = typename domain<Arg>::const_iterator;
    using in1_edge_iterator  =
      bind2_iterator<neighbor1_iterator, edge_type, Arg>;
    using in2_edge_iterator  =
      bind2_iterator<neighbor2_iterator, edge_type, id_t>;
    using out1_edge_iterator =
      bind1_iterator<neighbor1_iterator, edge_type, Arg>;
    using out2_edge_iterator =
      bind1_iterator<neighbor2_iterator, edge_type, id_t>;
    using edge_iterator =
      map_range_iterator<vertex1_data_map, std::vector<id_t>, edge_type>;
    using iterator = join_iterator<
      map_value_property_iterator<vertex1_data_map>,
      map_value_property_iterator<vertex2_data_map> >;

    // Constructors, destructors, and related functions
    //--------------------------------------------------------------------------
  public:
    //! Creates an empty graph.
    factor_graph()
      : num_edges_(0) { }

    //! Swap with another graph in constant time
    friend void swap(factor_graph& a, factor_graph& b) {
      swap(a.data1_, b.data1_);
      swap(a.data2_, b.data2_);
      std::swap(a.num_edges_, b.num_edges_);
    }

    // Accessors
    //--------------------------------------------------------------------------
  public:
    //! Returns the range of all arguments (same a vertices1()).
    iterator_range<argument_iterator>
    arguments() const {
      return { data1_.begin(), data1_.end() };
    }

    //! Returns the range of all factor IDs (same as vertices2()).
    iterator_range<vertex2_iterator>
    factors() const {
      return { data2_.begin(), data2_.end() };
    }

    //! Returns the range of all argument (type-1) vertices.
    iterator_range<vertex1_iterator>
    vertices1() const {
      return { data1_.begin(), data1_.end() };
    }

    //! Returns the range of all factor (type-2) vertices.
    iterator_range<vertex2_iterator>
    vertices2() const {
      return { data2_.begin(), data2_.end() };
    }

    //! Returns all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return { { data1_.begin(), data1_.end(), &vertex_data1_map::neighbors },
               { data1_.end(), data1_.end(), &vertex_data1_map::neighbors } };
    }

    //! Returns the IDs of factors containing an argument.
    iterator_range<neighbor1_iterator>
    neighbors(Arg u) const {
      const vertex1_data& data = data1_.at(u);
      return { data.neighbors.begin(), data.neighbors.end() };
    }

    //! Returns the range of arguments of a factor.
    iterator_range<neighbor2_iterator>
    neighbors(id_t u) const {
      const domain<Arg>& args = data2_.at(u).property.domain;
      return { args.begin(), args.end() };
    }

    //! Returns the edges incoming to an argument.
    iterator_range<in1_edge_iterator>
    in_edges(Arg u) const {
      iterator_range<neighbor1_iterator> range = neighbors(u);
      return { { range.begin(), u }, { range.end(), u } };
    }

    //! Returns the edges incoming to a factor.
    iterator_range<in2_edge_iterator>
    in_edges(id_t u) const {
      iterator_range<neighbor2_iterator> range = neighbors(u);
      return { { range.begin(), u }, { range.end(), u } };
    }

    //! Returns the edges outgoing from an argument.
    iterator_range<out1_edge_iterator>
    out_edges(Arg u) const {
      iterator_range<neighbor1_iterator> range = neighbors(u);
      return { { range.begin(), u }, { range.end(), u } };
    }

    //! Returns the edges outgoing from a factor.
    iterator_range<out2_edge_iterator>
    out_edges(id_t u) const {
      iterator_range<neighbor2_iterator> range = neighbors(u);
      return { { range.begin(), u }, { range.end(), u } };
    }

    //! Returns true if the graph contains the given argument.
    bool contains(Arg u) const {
      return data1_.find(u) != data1_.end();
    }

    //! Returns true if the graph contains the given factor ID.
    bool contains(id_t u) const {
      return data2_.find(u) != data2_.end();
    }

    //! Returns true if the graph contains an edge {u, v}.
    bool contains(Arg u, id_t v) const {
      auto it = data1_.find(u);
      return it != data1_.end() && it->second.contains(v);
    }

    //! Returns true if the graph contains an undirected edge.
    bool contains(bipartite_edge<Arg, id_t> e) const {
      return contains(e.v1(), e.v2());
    }

    //! Returns an undirected edge between u and v, assuming one exists.
    bipartite_edge<Arg, id_t> edge(Arg u, id_t v) const {
      return { u, v };
    }

    //! Returns an undirected edge between u and v, assuming one exists.
    bipartite_Edge<id_t, Arg> edge(id_t u, Arg v) const {
      return { u, v };
    }

    //! Returns the number of factors containing the given argument.
    std::size_t degree(Arg u) const {
      return data1_.at(u).neighbors.size();
    }

    //! Returns the number of arguments in the given factor.
    std::size_t degree(id_t u) const {
      return data2_.at(u).property.domain.size();
    }

    //! Returns true if the graph has no vertices.
    bool empty() const {
      return data1_.empty() && data2_.empty();
    }

    //! Returns the number of arguments (same as num_vertices1()).
    std::size_t num_arguments() const {
      return data1_.size();
    }

    //! Returns the number of factors (same as num_vertices2()).
    std::size_t num_factors() const {
      return data2_.size();
    }

    //! Returns the number of type-1 vertices (arguments).
    std::size_t num_vertices1() const {
      return data1_.size();
    }

    //! Returns the number of type-2 vertices (factors).
    std::size_t num_vertices2() const {
      return data2_.size();
    }

    //! Returns the total number of vertices.
    std::size_t num_vertices() const {
      return data1_.size() + data2_.size();
    }

    //! Returns the total number of edges.
    std::size_t num_edges() const {
      return num_edges_;
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u).
    bipartite_edge<Arg, id_t> reverse(bipartite_edge<Arg, id_t> e) const {
      return e.reverse();
    }

    //! Returns the property associated with an argument.
    const ArgumentProperty& operator[](Arg u) const {
      return data1_.at(u).property.object;
    }

    //! Returns the property associated with a factor.
    const FactorProperty& operator[](id_t u) const {
      return data2_.at(u).property.object;
    }

    //! Returns the property associated with an argument.
    ArgumentProperty& operator[](Arg u) {
      return data1_.at(u).property.object;
    }

    //! Returns the property associated with a factor.
    FactorProperty& operator[](id_t u) {
      return data2_.at(u).property.object;
    }

    //! Returns the arguments of a factor.
    const domain<Arg>& arguments(id_t u) const {
      return data2_.at(u).property.domain;
    }

    //! Returns the annotated property associated with an argument.
    const annotated<Arg, ArgumentProperty>& property(Arg u) const {
      return data1_.at(u).property;
    }

    //! Returns the annotated property associated with a factor.
    const annotated<Arg, FactorProperty>& property(id_t u) const {
      return data2_.at(u).property;
    }

    //! Returns the begin iterator over the range of all properties.
    LIBGM_ENABLE_IF((std::is_same<ArgumentProperty, FactorProperty>::value))
    iterator begin() const {
      return { data1_.begin(), data1_.end(), data2_.begin() };
    }

    //! Returns the end iterator over the range of all properties.
    LIBGM_ENABLE_IF((std::is_same<ArgumentProperty, FactorProperty>::value))
    iterator end() const {
      return { data1_.end(), data1_.end(), data2_.end() };
    }

    /**
     * Draws a random argument vertex in the graph, assuming one exists.
     * \todo ensure sampling is uniform
     */
    template <typename Generator>
    Arg sample_argument(Generator& rng) const {
      return sample_key(data1_, rng); // in sampling.hpp
    }

    /**
     * Draws a random factor vertex in the graph, assuming one exists.
     * \todo ensure sampling is uniform
     */
    template <typename Generator>
    id_t sample_factor(Generator& rng) const {
      return sample_key(data2_, rng); // in sampling.hpp
    }

    /**
     * Returns a cover for a domain, other the vertex given,
     * or null if there is none.
     */
    id_t find_cover(const domain<Arg>& dom, id_t exclude = id_t()) const {
      assert(!dom.empty());

      // identify the argument with the fewest connected factors
      std::size_t min_degree = std::numeric_limits<std::size_t>::max();
      Arg min_arg;
      for (Arg arg : dom) {
        if (degree(arg) < min_degree) {
          min_degree = degree(arg);
          min_arg = arg;
        }
      }

      // identify a subsuming factor among the neighbors of min_arg
      for (id_t u : neighbors(min_arg)) {
        if (u != exclude && subset(dom, arguments(u))) {
          return u;
        }
      }
      return id_t();
    }

    /**
     * Compares the graph structure and the vertex & edge properties.
     * The property types must support operator!=().
     */
    bool operator==(const factor_graph& g) const {
      return
        num_vertices1() == g.num_vertices1() &&
        num_vertices2() == g.num_vertices2() &&
        num_edges() == g.num_edges() &&
        data1_ == g.data1_ &&
        data2_ == g.data2_;
    }

    //! Compares the graph structure and the vertex & edge properties
    bool operator!=(const factor_graph& g) const {
      return !(*this == g);
    }

    //! Prints the graph to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const factor_graph& g) {
      out << "Arguments" << std::endl;
      for (Arg u : g.arguments()) {
        out << u << ": " << g[u] << std::endl;
      }
      out << "Factors" << std::endl;
      for (id_t u : g.factors()) {
        out << u << ": " << g.property(u) << std::endl;
      }
      out << "Edges" << std::endl;
      for (edge_type e : g.edges()) {
        out << e << std::endl;
      }
      return out;
    }

    // Modifications
    //--------------------------------------------------------------------------
    /**
     * Adds an argument to a graph with the associated property,
     * overwriting the existing property.
     *
     * \return true if the insertion took place (i.e., vertex was not present).
     */
    bool add_argument(Arg u, const ArgumentProperty& p = ArgumentProperty()) {
      assert(u != Arg());
      auto it = data1_.find(u);
      if (it == data1_.end()) {
        data1_.emplace(arg, make_annotated({u}, p));
        return true;
      } else {
        it->second.property.domain = {u};
        it->second.property.object = p;
        return false;
      }
    }

    /**
     * Adds a new factor vertex with given domain to a graph and associates
     * a property with that that vertex.
     *
     * \return the id of thew newly added factor
     */
    id_t add_factor(const domain<Arg>& args,
                    const FactorProperty& p = FactorProperty()) {
      data2_[next_id_].property = make_annotated(args, p);
      for (Arg arg : args) {
        data1_[arg].neighbors.push_back(next_id_);
      }
      num_edges_ += args.size();
      return next_id_++;
    }

    /**
     * Removes an argument from the graph.
     * Requires that there are no factors adjacent to this argument.
     */
    void remove_vertex(Arg u) {
      assert(data1_.at(u).neighbors.empty());
      data1_.erase(u);
    }

    /**
     * Removes a factor from the graph and all the edges attached to it.
     */
    void remove_vertex(id_t u) {
      const domain<Arg>& args = arguments(u);
      num_edges_ -= args.size();
      for (Arg arg : args) {
        data1_.at(arg).erase(u);
      }
      data2_.erase(u);
    }

    /**
     * Simplify the model by merging factors. For each factor f(X),
     * if a factor g(Y) exists such that X \subseteq Y, the factor
     * g is merged into f using the specified function, taking the
     * source and target factor.
     */
    void simplify(std::function<void(id_t, id_t)> merge) {
      // create a copy because we will be mutating the factors
      std::vector<id_t> ids(factors().begin(), factors.end());
      for (id_t u : ids) {
        id_t v = find_cover(arguments(u), u);
        if (v) {
          merge(u, v);
          remove_vertex(u);
        }
      }
    }

    //! Removes all vertices and edges from the graph.
    void clear() {
      data1_.clear();
      data2_.clear();
      num_edges_ = 0;
    }

    //! Saves the graph to an archive.
    void save(oarchive& ar) const {
      ar << num_vertices1() << num_vertices2() << num_edges();
      for (const auto& p : data1_) {
        ar << p.first << p.second.property;
      }
      for (const auto& p : data2_) {
        ar << p.first << p.second.property;
      }
    }

    //! Loads the graph from an archive.
    void load(iarchive& ar) {
      clear();
      std::size_t num_vertices1, num_vertices2;
      Arg u;
      id_t v;
      ar >> num_vertices1 >> num_vertices2 >> num_edges_;
      while (num_vertices1-- > 0) {
        ar >> u;
        ar >> data1_[u].property;
      }
      while (num_vertices2-- > 0) {
        ar >> v;
        ar >> data2_[v].property;
        for (Arg arg : data2_[v].property.domain) {
          data1_[arg].neighbors.push_back(v);
        }
      }
    }

    // Private members
    //--------------------------------------------------------------------------
  private:
    //! The properties and neighbors of type-1 vertices (arguments).
    vertex1_data_map data1_;

    //! The properties and neighbors of type-2 vertices (factors).
    vertex2_data_map data2_;

    //! The number of edges.
    std::size_t num_edges_;

  }; // class factor_graph

} // namespace libgm

#endif
