#ifndef LIBGM_GRAPH_CONCEPTS_HPP
#define LIBGM_GRAPH_CONCEPTS_HPP

namespace libgm {

  /**
   * The concept that represents a basic graph. A graph may be thought
   * of as a container, providing a map from vertices and edges to the
   * associated data.  In addition to providing basic container
   * facilities, a graph represents the relationship between edges and
   * vertices.
   *
   * A graph must define the type of a vertex an edge, and the
   * associated vertex_property and edge_property.
   *
   * \ingroup concepts, graph_concepts
   */
  template <typename G>
  struct Graph : DefaultConstructible<G> {

    /**
     * The vertex type which must be boost::Comparable,
     * boost::DefaultConstructible, and
     * boost::CopyConstructable. Typically this will be a relativley
     * lightweight type such as pointer or std::size_t.  This will
     * typically be a user defined type.
     */
    typedef typename G::vertex vertex;
    concept_assert(( boost::Comparable<vertex> ));
    concept_assert(( boost::DefaultConstructible<vertex> ));
    concept_assert(( boost::CopyConstructible<vertex> ));

    /**
     * The data associated with a vertex. Graphs may be thought of as
     * associative containers mapping from vertices and edges to the
     * associated data.  This will typicall be a user defined type.
     */
    typedef typename G::vertex_property vertex_property;

    /**
     * The edge type which must be boost::Comparable,
     * boost::DefaultConstructible, and
     * boost::CopyConstructable. Typically this will be a light weight
     * compoenent over the vertex type.  However, unlike the vertex
     * type this will usually be an opaque type specified by the
     * graph.
     */
    typedef typename G::edge edge;
    concept_assert(( boost::Comparable<edge> ));
    concept_assert(( boost::DefaultConstructible<edge> ));
    concept_assert(( boost::CopyConstructible<edge> ));


    /**
     * The data associated with an edge.  Graphs may be thought of as
     * associative containers mapping from vertices and edges to the
     * associated data.  This will typically be a user defined type.
     */
    typedef typename G::edge_property edge_property;

    /**
     * The iterator over vertices.  This iterator must satisfy the
     * boost::ForwardIterator Concept.  Because we are not permitting
     * runtime poymorphism we require a separate vertex iterator for
     * every "type" of collection of vertices.
     */
    typedef typename G::vertex_iterator vertex_iterator;
    typedef typename G::parent_iterator parent_iterator;
    typedef typename G::child_iterator  child_iterator;
    concept_assert(( boost::InputIterator<vertex_iterator> ));
    concept_assert(( boost::ForwardIterator<parent_iterator> ));
    concept_assert(( boost::ForwardIterator<child_iterator> ));


    /**
     * A vertex range is a std::pair where the first element is the
     * start iterator and the second element is the end iterator.
     * This will be used to represent the range over all
     * vertices. Because we are not premitting runtime polymorphism we
     * require a separte range for each type of iterator.
     */
    typedef std::pair<vertex_iterator, vertex_iterator> vertex_range;
    typedef std::pair<parent_iterator, parent_iterator> parent_range;
    typedef std::pair<child_iterator, child_iterator>   child_range;


    typedef typename G::edge_iterator edge_iterator;
    typedef typename G::in_edge_iterator in_edge_iterator;
    typedef typename G::out_edge_iterator out_edge_iterator;
    concept_assert(( boost::ForwardIterator<edge_iterator> ));
    concept_assert(( boost::ForwardIterator<in_edge_iterator> ));
    concept_assert(( boost::ForwardIterator<out_edge_iterator> ));

    // For brevity we define the range types
    typedef std::pair<edge_iterator, edge_iterator> edge_range;
    typedef std::pair<in_edge_iterator, in_edge_iterator> in_edge_range;
    typedef std::pair<out_edge_iterator, out_edge_iterator> out_edge_range;


    /**
     * Returns the vertices in this graph.
     *
     * \return the range containing all vertices
     */
    vertex_range vertices() const;

    /**
     * Returns the edges of this graph()
     *
     * \return the range containing all edges
     */
    edge_range edges() const;

    /**
     * Test if this graph contains the vertex u.
     *
     * \param u a vertex
     * \return if this graph contains u
     */
    bool contains(vertex u) const;

    /**
     * Test if this graph contains the edge e.
     *
     * \param e an edge
     * \return if this graph contains e
     */
    bool contains(edge e) const;

    /**
     * Test if this graph contains the edge (u,v).
     *
     * \param u source vertex
     * \param v target vertex
     * \return if this graph contains (u,v)
     */
    bool contains(vertex u, vertex v) const;

    /**
     * Returns the parents of u.
     *
     * \param u the vertex whose parents are desired
     * \return the range of parents.
     * \exception std::out_of_range vertex u not present in graph
     */
    parent_range parents(vertex u) const;

    /**
     * Returns the children of u.
     *
     * \param u a vertex in the graph
     * \return the range of children.
     * \exception std::out_of_range vertex u not present in graph
     */
    child_range children(vertex u) const;

    /**
     * Returns the edges emanating from the vertex u.
     *
     * \param u a vertex in the graph
     * \return the range of edges emanating from u
     * \exception std::out_of_range vertex u not present in graph
     */
    out_edge_range out_edges(vertex u) const;

    /**
     * Returns the edges arriving at the vertex u.
     *
     * \param u a vertex in the graph
     * \return the range of edges arriving at u
     * \exception std::out_of_range vertex u not present in graph
     */
    in_edge_range in_edges(vertex u) const;

    /**
     * Returns the source u of the edge (u,v)=e.
     *
     * \param e an edge in the graph
     * \return the vertex from which e emanates
     * \exception std::out_of_range edge e is not present in graph
     */
    vertex source(edge e) const;

    /**
     * Returns the destination or target v of the edge (u,v)=e
     *
     * \param e an edge in the graph
     * \return the vertex to which e arrives
     * \exception std::out_of_range edge e is not present in graph
     */
    vertex target(edge e) const;

    /**
     * Return the number of edges emanating from u.
     *
     * \param u a vertex in the graph
     * \return the number of edges emanating from u
     * \exception std::out_of_range vertex is not present in graph
     */
    std::size_t out_degree(vertex u) const;

    /**
     * Return the number of edges arriving at u.
     *
     * \param u a vertex in the graph
     * \return the number of edges arriving at u
     * \exception std::out_of_range vertex is not present in graph
     */
    std::size_t in_degree(vertex u) const;

    /**
     * Returns the total number of edges leaving and arriving from u.
     *
     * \param u a vertex in the graph
     * \return the number of edges connected to u
     * \exception std::out_of_range vertex is not present in graph
     */
    std::size_t degree(vertex u) const;

    /**
     * Returns true if the graph() has no vertices
     *
     * \return true if the graph has no vertices
     */
    bool empty() const;

    /**
     * Returns the number of vertices in the graph()
     *
     * \return the number of vertices
     */
    std::size_t num_vertices() const;

    /**
     * Returns the number of edges in the graph()
     *
     * \return the number of edges in the graph
     */
    std::size_t num_edges() const;

    /**
     * Returns the edge connecting u to v in the graph.
     *
     * \param u the origin vertex
     * \param v the destination vertex
     * \return the edge (u,v)
     * \exception std::out_of_range if either u or v is not in the graph
     */
    edge get_edge(vertex u, vertex v) const;

    /**
     * Returns an edge with the reverse source and target
     *
     * \param e an edge in the graph
     * \return the edge (v,u) where e=(u,v)
     * \exception std::out_of_range if e is not present in the graph
     */
    edge reverse(edge e) const;

    /**
     * Get the property associated with the vertex u
     *
     * \param u a vertex in the graph
     * \return the property associated with the vertex u
     * \exception std::out_of_range if u is not present in the graph
     */
    const vertex_property& operator[](vertex u) const;
    vertex_property& operator[](vertex u);
    const vertex_property& property(vertex u) const;
    vertex_property& property(vertex u);


    /**
     * Get the property associated with the edge e
     *
     * \param e an edge in the graph
     * \return the property associated with the edge e
     * \exception std::out_of_range if e is not present in the graph
     */
    const edge_property& operator[](const edge& e) const;
    edge_property& operator[](const edge& u);
    const edge_property& property(const edge& e) const;
    edge_property& property(const edge& e);
    const edge_property& property(const vertex& u, const vertex& v) const;
    edge_property& property(const vertex& u, const vertex& v);



    concept_usage(Graph) {
      libgm::same_type( g.vertices(),                     vr);
      libgm::same_type( g.edges(),                        er);
      libgm::same_type( g.contains(vertex()),             bool());
      libgm::same_type( g.contains(edge()),               bool());
      libgm::same_type( g.contains(vertex(), vertex()),   bool());
      libgm::same_type( g.parents(vertex()),              pr);
      libgm::same_type( g.children(vertex()),             cr);
      libgm::same_type( g.out_edges(vertex()),            oer);
      libgm::same_type( g.in_edge(vertex()),              ier);
      libgm::same_type( g.source(edge()),                 vertex());
      libgm::same_type( g.target(edge()),                 vertex());
      libgm::same_type( g.out_degree(vertex()),           std::size_t());
      libgm::same_type( g.in_degree(vertex()),            std::size_t());
      libgm::same_type( g.degree(vertex()),               std::size_t());
      libgm::same_type( g.empty(),                        bool());
      libgm::same_type( g.num_vertices(),                 std::size_t());
      libgm::same_type( g.num_edges(),                    std::size_t());
      libgm::same_type( g.edge(vertex(), vertex()),       edge());
      libgm::same_type( g.reverse(edge()),                edge());
      libgm::same_type( g[vertex()],                      vertex_property());
      libgm::same_type( g.vertex_property(vertex()),      vertex_property());
      libgm::same_type( g[edge()],                        edge_property());
      libgm::same_type( g.edge_property(edge()),          edge_property());
      libgm::same_type( g.edge_property(vertex(),vertex()), edge_property());
    }
  private:
    G g;
    vertex_range vr;
    edge_range er;
    parent_range pr;
    child_range cr;
    in_edge_range ier;
    out_edge_range oer;

  }; // concept Graph

  /**
   * An extension of graph which permits adding and removing vertices
   * and edges.
   *
   * \ingroup concepts, graph_concepts
   */
  template <typename G>
  struct MutableGraph : Graph<G> {

    // Inherit types from base concept.  These already checked.
    typedef Graph<G> base;
    typedef typename base::vertex             vertex;
    typedef typename base::vertex_property    vertex_property;
    typedef typename base::edge               edge;
    typedef typename base::edge_property      edge_property;
    typedef typename base::vertex_iterator    vertex_iterator;
    typedef typename base::parent_iterator    parent_iterator;
    typedef typename base::child_iterator     child_iterator;
    typedef typename base::edge_iterator      edge_iterator;
    typedef typename base::in_edge_iterator   in_edge_iterator;
    typedef typename base::out_edge_iterator  out_edge_iterator;
    typedef typename base::vertex_range        vertex_range;
    typedef typename base::parent_range       parent_range;
    typedef typename base::child_range        child_range;
    typedef typename base::edge_range         edge_range;
    typedef typename base::in_edge_range      in_edge_range;
    typedef typename base::out_edge_range     out_edge_range;



    /**
     * Adds the vertex u to this graph and associates the vertex
     * property p with that vertex. Returns true if the vertex was
     * already present. If the vertex was already present then the
     * the property is replaced with p.
     *
     * \param u the vertex
     * \return boolean that is true if the vertex was already present
     */
    bool add(vertex u, const vertex_property& p);

    /**
     * Adds the edge e to this graph and associates the edge
     * property p with that edge. Returns true if the edge was
     * already present. If the edge was already present then the
     * the property is replaced with p.
     *
     * \param e the edge
     * \param p the vertex
     * \return boolean that is true if the edge was already present
     */
    bool add(vertex u, vertex v, const edge_property& p);


    /**
     * Remove all edges to and from a vertex as well as the vertex
     * itself
     *
     * \param u the vertex
     * \exception std::out_of_range if the vertex is not present.
     */
    void remove(vertex u);

    /**
     * Remove the edge.
     *
     * \param e an edge to remove
     * \exception std::out_of_range if the edge is not present.
     */
    void remove(edge e);

    /**
     * Removes all edges to and from a vertex.  Does not remove the
     * vertex itself.
     *
     * \param the vertex
     * \exception std::out_of_range if the vertex is not present.
     */
    void detach(vertex u);

    /**
     * Remove all out edges from the vertex u that satisfy the
     * predicate.
     *
     * \param u a vertex in the graph
     * \param predicate a functor that has type edge -&gt bool
     * \exception std::out_of_range if the vertex u is not present
     */
    template <typename Predicate>
    void remove_out_edge_if(vertex u, const Predicate& predicate);

    /**
     * Remove all in edges from the vertex u that satisfy the
     * predicate.
     *
     * \param u a vertex in the graph
     * \param predicate a functor that has type edge -&gt bool
     * \exception std::out_of_range if the vertex u is not present
     */
    template <typename Predicate>
    void remove_in_edge_if(vertex u, const Predicate& predicate);

    /**
     * Remove all edges from the vertex u that satisfy the
     * predicate.
     *
     * \param u a vertex in the graph
     * \param predicate a functor that has type edge -&gt bool
     * \exception std::out_of_range if the vertex u is not present
     */
    template <typename Predicate>
    void remove_edge_if(vertex u, const Predicate& predicate);

  }; // concept MutableGraph



  /**
   * The variable elimination strategy concept. A variable elimination
   * strategy defines a priority type, and two member functions for
   * computing the priority of a node and determining whose priority
   * may have changed as a result of eliminating a node.
   *
   * \ingroup graph_concepts
   * @see min_degree_strategy, min_fill_strategy, constrained_elim_strategy
   */
  template <typename Strategy, typename Graph>
  struct EliminationStrategy
  {
  private:

  public:
    //concept_assert((Graph<Graph>));
    typedef typename Graph::vertex vertex;

    typedef typename Strategy::priority_type priority_type;

    //! Computes the priority of a vertex.
    priority_type priority(vertex, const Graph& g);

    //! Computes the set of vertices whose priority may change if a
    //! designated vertex is eliminated.
    template <typename OutputIterator>
    void updated(vertex v, const Graph& g, OutputIterator updated);

    //! Implementation of the concept checking class
    concept_usage(EliminationStrategy) {
      libgm::same_type(s.priority(v, g), p);
      s.updated(v, g, out);
    }

  private:
    Strategy s;
    Graph g;
    vertex v;
    priority_type p;
    boost::output_iterator_archetype<vertex> out;

  }; // concept EliminationStrategy

} // namespace libgm

#endif
