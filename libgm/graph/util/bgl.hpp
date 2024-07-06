#include <utility>

namespace libgm {


// Ranges
//============================================================================

template <typename G>
auto out_edges(typename G::vertex_descriptor u, const G& g) {
  return make_iterator_pair(g.out_edges(u));
}

template <typename G>
auto in_edges(typename G::vertex_descriptor u, const G& g) {
  return make_iterator_pair(g.in_edges(u));
}

template <typename G>
auto adjacent_vertices(typename G::vertex_descriptor u, const G& g) {
  return make_iterator_pair(g.adjacent_vertices(u));
}

template <typename G>
auto vertices(const G& g) {
  return make_iterator_pair(g.vertices());
}

template <typename G>
auto edges(const G& g) {
  return make_iterator_pair(e.edges());
}

// Accessors
//============================================================================

template <typename G>
typename G::vertex_descriptor source(typename G::edge_descriptor e, const G&) {
  return e.source();
}

template <typename G>
typename G::vertex_descriptor target(typename G::edge_descriptor e, const G&) {
  return e.target();
}

template <typename G>
typename G::degree_size_type out_degree(typename G::vertex_descriptor v, const G& g) {
  return g.out_degree(v);
}

template <typename G>
typename G::degree_size_type in_degree(typename G::vertex_descriptor v, const G& g) {
  return g.in_degree(v);
}

template <typename G>
typename G::degree_size_type degree(typename G::vertex_descriptor v, const G& g) {
  return g.degree(v);
}

template <typename G>
typename G::vertices_size_type num_vertices(const G& g) {
  return g.num_vertices();
}

template <typename G>
typename G::edges_size_type num_edges(const G& g) {
  return g.num_edges();
}

template <typename G>
std::pair<typename G::edge_descriptor, bool> edge(typename G::vertex_descriptor u,
                                                  typename G::vertex_descriptor v,
                                                  const G& g) {
  return g.edge(u, v);
}

// Modifications
//============================================================================

template <typename G>
typename G::vertex_descriptor add_vertex(G& g, typename G::vertex_descriptor* = 0) {
  static_assert(sizeof(G) == 0, "Unsupported function");
}

template <typename G>
void clear_vertex(typename G::vertex_descriptor v, G& g) {
  g.clear_edges(v);
}

template <typename G>
void clear_out_edges(typename G::vertex_descriptor u, G& g) {
  g.clear_out_edges(u);
}

template <typename G>
void clear_in_edges(typename G::vertex_descriptor v, G& g) {
  g.clear_in_edges(v);
}

template <typename G>
void remove_vertex(typename G::vertex_descriptor v, G& g) {
  g.remove_vertex(v);
}

template <typename G>
std::pair<typename G::edge_descriptor, bool>
add_edge(typename G::vertex_descriptor u, typename G::vertex_descriptor v, G& g) {
  return g.add_edge(u, v);
}

template <typename G>
void remove_edge(typename G::vertex_descriptor u, typename G::vertex_descriptor v, G& g) {
  g.remove_edge(u, v);
}

template <typename G>
void remove_edge(typename G::edge_descriptor e, G& g) {
  g.remove_edge(e);
}

template <typename G>
void remove_edge(typename G::out_edge_iterator it, G& g) {
  g.remove_edge(it);
}

template <typename G, typename Predicate>
void remove_edge_if(typename G::vertex_descriptor u, Predicate p, G& g) {
  g.remove_edge_if(u, p);
}

template <typename G, typename Predicate>
void remove_out_edge_if(typename G::vertex_descriptor u, Predicate p, G& g) {
  g.remove_out_edge_if(u, p);
}

template <typename G, typename Predicate>
void remove_in_edge_if(typename G::vertex_descriptor u, Predicate p, G& g) {
  g.remove_in_edge_if(u, p);
}

}