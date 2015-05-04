#ifndef LIBGM_BOOST_GRAPH_HELPERS_HPP
#define LIBGM_BOOST_GRAPH_HELPERS_HPP

/**
 * \file boost_graph_helper.hpp
 * This file contains a number of generic functions that convert BGL's 
 * free function calls to LibGM graph member function calls.
 */

#include <utility>

namespace libgm {
  
  // Accessors
  //============================================================================

  template <typename G>
  std::pair<typename G::vertex_iterator,
	    typename G::vertex_iterator> 
  vertices(const G& g, typename G::vertex_type* = 0) {
    return g.vertices();
  }    

  template <typename G>
  std::pair<typename G::edge_iterator,
	    typename G::edge_iterator>
  edges(const G& g, typename G::vertex_type* = 0) {
    return g.edges();
  }

  template <typename G>
  std::pair<typename G::neighbor_iterator,
	    typename G::neighbor_iterator>
  adjacent_vertices(typename G::vertex_type v, const G& g) {
    return g.neighbors(v);
  }

  template <typename G>
  std::pair<typename G::neighbor_iterator,
	    typename G::neighboriterator>
  inv_adjacent_vertices(typename G::vertex_type v, const G& g) {
    return g.parents(v);
  }

  // todo: children

  template <typename G>
  std::pair<typename G::out_edge_iterator, 
	    typename G::out_edge_iterator>
  out_edges(typename G::vertex_type v, const G& g) {
    return g.out_edges(v);
  }

  template <typename G>
  std::pair<typename G::in_edge_iterator, 
	    typename G::in_edge_iterator>
  in_edges(typename G::vertex_type v, const G& g) {
    return g.in_edges(v);
  }

  template <typename G>
  typename G::vertex_type source(typename G::edge_type e, const G& g) {
    return e.source();
  }

  template <typename G>
  typename G::vertex_type target(typename G::edge_type e, const G& g) {
    return e.target();
  }

  template <typename G>
  size_t out_degree(typename G::vertex_type v, const G& g){
    return g.out_degree(v);
  }

  template <typename G>
  size_t in_degree(typename G::vertex_type v, const G& g) {
    return g.in_degree(v);
  }

  template <typename G>
  size_t num_vertices(const G& g, typename G::vertex_type* = 0) {
    return g.num_vertices();
  }

  template <typename G>
  size_t num_edges(const G& g, typename G::vertex_type* = 0) {
    return g.num_edges();
  }

  template <typename G>
  std::pair<typename G::edge, bool>
  edge(typename G::vertex_type u, typename G::vertex_type v, const G& g) {
    return g.edge(u, v); // TODO: What to do here?
  }

  // Modification
  //============================================================================

  template <typename G>
  typename G::vertex_type add_vertex(G& g, typename G::vertex_type* = 0) {
    static_assert(sizeof(G) == 0, "Unsupported function");
  }

  template <typename G, typename VertexProperty>
  typename G::vertex_type 
  add_vertex(const VertexProperty& p, G& g, typename G::vertex_type* = 0) {
    static_assert(sizeof(G) == 0, "Unsupported function");
  }

  template <typename G>
  void clear_vertex(typename G::vertex_type v, G& g) {
    g.clear_edges(v);
  }

  template <typename G>
  void clear_out_edges(typename G::vertex_type u, G& g) {
    g.clear_out_edges(u);
  }

  template <typename G>
  void clear_in_edges(typename G::vertex_type v, G& g) {
    g.clear_in_edges(v);
  }

  template <typename G>
  void remove_vertex(typename G::vertex_type v, G& g) {
    g.remove_vertex(v);
  }

  template <typename G>
  std::pair<typename G::edge, bool>
  add_edge(typename G::vertex_type u, typename G::vertex_type v, G& g) {
    return g.add_edge(u, v); // TODO: check
  }

  template <typename G, typename EdgeProperty>
  std::pair<typename G::edge, bool>
  add_edge(typename G::vertex_type u, typename G::vertex_type v,
           const EdgeProperty& p, G& g) {
    return g.add_edge(u, v, p); // TODO: check
  }

  template <typename G>
  void remove_edge(typename G::vertex_type u, typename G::vertex_type v, G& g) {
    g.remove_edge(u, v);
  }


  template <typename G>
  void remove_edge(typename G::edge_type e, G& g, typename G::bgl* = 0) {
    g.remove_edge(e);
  }

  template <typename G>
  void remove_edge(typename G::out_edge_iterator it, G& g, 
                   typename G::vertex_type* =0){
    g.remove_edge(it);
  }

  template <typename G, typename Predicate>
  void remove_edge_if(typename G::vertex_type u, Predicate p, G& g) {
    g.remove_edge_if(u, p);
  }

  template <typename G, typename Predicate>
  void remove_out_edge_if(typename G::vertex_type u, Predicate p, G& g) {
    g.remove_out_edge_if(u, p);
  }

  template <typename G, typename Predicate>
  void remove_in_edge_if(typename G::vertex_type u, Predicate p, G& g) {
    g.remove_in_edge_if(u, p);
  }

} // namespace libgm

#endif
