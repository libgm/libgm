#ifndef LIBGM_GRAPH_PROPERTY_FN_HPP
#define LIBGM_GRAPH_PROPERTY_FN_HPP

#include <type_traits>

namespace libgm {

  /**
   * A functor type that returns the vertex property by reference.
   * Depending on whether Graph is const-qualified or not,
   * returns a const- or mutable reference to the property.
   * \ingroup graph_types
   */
  template <typename Graph>
  struct vertex_property_fn {
    typedef typename Graph::vertex_type argument_type;
    typedef typename Graph::vertex_property vertex_property;
    typedef typename std::conditional< 
      std::is_const<Graph>::value, const vertex_property&, vertex_property& 
    >::type result_type;

    vertex_property_fn(Graph* graph) : graph_(graph) { }
    
    result_type operator()(argument_type v) const {
      return (*graph_)[v];
    }

  private:
    Graph* graph_;
  }; // struct vertex_property_fn

  /**
   * A functor type that returns the type-1 vertex property by reference.
   * Depending on whether Graph is const-qualified or not,
   * returns a const- or mutable reference to the property.
   * \ingroup graph_types
   */
  template <typename Graph>
  struct vertex1_property_fn {
    typedef typename Graph::vertex1_type argument_type;
    typedef typename Graph::vertex1_property vertex_property;
    typedef typename std::conditional< 
      std::is_const<Graph>::value, const vertex_property&, vertex_property& 
    >::type result_type;

    vertex1_property_fn(Graph* graph) : graph_(graph) { }
    
    result_type operator()(argument_type v) const {
      return (*graph_)[v];
    }

  private:
    Graph* graph_;
  }; // struct vertex_property_fn

  /**
   * A functor type that returns the type-1 vertex property by reference.
   * Depending on whether Graph is const-qualified or not,
   * returns a const- or mutable reference to the property.
   * \ingroup graph_types
   */
  template <typename Graph>
  struct vertex2_property_fn {
    typedef typename Graph::vertex2_type argument_type;
    typedef typename Graph::vertex2_property vertex_property;
    typedef typename std::conditional< 
      std::is_const<Graph>::value, const vertex_property&, vertex_property& 
    >::type result_type;

    vertex2_property_fn(Graph* graph) : graph_(graph) { }
    
    result_type operator()(argument_type v) const {
      return (*graph_)[v];
    }

  private:
    Graph* graph_;
  }; // struct vertex_property_fn

  /**
   * A functor type that returns the edge property by reference.
   * Depending on whether Graph is const-qualified or not,
   * returns a const- or mutable reference to the property.
   * \ingroup graph_types
   */
  template <typename Graph>
  struct edge_property_fn {
    typedef typename Graph::edge_type argument_type;
    typedef typename Graph::edge_property edge_property;
    typedef typename std::conditional< 
      std::is_const<Graph>::value, const edge_property&, edge_property& 
    >::type result_type;
  
    edge_property_fn(Graph* graph) : graph_(graph) { }

    result_type operator()(const argument_type& e) const {
      return (*graph_)[e];
    }

  private:
    Graph* graph_;
  }; // struct edge_property_fn

} // namespace libgm

#endif
