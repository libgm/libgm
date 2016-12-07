#ifndef LIBGM_VERTEX_EDGE_PROPERTY_ITERATOR_HPP
#define LIBGM_VERTEX_EDGE_PROPERTY_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * This iterator is used to iterate over the key in an associative container.
   * \ingroup iterator
   */
  template <class Graph>
  class vertex_edge_property_iterator:
    public std::iterator<std::forward_iterator_tag,
                         const annotated<typename Graph::vertex_type,
                                         typename Graph::vertex_property>  > {

  public:
    using vertex_type     = typename Graph::vertex_type;
    using vertex_property = typename Graph::vertex_property;
    using edge_property   = typename Graph::edge_property;
    using vertex_iterator = typename Graph::vertex_iterator;
    using edge_iterator   = typename Graph::edge_iterator;

    static_assert(std::is_same<vertex_property, edge_property>::value,
                  "The vertex and edge property must be the same type");

  public:
    vertex_edge_property_iterator()
      : vit_(), eit_() { }

    vertex_edge_property_iterator(vertex_iterator vit,
                                  vertex_iterator vend,
                                  edge_iterator eit)
      : vit_(vit), vend_(vend), eit_(eit) {
      if (vit != vend) {
        property_.domain.assign(1, *vit);
        property_.object = (*graph_)[*vit];
      }
    }

    const annotated<vertex_type, vertex_property>& operator*() const {
      return property_;
    }

    vertex_edge_property_iterator& operator++() {
      if (vit != vend) {
        ++vit;
        property_.domain.assign(1, *vit);
        property_.object = (*graph_)[*vit];
      } else if (eit) {
        ++eit;
        property_.domain.assign({eit->source(), eit->target()});
        property_.object = (*graph_)[*eit];
      }
      return *this;
    }

    vertex_edge_property_iterator operator++(int) {
      return map_key_iterator(it++);
    }

    bool operator==(const vertex_edge_property_iterator& other) const {
      return vit_ == other.vit_ && eit_ == other.eit_;
    }

    bool operator!=(const vertex_edge_property_iterator& other) const {
      return vit_ != other.vit_ || eit_ != other.eit_;
    }

  private:
    const Graph* graph_;
    vertex_iterator vit_;
    edge_iterator eit_;
    annotated<vertex_type, vertex_property> property_;

  }; // class vertex_edge_property_iterator

} // namespace libgm

#endif

