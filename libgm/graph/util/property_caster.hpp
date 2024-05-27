namespace libgm {

template <typename Graph, typename VertexProperty, typename EdgeProperty>
class PropertyCaster : public Graph {
  std::add_lvalue_reference_t<VertexProperty> operator[](vertex_descriptor u) {
    if constexpr (!std::is_same_v<VertexProperty, void>) {
      return static_cast<VertexProperty&>(ClusterGraph::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with a vertex.
  std::add_lvalue_reference_t<const VertexProperty> operator[](vertex_descriptor u) const {
    if constexpr (!std::is_same_v<VertexProperty, void>) {
      return static_cast<const VertexProperty&>(ClusterGraph::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with an edge.
  std::add_lvalue_reference_t<EdgeProperty> operator[](const edge_descriptor& e) {
    if constexpr (!std::is_same_v<EdgeProperty, void>) {
      return static_cast<EdgeProperty&>(ClusterGraph::operator[](e));
    }
  }

  /// Returns the strongly-typed property associated with an edge.
  std::add_lvalue_reference_t<const EdgeProperty> operator[](const edge_descriptor& e) const {
    if constexpr (!std::is_same_v<EdgeProperty, void>) {
      return static_cast<const EdgeProperty&>(ClusterGraph::operator[](e));
    }
  }
};

}  // namespace libgm
