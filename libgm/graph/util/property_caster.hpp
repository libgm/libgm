#pragma once

namespace libgm {

template <typename Graph, typename VertexProperty, typename EdgeProperty>
class PropertyCaster : public Graph {
  /// Returns the strongly-typed property associated with a vertex.
  std::add_lvalue_reference_t<VertexProperty> operator[](vertex_descriptor u) {
    if constexpr (!std::is_same_v<VertexProperty, void>) {
      return static_cast<VertexProperty&>(Graph::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with a vertex.
  std::add_lvalue_reference_t<const VertexProperty> operator[](vertex_descriptor u) const {
    if constexpr (!std::is_same_v<VertexProperty, void>) {
      return static_cast<const VertexProperty&>(Graph::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with an edge.
  std::add_lvalue_reference_t<EdgeProperty> operator[](edge_descriptor e) {
    if constexpr (!std::is_same_v<EdgeProperty, void>) {
      return static_cast<EdgeProperty&>(Graph::operator[](e));
    }
  }

  /// Returns the strongly-typed property associated with an edge.
  std::add_lvalue_reference_t<const EdgeProperty> operator[](edge_descriptor e) const {
    if constexpr (!std::is_same_v<EdgeProperty, void>) {
      return static_cast<const EdgeProperty&>(Graph::operator[](e));
    }
  }
};

}  // namespace libgm
