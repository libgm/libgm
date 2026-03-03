#pragma once

namespace libgm {

template <typename GRAPH, typename VP, typename EP>
struct PropertyCaster : GRAPH {
  using GRAPH::GRAPH;

  /// Returns the strongly-typed property associated with a vertex.
  std::add_lvalue_reference_t<VP> operator[](typename GRAPH::vertex_descriptor u) {
    if constexpr (!std::is_same_v<VP, void>) {
      return static_cast<VP&>(GRAPH::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with a vertex.
  std::add_lvalue_reference_t<const VP> operator[](typename GRAPH::vertex_descriptor u) const {
    if constexpr (!std::is_same_v<VP, void>) {
      return static_cast<const VP&>(GRAPH::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with an edge.
  std::add_lvalue_reference_t<EP> operator[](typename GRAPH::edge_descriptor e) {
    if constexpr (!std::is_same_v<EP, void>) {
      return static_cast<EP&>(GRAPH::operator[](e));
    }
  }

  /// Returns the strongly-typed property associated with an edge.
  std::add_lvalue_reference_t<const EP> operator[](typename GRAPH::edge_descriptor e) const {
    if constexpr (!std::is_same_v<EP, void>) {
      return static_cast<const EP&>(GRAPH::operator[](e));
    }
  }
}; // struct PropertyCaster

template <typename GRAPH, typename VP>
struct VertexPropertyCaster : GRAPH {
  using GRAPH::GRAPH;

  /// Returns the strongly-typed property associated with a vertex.
  std::add_lvalue_reference_t<VP> operator[](typename GRAPH::vertex_descriptor u) {
    if constexpr (!std::is_same_v<VP, void>) {
      return static_cast<VP&>(GRAPH::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with a vertex.
  std::add_lvalue_reference_t<const VP> operator[](typename GRAPH::vertex_descriptor u) const {
    if constexpr (!std::is_same_v<VP, void>) {
      return static_cast<const VP&>(GRAPH::operator[](u));
    }
  }
}; // struct VertexPropertyCaster

}  // namespace libgm
