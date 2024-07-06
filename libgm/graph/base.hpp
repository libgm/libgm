struct VertexBase : boost::intrusive::list_base_hook<> {
  /// The id of the vertex (for visualization and hashing).
  size_t id = -1;

  template <typename V>
  V* cast() const {
    return static_cast<T*>(const_cast<VertexBase*>(this));
  }
};

struct EdgeBase : boost::intrusive::list_base_hook<> {
  /// The endpoints of this edge.
  std::array<VertexBase*, 2> endpoints;

  template <typename V, typename E>
  EdgeReference<V, E> ref() const {

  }
};


struct BaseHash {
  size_t operator()(VertexBase*) const noexcept;
};
