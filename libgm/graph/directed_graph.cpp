#include "directed_graph.hpp"

#include <algorithm>

namespace libgm {

struct DirectedGraph::Vertex {
  std::vector<Vertex*> parents;
  IntrusiveList<Vertex> children;
  IntrusiveList<Vertex>::Hook hook;
  IntrusiveList<Vertex>::HookArray parent_hooks;
  const Impl* impl = nullptr;
  mutable size_t index = static_cast<size_t>(-1);

  explicit Vertex(Impl* impl)
    : impl(impl) {}
};

struct DirectedGraph::Impl {
  IntrusiveList<Vertex> vertices;
  size_t num_vertices = 0;
  size_t num_edges = 0;
  PropertyLayout property_layout;

  explicit Impl(PropertyLayout layout = {})
    : property_layout(layout) {}

  ~Impl() {
    clear();
  }

  Vertex* allocate_vertex() const {
    return property_layout.allocate<Vertex>(const_cast<Impl*>(this));
  }

  void free_vertex(Vertex* vertex) const {
    property_layout.free(vertex);
  }

  void remove_in_edges(Vertex* vertex) {
    num_edges -= vertex->parents.size();
    for (size_t i = 0; i < vertex->parents.size(); ++i) {
      Vertex* parent = vertex->parents[i];
      parent->children.erase(vertex, vertex->parent_hooks[i]);
    }
    vertex->parents.clear();
    vertex->parent_hooks = {};
  }

  void validate_parents(Vertex* vertex, const std::vector<Vertex*>& parents) const {
    for (Vertex* parent : parents) {
      if (!parent || parent->impl != this) {
        throw std::out_of_range("DirectedGraph::set_parents: parent not found");
      }
      if (parent == vertex) {
        throw std::invalid_argument("DirectedGraph::set_parents: self-parent is not allowed");
      }
    }
  }

  void set_parents(Vertex* vertex, std::vector<Vertex*> parents) {
    validate_parents(vertex, parents);
    remove_in_edges(vertex);
    vertex->parent_hooks.reset(parents.size());
    for (size_t i = 0; i < parents.size(); ++i) {
      parents[i]->children.push_back(vertex, vertex->parent_hooks[i]);
    }
    num_edges += parents.size();
    vertex->parents = std::move(parents);
  }

  void clear() {
    while (Vertex* vertex = vertices.front()) {
      free_vertex(vertex);
    }
    num_vertices = 0;
    num_edges = 0;
  }

  void compute_indices() const {
    size_t index = 0;
    for (Vertex* vertex : vertices) {
      vertex->index = index++;
    }
  }

  std::unique_ptr<Impl> clone() const {
    auto result = std::make_unique<Impl>(property_layout);
    ankerl::unordered_dense::map<const Vertex*, Vertex*> map;
    map.reserve(num_vertices);

    for (Vertex* src : vertices) {
      Vertex* dst = result->allocate_vertex();
      result->vertices.push_back(dst, dst->hook);
      ++result->num_vertices;
      map.emplace(src, dst);
    }

    auto src_it = vertices.begin();
    auto dst_it = result->vertices.begin();
    for (; src_it != vertices.end(); ++src_it, ++dst_it) {
      Vertex* src = *src_it;
      Vertex* dst = *dst_it;
      std::vector<Vertex*> mapped_parents;
      mapped_parents.reserve(src->parents.size());
      for (Vertex* parent : src->parents) {
        mapped_parents.push_back(map.at(parent));
      }
      result->set_parents(dst, std::move(mapped_parents));
      property_layout.destroy_and_copy_construct(dst, src);
    }

    return result;
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::make_size_tag(num_vertices));
    compute_indices();

    for (Vertex* vertex : vertices) {
      std::vector<size_t> parent_indices;
      parent_indices.reserve(vertex->parents.size());
      for (Vertex* parent : vertex->parents) {
        parent_indices.push_back(parent->index);
      }
      ar(cereal::make_nvp("parents", parent_indices));
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    clear();

    cereal::size_type n;
    ar(cereal::make_size_tag(n));

    std::vector<Vertex*> loaded_vertices;
    loaded_vertices.reserve(n);
    std::vector<size_t> parent_indices;

    for (size_t i = 0; i < n; ++i) {
      Vertex* vertex = allocate_vertex();
      vertices.push_back(vertex, vertex->hook);
      loaded_vertices.push_back(vertex);
      ++num_vertices;
      vertex->index = i;
    }

    for (size_t i = 0; i < n; ++i) {
      ar(cereal::make_nvp("parents", parent_indices));
      std::vector<Vertex*> parents;
      parents.reserve(parent_indices.size());
      for (size_t parent_index : parent_indices) {
        parents.push_back(loaded_vertices.at(parent_index));
      }
      set_parents(loaded_vertices[i], std::move(parents));
    }
  }
};

DirectedGraph::Impl& DirectedGraph::impl() {
  return *impl_;
}

const DirectedGraph::Impl& DirectedGraph::impl() const {
  return *impl_;
}

DirectedGraph::Vertex& DirectedGraph::data(Vertex* u) {
  assert(contains(u));
  return *u;
}

const DirectedGraph::Vertex& DirectedGraph::data(Vertex* u) const {
  assert(contains(u));
  return *u;
}

DirectedGraph::DirectedGraph()
  : impl_(std::make_unique<Impl>()) {}

DirectedGraph::DirectedGraph(PropertyLayout layout)
  : impl_(std::make_unique<Impl>(layout)) {}

DirectedGraph::DirectedGraph(const DirectedGraph& other)
  : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

DirectedGraph::DirectedGraph(DirectedGraph&& other) noexcept = default;

DirectedGraph& DirectedGraph::operator=(const DirectedGraph& other) {
  if (this != &other) {
    impl_ = other.impl_ ? other.impl_->clone() : nullptr;
  }
  return *this;
}

DirectedGraph& DirectedGraph::operator=(DirectedGraph&& other) noexcept = default;

DirectedGraph::~DirectedGraph() = default;

std::ranges::subrange<DirectedGraph::out_edge_iterator>
DirectedGraph::out_edges(Vertex* u) const {
  const IntrusiveList<Vertex>& children = data(u).children;
  return {out_edge_iterator(children.begin(), u), out_edge_iterator(children.end(), u)};
}

std::ranges::subrange<DirectedGraph::in_edge_iterator>
DirectedGraph::in_edges(Vertex* u) const {
  const std::vector<Vertex*>& parents = data(u).parents;
  return {in_edge_iterator(parents.begin(), u), in_edge_iterator(parents.end(), u)};
}

std::ranges::subrange<DirectedGraph::adjacency_iterator>
DirectedGraph::adjacent_vertices(Vertex* u) const {
  const IntrusiveList<Vertex>& children = data(u).children;
  return {children.begin(), children.end()};
}

std::ranges::subrange<DirectedGraph::vertex_iterator> DirectedGraph::vertices() const {
  return {impl().vertices.begin(), impl().vertices.end()};
}

bool DirectedGraph::contains(Vertex* u) const {
  return u && u->impl == impl_.get();
}

bool DirectedGraph::contains(Vertex* u, Vertex* v) const {
  if (!contains(u) || !contains(v)) {
    return false;
  }
  return std::find(v->parents.begin(), v->parents.end(), u) != v->parents.end();
}

bool DirectedGraph::contains(edge_descriptor e) const {
  return contains(e.source(), e.target());
}

DirectedGraph::edge_descriptor DirectedGraph::edge(Vertex* u, Vertex* v) const {
  return contains(u, v) ? edge_descriptor(u, v) : edge_descriptor();
}

size_t DirectedGraph::in_degree(Vertex* u) const {
  return data(u).parents.size();
}

size_t DirectedGraph::out_degree(Vertex* u) const {
  return data(u).children.empty() ? 0 : std::ranges::distance(data(u).children);
}

size_t DirectedGraph::degree(Vertex* u) const {
  return in_degree(u) + out_degree(u);
}

bool DirectedGraph::empty() const {
  return impl().num_vertices == 0;
}

size_t DirectedGraph::num_vertices() const {
  return impl().num_vertices;
}

size_t DirectedGraph::num_edges() const {
  return impl().num_edges;
}

const std::vector<DirectedGraph::Vertex*>& DirectedGraph::parents(Vertex* u) const {
  return data(u).parents;
}

OpaqueRef DirectedGraph::property(Vertex* u) {
  assert(contains(u));
  return impl().property_layout.get(u);
}

OpaqueCref DirectedGraph::property(Vertex* u) const {
  assert(contains(u));
  return impl().property_layout.get(static_cast<const Vertex*>(u));
}

DirectedGraph::Vertex* DirectedGraph::add_vertex(std::vector<Vertex*> parents) {
  Vertex* vertex = impl().allocate_vertex();
  impl().vertices.push_back(vertex, vertex->hook);
  ++impl().num_vertices;
  try {
    impl().set_parents(vertex, std::move(parents));
  } catch (...) {
    --impl().num_vertices;
    impl().free_vertex(vertex);
    throw;
  }
  return vertex;
}

void DirectedGraph::set_parents(Vertex* u, std::vector<Vertex*> parents) {
  assert(contains(u));
  impl().set_parents(u, std::move(parents));
}

size_t DirectedGraph::remove_vertex(Vertex* u) {
  if (!contains(u)) {
    return 0;
  }
  if (!u->children.empty()) {
    throw std::logic_error("DirectedGraph::remove_vertex: vertex has outgoing edges");
  }
  impl().remove_in_edges(u);
  --impl().num_vertices;
  impl().free_vertex(u);
  return 1;
}

void DirectedGraph::remove_in_edges(Vertex* u) {
  if (!contains(u)) {
    throw std::out_of_range("DirectedGraph::remove_in_edges: vertex does not exist");
  }
  impl().remove_in_edges(u);
}

void DirectedGraph::clear() {
  impl().clear();
}

} // namespace libgm
