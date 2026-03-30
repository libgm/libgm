#include "bipartite_graph.hpp"

#include <algorithm>
#include <stdexcept>

namespace libgm {

struct BipartiteGraph::Vertex1 {
  IntrusiveList<Vertex2> adjacency;
  size_t degree = 0;
  const Impl* impl = nullptr;
  IntrusiveList<Vertex1>::Hook hook;
  mutable size_t index = static_cast<size_t>(-1);

  explicit Vertex1(Impl* impl)
    : impl(impl) {}
};

struct BipartiteGraph::Vertex2 {
  std::vector<Vertex1*> neighbors;
  Impl* impl = nullptr;
  IntrusiveList<Vertex2>::Hook hook;
  IntrusiveList<Vertex2>::HookArray adjacency_hooks;

  explicit Vertex2(Impl* impl)
    : impl(impl) {}

  Vertex2(std::vector<Vertex1*> neighbors, Impl* impl)
    : neighbors(std::move(neighbors)),
      impl(impl),
      adjacency_hooks(this->neighbors.size()) {}
};

struct BipartiteGraph::Impl {
  IntrusiveList<Vertex1> vertices1;
  IntrusiveList<Vertex2> vertices2;
  size_t num_vertices1 = 0;
  size_t num_vertices2 = 0;
  PropertyLayout vertex1_property_layout;
  PropertyLayout vertex2_property_layout;

  explicit Impl(PropertyLayout vertex1_layout = {}, PropertyLayout vertex2_layout = {})
    : vertex1_property_layout(vertex1_layout),
      vertex2_property_layout(vertex2_layout) {}

  ~Impl() {
    clear();
  }

  Vertex1* allocate_vertex1() const {
    return vertex1_property_layout.allocate<Vertex1>(const_cast<Impl*>(this));
  }

  Vertex2* allocate_vertex2(std::vector<Vertex1*> neighbors) const {
    return vertex2_property_layout.allocate<Vertex2>(std::move(neighbors), const_cast<Impl*>(this));
  }

  void free_vertex1(Vertex1* vertex) const {
    vertex1_property_layout.free(vertex);
  }

  void free_vertex2(Vertex2* vertex) const {
    vertex2_property_layout.free(vertex);
  }

  void validate_neighbors(const std::vector<Vertex1*>& neighbors) const {
    for (Vertex1* neighbor : neighbors) {
      if (!neighbor || neighbor->impl != this) {
        throw std::out_of_range("BipartiteGraph::add_vertex2: neighbor not found");
      }
    }
  }

  Vertex1* add_vertex1() {
    Vertex1* vertex = allocate_vertex1();
    vertices1.push_back(vertex, vertex->hook);
    ++num_vertices1;
    return vertex;
  }

  Vertex2* add_vertex2(std::vector<Vertex1*> neighbors) {
    validate_neighbors(neighbors);
    Vertex2* vertex = allocate_vertex2(std::move(neighbors));
    vertices2.push_back(vertex, vertex->hook);
    ++num_vertices2;
    for (size_t i = 0; i < vertex->neighbors.size(); ++i) {
      Vertex1* neighbor = vertex->neighbors[i];
      neighbor->adjacency.push_back(vertex, vertex->adjacency_hooks[i]);
      ++neighbor->degree;
    }
    return vertex;
  }

  void remove_vertex2(Vertex2* vertex) {
    for (size_t i = 0; i < vertex->neighbors.size(); ++i) {
      Vertex1* neighbor = vertex->neighbors[i];
      --neighbor->degree;
      neighbor->adjacency.erase(vertex, vertex->adjacency_hooks[i]);
    }
    --num_vertices2;
    free_vertex2(vertex);
  }

  void remove_vertex1(Vertex1* vertex) {
    while (!vertex->adjacency.empty()) {
      remove_vertex2(vertex->adjacency.front());
    }
    --num_vertices1;
    free_vertex1(vertex);
  }

  void clear() {
    while (Vertex2* vertex = vertices2.front()) {
      free_vertex2(vertex);
    }
    num_vertices2 = 0;

    while (Vertex1* vertex = vertices1.front()) {
      free_vertex1(vertex);
    }
    num_vertices1 = 0;
  }

  void compute_vertex1_indices() const {
    size_t index = 0;
    for (Vertex1* vertex : vertices1) {
      vertex->index = index++;
    }
  }

  std::unique_ptr<Impl> clone() const {
    auto result = std::make_unique<Impl>(vertex1_property_layout, vertex2_property_layout);
    ankerl::unordered_dense::map<const Vertex1*, Vertex1*> map;
    map.reserve(num_vertices1);

    for (Vertex1* src : vertices1) {
      Vertex1* dst = result->add_vertex1();
      map.emplace(src, dst);
      vertex1_property_layout.destroy_and_copy_construct(dst, src);
    }

    for (Vertex2* src : vertices2) {
      std::vector<Vertex1*> neighbors;
      neighbors.reserve(src->neighbors.size());
      for (Vertex1* neighbor : src->neighbors) {
        neighbors.push_back(map.at(neighbor));
      }
      Vertex2* dst = result->add_vertex2(std::move(neighbors));
      vertex2_property_layout.destroy_and_copy_construct(dst, src);
    }

    return result;
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::make_size_tag(num_vertices1));
    ar(cereal::make_size_tag(num_vertices2));
    compute_vertex1_indices();
    for (Vertex2* vertex : vertices2) {
      std::vector<size_t> neighbor_indices;
      neighbor_indices.reserve(vertex->neighbors.size());
      for (Vertex1* neighbor : vertex->neighbors) {
        neighbor_indices.push_back(neighbor->index);
      }
      ar(cereal::make_nvp("neighbors", neighbor_indices));
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    clear();

    cereal::size_type vertex1_count;
    cereal::size_type vertex2_count;
    ar(cereal::make_size_tag(vertex1_count));
    ar(cereal::make_size_tag(vertex2_count));

    std::vector<Vertex1*> loaded_vertices1;
    loaded_vertices1.reserve(vertex1_count);
    for (size_t i = 0; i < vertex1_count; ++i) {
      Vertex1* vertex = add_vertex1();
      vertex->index = i;
      loaded_vertices1.push_back(vertex);
    }

    std::vector<size_t> neighbor_indices;
    for (size_t i = 0; i < vertex2_count; ++i) {
      ar(cereal::make_nvp("neighbors", neighbor_indices));
      std::vector<Vertex1*> neighbors;
      neighbors.reserve(neighbor_indices.size());
      for (size_t index : neighbor_indices) {
        neighbors.push_back(loaded_vertices1.at(index));
      }
      add_vertex2(std::move(neighbors));
    }
  }
};

BipartiteGraph::BipartiteGraph()
  : impl_(std::make_unique<Impl>()) {}

BipartiteGraph::BipartiteGraph(PropertyLayout vertex1_layout, PropertyLayout vertex2_layout)
  : impl_(std::make_unique<Impl>(vertex1_layout, vertex2_layout)) {}

BipartiteGraph::BipartiteGraph(const BipartiteGraph& other)
  : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

BipartiteGraph::BipartiteGraph(BipartiteGraph&& other) noexcept = default;

BipartiteGraph& BipartiteGraph::operator=(const BipartiteGraph& other) {
  if (this != &other) {
    impl_ = other.impl_ ? other.impl_->clone() : nullptr;
  }
  return *this;
}

BipartiteGraph& BipartiteGraph::operator=(BipartiteGraph&& other) noexcept = default;

BipartiteGraph::~BipartiteGraph() = default;

BipartiteGraph::Impl& BipartiteGraph::impl() {
  return *impl_;
}

const BipartiteGraph::Impl& BipartiteGraph::impl() const {
  return *impl_;
}

std::ranges::subrange<BipartiteGraph::vertex1_iterator> BipartiteGraph::vertices1() const {
  return {impl().vertices1.begin(), impl().vertices1.end()};
}

std::ranges::subrange<BipartiteGraph::vertex2_iterator> BipartiteGraph::vertices2() const {
  return {impl().vertices2.begin(), impl().vertices2.end()};
}

const IntrusiveList<BipartiteGraph::Vertex2>& BipartiteGraph::neighbors(Vertex1* u) const {
  assert(contains(u));
  return u->adjacency;
}

const std::vector<BipartiteGraph::Vertex1*>& BipartiteGraph::neighbors(Vertex2* u) const {
  assert(contains(u));
  return u->neighbors;
}

std::ranges::subrange<BipartiteGraph::out_edge1_iterator> BipartiteGraph::out_edges(Vertex1* u) const {
  const IntrusiveList<Vertex2>& adjacent = neighbors(u);
  return {out_edge1_iterator(adjacent.begin(), u), out_edge1_iterator(adjacent.end(), u)};
}

std::ranges::subrange<BipartiteGraph::in_edge1_iterator> BipartiteGraph::in_edges(Vertex1* u) const {
  const IntrusiveList<Vertex2>& adjacent = neighbors(u);
  return {in_edge1_iterator(adjacent.begin(), u), in_edge1_iterator(adjacent.end(), u)};
}

std::ranges::subrange<BipartiteGraph::out_edge2_iterator> BipartiteGraph::out_edges(Vertex2* u) const {
  const std::vector<Vertex1*>& adjacent = neighbors(u);
  return {out_edge2_iterator(adjacent.begin(), u), out_edge2_iterator(adjacent.end(), u)};
}

std::ranges::subrange<BipartiteGraph::in_edge2_iterator> BipartiteGraph::in_edges(Vertex2* u) const {
  const std::vector<Vertex1*>& adjacent = neighbors(u);
  return {in_edge2_iterator(adjacent.begin(), u), in_edge2_iterator(adjacent.end(), u)};
}

bool BipartiteGraph::contains(Vertex1* u) const {
  return u && u->impl == impl_.get();
}

bool BipartiteGraph::contains(Vertex2* u) const {
  return u && u->impl == impl_.get();
}

bool BipartiteGraph::contains(Vertex1* u, Vertex2* v) const {
  if (!contains(u) || !contains(v)) {
    return false;
  }
  return std::find(v->neighbors.begin(), v->neighbors.end(), u) != v->neighbors.end();
}

size_t BipartiteGraph::degree(Vertex1* u) const {
  assert(contains(u));
  return u->degree;
}

size_t BipartiteGraph::degree(Vertex2* u) const {
  assert(contains(u));
  return u->neighbors.size();
}

bool BipartiteGraph::empty() const {
  return impl().num_vertices1 == 0 && impl().num_vertices2 == 0;
}

size_t BipartiteGraph::num_vertices1() const {
  return impl().num_vertices1;
}

size_t BipartiteGraph::num_vertices2() const {
  return impl().num_vertices2;
}

OpaqueRef BipartiteGraph::property(Vertex1* u) {
  assert(contains(u));
  return impl().vertex1_property_layout.get(u);
}

OpaqueCref BipartiteGraph::property(Vertex1* u) const {
  assert(contains(u));
  return impl().vertex1_property_layout.get(static_cast<const Vertex1*>(u));
}

OpaqueRef BipartiteGraph::property(Vertex2* u) {
  assert(contains(u));
  return impl().vertex2_property_layout.get(u);
}

OpaqueCref BipartiteGraph::property(Vertex2* u) const {
  assert(contains(u));
  return impl().vertex2_property_layout.get(static_cast<const Vertex2*>(u));
}

BipartiteGraph::Vertex1* BipartiteGraph::add_vertex1() {
  return impl().add_vertex1();
}

BipartiteGraph::Vertex2* BipartiteGraph::add_vertex2(std::vector<Vertex1*> neighbors) {
  return impl().add_vertex2(std::move(neighbors));
}

void BipartiteGraph::remove_vertex1(Vertex1* u) {
  if (!contains(u)) {
    throw std::out_of_range("BipartiteGraph::remove_vertex1: vertex does not exist");
  }
  impl().remove_vertex1(u);
}

void BipartiteGraph::remove_vertex2(Vertex2* u) {
  if (!contains(u)) {
    throw std::out_of_range("BipartiteGraph::remove_vertex2: vertex does not exist");
  }
  impl().remove_vertex2(u);
}

void BipartiteGraph::clear() {
  impl().clear();
}

} // namespace libgm
