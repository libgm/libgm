#include "markov_network.hpp"

#include <libgm/argument/domain.hpp>

#include <boost/heap/fibonacci_heap.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

namespace libgm {

struct MarkovNetwork::VertexData {
  AdjacencyMap neighbors;
};

struct MarkovNetwork::Impl {
  VertexDataMap data;
  size_t num_edges = 0;
  PropertyLayout vertex_property_layout;
  PropertyLayout edge_property_layout;
  size_t vertex_property_offset = sizeof(VertexData);
  size_t vertex_allocation_size = sizeof(VertexData);

  void initialize_layout() {
    vertex_property_offset = vertex_property_layout.align_up(sizeof(VertexData));
    vertex_allocation_size = vertex_property_offset + vertex_property_layout.size;
  }

  void* vertex_property(VertexData* vertex) const {
    return reinterpret_cast<char*>(vertex) + vertex_property_offset;
  }

  const void* vertex_property(const VertexData* vertex) const {
    return reinterpret_cast<const char*>(vertex) + vertex_property_offset;
  }

  VertexData* allocate_vertex() const {
    void* buffer = ::operator new(vertex_allocation_size);
    VertexData* vertex = new (buffer) VertexData;
    if (vertex_property_layout.size != 0) {
      assert(vertex_property_layout.default_constructor);
      vertex_property_layout.default_constructor(vertex_property(vertex));
    }
    return vertex;
  }

  void free_vertex(VertexData* vertex) const {
    if (vertex_property_layout.size != 0) {
      assert(vertex_property_layout.deleter);
      vertex_property_layout.deleter(vertex_property(vertex));
    }
    vertex->~VertexData();
    ::operator delete(vertex);
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::make_size_tag(data.size()));
    for (auto [u, _] : data) {
      ar(u);
    }

    ar(cereal::make_size_tag(num_edges));
    for (auto [u, vertex] : data) {
      for (auto [v, _] : vertex->neighbors) {
        if (u <= v) {
          ar(u, v);
        }
      }
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    free_edge_data();
    free_vertex_data();

    cereal::size_type vertex_count;
    ar(cereal::make_size_tag(vertex_count));
    for (size_t i = 0; i < vertex_count; ++i) {
      Arg u;
      ar(u);
      data.emplace(u, allocate_vertex());
    }

    cereal::size_type edge_count;
    ar(cereal::make_size_tag(edge_count));
    for (size_t i = 0; i < edge_count; ++i) {
      Arg u, v;
      ar(u, v);
      auto uit = data.find(u);
      auto vit = data.find(v);
      assert(uit != data.end());
      assert(vit != data.end());

      void* ptr = edge_property_layout.allocate_default_constructed();
      uit->second->neighbors.emplace(v, ptr);
      vit->second->neighbors.emplace(u, ptr);
      ++num_edges;
    }
    assert(num_edges == edge_count);
  }

  Impl() = default;

  explicit Impl(size_t count, PropertyLayout vertex_layout = {}, PropertyLayout edge_layout = {})
    : data(count),
      vertex_property_layout(vertex_layout),
      edge_property_layout(edge_layout) {
    initialize_layout();
  }

  ~Impl() {
    free_edge_data();
    free_vertex_data();
  }

  std::unique_ptr<Impl> clone() const {
    auto result = std::make_unique<Impl>(data.size(), vertex_property_layout, edge_property_layout);
    result->num_edges = num_edges;

    for (auto [u, vertex] : data) {
      VertexData* dst = result->allocate_vertex();
      vertex_property_layout.destroy_and_copy_construct(result->vertex_property(dst),
                                                       vertex_property(vertex));
      result->data.emplace(u, dst);
    }

    for (auto [u, vertex] : data) {
      for (auto [v, _] : vertex->neighbors) {
        result->data.at(u)->neighbors.emplace(v, nullptr);
      }
    }

    for (auto [u, vertex] : data) {
      for (auto [v, property] : vertex->neighbors) {
        if (u <= v) {
          void* copied = edge_property_layout.allocate_copy_constructed(property);
          result->data.at(u)->neighbors.at(v) = copied;
          if (u != v) {
            result->data.at(v)->neighbors.at(u) = copied;
          }
        }
      }
    }
    return result;
  }

#if 0
  bool compare(const Impl& impl) const {
    if (data.size() != impl.data.size()) return false;
    if (num_edges != impl.num_edges) return false;

    for (auto [u, vertex] : data) {
      auto uit = impl.data.find(u);
      if (uit == impl.data.end()) return false;
      if (vertex->property != uit->second->property) return false;

      const AdjacencyMap& neighbors = uit->second->neighbors;
      for (auto [v, object] : vertex->neighbors) {
        auto eit = neighbors.find(v);
        if (eit == neighbors.end() || *object != *eit->second) return false;
      }
    }
    return true;
  }
#endif

  void print(std::ostream& out) const {
    std::vector<std::pair<Arg, VertexData*>> values = data.values();
    std::sort(values.begin(), values.end());
    for (auto [u, vertex] : values) {
      Domain neighbors;
      for (auto [v, property] : vertex->neighbors) neighbors.push_back(v);
      neighbors.sort();
      out << u << ": " << neighbors << std::endl;
    }
  }

  std::pair<VertexDataMap::const_iterator, bool> find(Arg arg) const {
    auto it = data.find(arg);
    return {it, it != data.end()};
  }

  void free_edge_data() {
    for (auto [u, vertex] : data) {
      for (auto [v, object] : vertex->neighbors) {
        if (u <= v) edge_property_layout.free_allocated(object);
      }
    }
    num_edges = 0;
  }

  void free_vertex_data() {
    for (auto [_, vertex] : data) {
      free_vertex(vertex);
    }
    data.clear();
  }
};

MarkovNetwork::MarkovNetwork(size_t count)
  : impl_(std::make_unique<Impl>(count)) {}

MarkovNetwork::MarkovNetwork(size_t count, PropertyLayout vertex_layout, PropertyLayout edge_layout)
  : impl_(std::make_unique<Impl>(count, vertex_layout, edge_layout)) {}

MarkovNetwork::MarkovNetwork(const MarkovNetwork& g)
  : impl_(g.impl_ ? g.impl_->clone() : nullptr) {}

MarkovNetwork::MarkovNetwork(MarkovNetwork&& g) noexcept = default;

MarkovNetwork& MarkovNetwork::operator=(const MarkovNetwork& g) {
  if (this != &g) {
    impl_ = g.impl_ ? g.impl_->clone() : nullptr;
  }
  return *this;
}

MarkovNetwork& MarkovNetwork::operator=(MarkovNetwork&& g) noexcept = default;

MarkovNetwork::~MarkovNetwork() = default;

MarkovNetwork::Impl& MarkovNetwork::impl() {
  return *impl_;
}

const MarkovNetwork::Impl& MarkovNetwork::impl() const {
  return *impl_;
}

MarkovNetwork::VertexData& MarkovNetwork::data(Arg arg) {
  return *impl().data.at(arg);
}

const MarkovNetwork::VertexData& MarkovNetwork::data(Arg arg) const {
  return *impl().data.at(arg);
}

SubRange<MarkovNetwork::out_edge_iterator> MarkovNetwork::out_edges(Arg u) const {
  const AdjacencyMap& neighbors = data(u).neighbors;
  return { out_edge_iterator(neighbors.begin(), u), out_edge_iterator(neighbors.end(), u) };
}

SubRange<MarkovNetwork::in_edge_iterator> MarkovNetwork::in_edges(Arg u) const {
  const AdjacencyMap& neighbors = data(u).neighbors;
  return { in_edge_iterator(neighbors.begin(), u), in_edge_iterator(neighbors.end(), u) };
}

SubRange<MarkovNetwork::adjacency_iterator> MarkovNetwork::adjacent_vertices(Arg u)
const {
  const AdjacencyMap& neighbors = data(u).neighbors;
  return { neighbors.begin(), neighbors.end() };
}

SubRange<MarkovNetwork::vertex_iterator> MarkovNetwork::vertices() const {
  return { impl().data.begin(), impl().data.end() };
}

bool MarkovNetwork::contains(Arg u) const {
  return impl().find(u).second;
}

bool MarkovNetwork::contains(Arg u, Arg v) const {
  auto [it, found] = impl().find(u);
  return found && it->second->neighbors.contains(v);
}

bool MarkovNetwork::contains(const UndirectedEdge<Arg>& e) const {
  return contains(e.source(), e.target());
}

UndirectedEdge<Arg> MarkovNetwork::edge(Arg u, Arg v) const {
  return { u, v, data(u).neighbors.at(v) };
}

size_t MarkovNetwork::out_degree(Arg u) const {
  return data(u).neighbors.size();
}

size_t MarkovNetwork::in_degree(Arg u) const {
  return data(u).neighbors.size();
}

size_t MarkovNetwork::degree(Arg u) const {
  return data(u).neighbors.size();
}

bool MarkovNetwork::empty() const {
  return impl().data.size() == 0;
}

size_t MarkovNetwork::num_vertices() const {
  return impl().data.size();
}

size_t MarkovNetwork::num_edges() const {
  return impl().num_edges;
}

OpaqueRef MarkovNetwork::property(Arg u) {
  return {impl().vertex_property_layout.type_info,
          impl().vertex_property(impl().data.at(u))};
}

OpaqueCref MarkovNetwork::property(Arg u) const {
  return {impl().vertex_property_layout.type_info,
          impl().vertex_property(impl().data.at(u))};
}

OpaqueRef MarkovNetwork::property(const UndirectedEdge<Arg>& e) {
  return {impl().edge_property_layout.type_info, e.property()};
}

OpaqueCref MarkovNetwork::property(const UndirectedEdge<Arg>& e) const {
  return {impl().edge_property_layout.type_info, e.property()};
}

bool MarkovNetwork::add_vertex(Arg u) {
  assert(u != Arg());
  if (contains(u)) {
    return false;
  } else {
    impl().data.emplace(u, impl().allocate_vertex());
    return true;
  }
}

std::pair<UndirectedEdge<Arg>, bool> MarkovNetwork::add_edge(Arg u, Arg v) {
  auto [uit, ufound] = impl().find(u);
  auto [vit, vfound] = impl().find(v);
  assert(ufound && vfound);
  auto nbr = uit->second->neighbors.find(v);
  if (nbr != uit->second->neighbors.end()) {
    return { {u, v, nbr->second}, false };
  }

  void* ptr = impl().edge_property_layout.allocate_default_constructed();
  uit->second->neighbors[v] = ptr;
  vit->second->neighbors[u] = ptr;
  ++impl().num_edges;
  return { {u, v, ptr}, true};
}

void MarkovNetwork::add_edges(Arg u, const std::vector<Arg>& vs) {
  for (Arg v : vs) {
    add_edge(u, v);
  }
}

void MarkovNetwork::add_clique(const Domain& vertices) {
  for (Arg arg : vertices) add_vertex(arg);
  auto it1 = vertices.begin(), end = vertices.end();
  for (; it1 != end; ++it1) {
    for (auto it2 = std::next(it1); it2 != end; ++it2) {
      add_edge(*it1, *it2);
    }
  }
}

size_t MarkovNetwork::remove_vertex(Arg u) {
  auto it = impl().data.find(u);
  if (it == impl().data.end()) {
    return 0;
  }

  remove_edges(u);
  impl().free_vertex(it->second);
  impl().data.erase(it);
  return 1;
}

size_t MarkovNetwork::remove_edge(Arg u, Arg v) {
  auto [uit, ufound] = impl().find(u);
  auto [vit, vfound] = impl().find(v);
  if (!ufound || !vfound) {
    return 0;
  }
  AdjacencyMap& neighbors_u = uit->second->neighbors;
  AdjacencyMap& neighbors_v = vit->second->neighbors;

  // Look up the edge data
  auto it = neighbors_u.find(v);
  if (it == neighbors_u.end()) {
    return 0;
  }

  // delete the edge data and the edge itself
  impl().edge_property_layout.free_allocated(it->second);
  neighbors_u.erase(it);
  neighbors_v.erase(u);
  --impl().num_edges;
  return 1;
}

void MarkovNetwork::remove_edges(Arg u) {
  // Look up the vertex data
  AdjacencyMap& neighbors = data(u).neighbors;

  // Delete the edge data and mirror edges
  for (auto [v, object] : neighbors) {
    if (u <= v) {
      impl().edge_property_layout.free_allocated(object);
    }
    if(u != v) impl().data[v]->neighbors.erase(u);
  }

  // Clear the neighbors
  impl().num_edges -= neighbors.size();
  neighbors.clear();
}

void MarkovNetwork::remove_edges() {
  impl().free_edge_data();
  for (auto [_, vertex] : impl().data) {
    vertex->neighbors.clear();
  }
}

void MarkovNetwork::clear() {
  impl().free_edge_data();
  impl().free_vertex_data();
}

void MarkovNetwork::eliminate(const EliminationStrategy& strategy, VertexVisitor visitor) {
  // Construct the heap and a map from arguments to handle used to update the heap.
  using Value = std::pair<ptrdiff_t, Arg>;
  using Heap = boost::heap::fibonacci_heap<Value>;
  Heap heap;
  ankerl::unordered_dense::map<Arg, Heap::handle_type> handles;

  // Initialize the heap and the map
  for (Arg u : vertices()) {
    Heap::handle_type handle = heap.emplace(strategy.priority(u, *this), u);
    handles.emplace(u, handle);
  }

  // Reuse the affected vertices vector.
  std::vector<Arg> affected_vertices;
  while (!heap.empty()) {
    // The next vertex to be eliminated.
    Arg u = heap.top().second;
    heap.pop();
    handles.erase(u);

    // Find out vertices whose priority may change
    affected_vertices.clear();
    strategy.updated(u, *this, affected_vertices);

    // Eliminate the vertex
    visitor(u);
    add_clique(adjacent_vertices(u));
    remove_edges(u);

    // Update the priorities
    for (Arg v : affected_vertices) {
      if (v != u) {
        ptrdiff_t priority = strategy.priority(v, *this);
        heap.update(handles.at(v), {priority, v});
      }
    }
  }
}

std::ostream& operator<<(std::ostream& out, const MarkovNetwork& mn) {
  mn.impl().print(out);
  return out;
}

} // namespace libgm
