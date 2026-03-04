#include "markov_network.hpp"

#include <libgm/argument/domain.hpp>

#include <boost/heap/fibonacci_heap.hpp>

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace libgm {

struct MarkovNetwork::VertexData {
  AdjacencyMap neighbors;

  template <typename ARCHIVE>
  void save(ARCHIVE& ar) {
    // serialize the keys
    ar(cereal::make_size_tag(neighbors.size()));
    for (auto [v, _] : neighbors) {
      ar(v);
    }
  }

  template <typename ARCHIVE>
  void load(ARCHIVE& ar) {
    // deserialize the keys
    cereal::size_type degree;
    ar(cereal::make_size_tag(degree));
    for (size_t i = 0; i < degree; ++i) {
      Arg v;
      ar(v);
      neighbors.emplace(v, nullptr);
    }
  }
};

struct MarkovNetwork::Impl : Object::Impl {
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

  void* allocate_edge_property() const {
    if (edge_property_layout.size == 0) return nullptr;
    assert(edge_property_layout.default_constructor);
    void* ptr = ::operator new(edge_property_layout.size);
    edge_property_layout.default_constructor(ptr);
    return ptr;
  }

  void* copy_edge_property(const void* src) const {
    if (edge_property_layout.size == 0) return nullptr;
    assert(src);
    assert(edge_property_layout.copy_constructor);
    void* dst = ::operator new(edge_property_layout.size);
    edge_property_layout.copy_constructor(dst, src);
    return dst;
  }

  void free_edge_property(void* ptr) const {
    if (edge_property_layout.size == 0) {
      assert(ptr == nullptr);
      return;
    }
    assert(ptr);
    assert(edge_property_layout.deleter);
    edge_property_layout.deleter(ptr);
    ::operator delete(ptr);
  }

  template <typename ARCHIVE>
  void save(ARCHIVE& ar) const {
    if (vertex_property_layout.size != 0 || edge_property_layout.size != 0) {
      throw std::logic_error("Serializing MarkovNetwork properties is unsupported.");
    }
    ar(data);
  }

  template <typename ARCHIVE>
  void load(ARCHIVE& ar) {
    if (vertex_property_layout.size != 0 || edge_property_layout.size != 0) {
      throw std::logic_error("Deserializing MarkovNetwork properties is unsupported.");
    }
    free_edge_data();
    free_vertex_data();
    ar(data);
    num_edges = 0;
    for (auto [u, vertex] : data) {
      for (auto& [v, property] : vertex->neighbors) {
        property = nullptr;
        if (u <= v) ++num_edges;
      }
    }
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

  Impl* clone() const override {
    Impl* result = new Impl(data.size(), vertex_property_layout, edge_property_layout);
    result->num_edges = num_edges;

    for (auto [u, vertex] : data) {
      VertexData* dst = result->allocate_vertex();
      if (vertex_property_layout.size != 0) {
        assert(vertex_property_layout.copy_constructor);
        assert(vertex_property_layout.deleter);
        result->vertex_property_layout.deleter(result->vertex_property(dst));
        vertex_property_layout.copy_constructor(result->vertex_property(dst), vertex_property(vertex));
      }
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
          void* copied = result->copy_edge_property(property);
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
  bool compare(const Object::Impl& other) const override {
    const Impl& impl = static_cast<const Impl&>(other);
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

  void print(std::ostream& out) const override {
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
        if (u <= v) free_edge_property(object);
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
  : Object(std::make_unique<Impl>(count)) {}

MarkovNetwork::MarkovNetwork(size_t count, PropertyLayout vertex_layout, PropertyLayout edge_layout)
  : Object(std::make_unique<Impl>(count, vertex_layout, edge_layout)) {}

MarkovNetwork::Impl& MarkovNetwork::impl() {
  return static_cast<Impl&>(*impl_);
}

const MarkovNetwork::Impl& MarkovNetwork::impl() const {
  return static_cast<const Impl&>(*impl_);
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

// SubRange<edge_iterator> MarkovNetwork::edges() const {
//   return { { data_.begin(), data_.end() }, { data_.end(), data_.end() } };
// }

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

void* MarkovNetwork::property(Arg u) {
  return impl().vertex_property(impl().data.at(u));
}

const void* MarkovNetwork::property(Arg u) const {
  return impl().vertex_property(impl().data.at(u));
}

void* MarkovNetwork::property(const UndirectedEdge<Arg>& e) {
  return e.property();
}

const void* MarkovNetwork::property(const UndirectedEdge<Arg>& e) const {
  return e.property();
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

  void* ptr = impl().allocate_edge_property();
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

void MarkovNetwork::remove_vertex(Arg u) {
  remove_edges(u);
  impl().free_vertex(impl().data.at(u));
  impl().data.erase(u);
}

void MarkovNetwork::remove_edge(Arg u, Arg v) {
  AdjacencyMap& neighbors_u = data(u).neighbors;
  AdjacencyMap& neighbors_v = data(v).neighbors;

  // Look up the edge data
  auto it = neighbors_u.find(v);
  assert(it != neighbors_u.end());

  // delete the edge data and the edge itself
  if (u <= v) {
    impl().free_edge_property(it->second);
  }
  neighbors_u.erase(it);
  neighbors_v.erase(u);
  --impl().num_edges;
}

void MarkovNetwork::remove_edges(Arg u) {
  // Look up the vertex data
  AdjacencyMap& neighbors = data(u).neighbors;

  // Delete the edge data and mirror edges
  for (auto [v, object] : neighbors) {
    if (u <= v) {
      impl().free_edge_property(object);
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
    strategy.update(u, *this, affected_vertices);

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

#if 0
class edge_iterator
  : public std::iterator<std::forward_iterator_tag, edge_type> {
public:
  using reference = edge_type;
  using outer_iterator = typename VertexDataMap::const_iterator;
  using inner_iterator = typename neighbor_map::const_iterator;

  edge_iterator() {}

  edge_iterator(outer_iterator it1, outer_iterator end1)
    : it1_(it1), end1_(end1) {
    find_next();
  }

  edge_type operator*() const {
    return edge_type(it1_->first, it2_->first, it2_->second);
  }

  edge_iterator& operator++() {
    do {
      ++it2_;
    } while (it2_ != it1_->second.neighbors.end() &&
              it2_->first < it1_->first);
    if (it2_ == it1_->second.neighbors.end()) {
      ++it1_;
      find_next();
    }
    return *this;
  }

  edge_iterator operator++(int) {
    edge_iterator copy = *this;
    operator++();
    return copy;
  }

  bool operator==(const edge_iterator& o) const {
    return
      (it1_ == end1_ && o.it1_ == o.end1_) ||
      (it1_ == o.it1_ && it2_ == o.it2_);
  }

  bool operator!=(const edge_iterator& other) const {
    return !(operator==(other));
  }

private:
  /// find the next non-empty neighbor map with it1_->firstt <= it2_->first
  void find_next() {
    while (it1_ != end1_) {
      it2_ = it1_->second.neighbors.begin();
      while (it2_ != it1_->second.neighbors.end() &&
              it2_->first < it1_->first) {
        ++it2_;
      }
      if (it2_ != it1_->second.neighbors.end()) {
        break;
      } else {
        ++it1_;
      }
    }
  }

  outer_iterator it1_;  ///< the iterator to the vertex data
  outer_iterator end1_; ///< the iterator past the last vertex data
  inner_iterator it2_;  ///< the iterator to the current neighbor

}; // class edge_iterator
#endif

} // namespace libgm
