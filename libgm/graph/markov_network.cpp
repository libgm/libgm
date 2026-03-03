#include "markov_network.hpp"

#include <libgm/argument/domain.hpp>

#include <boost/heap/fibonacci_heap.hpp>

namespace libgm {

struct MarkovNetwork::VertexData {
  Object property;
  AdjacencyMap neighbors;

  template <typename ARCHIVE>
  void save(ARCHIVE& ar) {
    ar(CEREAL_NVP(property));

    // serialize the keys
    ar(cereal::make_size_tag(neighbors.size()));
    for (auto [v, _] : neighbors) {
      ar(v);
    }
  }

  template <typename ARCHIVE>
  void load(ARCHIVE& ar) {
    ar(CEREAL_NVP(property));

    // deerialize the keys
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

  template <typename ARCHIVE>
  void save(ARCHIVE& ar) const {
    // Serialize the adjacency
    ar(data);

    // Serialize the edge properties
    ar(cereal::make_size_tag(num_edges));
    for (auto [u, vertex] : data) {
      for (auto [v, property] : vertex->neighbors) {
        ar(CEREAL_NVP(u), CEREAL_NVP(v), cereal::make_nvp("property", *property));
      }
    }
  }

  template <typename ARCHIVE>
  void load(ARCHIVE& ar) {
    // Deserialize the adjacency
    ar(data);

    // Deserialize the edge properties
    cereal::size_type m;
    ar(cereal::make_size_tag(m));
    num_edges = m;
    for (size_t i = 0; i < num_edges; ++i) {
      Arg u, v;
      Object* property = new Object;
      ar(CEREAL_NVP(u), CEREAL_NVP(v), cereal::make_nvp("property", *property));
      data[u]->neighbors[v] = property;
      data[v]->neighbors[u] = property;
    }
  }

  Impl() = default;

  explicit Impl(size_t count) : data(count) {}

  Impl* clone() const override {
    Impl* result = new Impl(*this);
    for (auto& [_, vertex] : result->data) {
      vertex = new VertexData(*vertex);
    }
    for (auto& [u, vertex] : result->data) {
      for (auto& [v, property] : vertex->neighbors) {
        if (u <= v) {
          property = new Object(*property);
          if (u != v) data.at(v)->neighbors[u] = property;
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
      out << u << ": " << vertex->property << " -- " << neighbors << std::endl;
    }
  }

  std::pair<VertexDataMap::const_iterator, bool> find(Arg arg) const {
    auto it = data.find(arg);
    return {it, it != data.end()};
  }

  void free_edge_data() {
    for (auto [u, vertex] : data) {
      for (auto [v, object] : vertex->neighbors) {
        if (u <= v) delete object;
      }
    }
    num_edges = 0;
  }
};

MarkovNetwork::MarkovNetwork(size_t count)
  : Object(std::make_unique<Impl>(count)) {}

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

Object& MarkovNetwork::operator[](Arg u) {
  return data(u).property;
}

const Object& MarkovNetwork::operator[](Arg u) const {
  return data(u).property;
}

Object& MarkovNetwork::operator[](const UndirectedEdge<Arg>& e) {
  return *static_cast<Object*>(e.property());
}

const Object& MarkovNetwork::operator[](const UndirectedEdge<Arg>& e) const {
  return *static_cast<Object*>(e.property());
}

bool MarkovNetwork::add_vertex(Arg u, Object object) {
  assert(u != Arg());
  if (contains(u)) {
    return false;
  } else {
    impl().data[u]->property = std::move(object);
    return true;
  }
}

std::pair<UndirectedEdge<Arg>, bool> MarkovNetwork::add_edge(Arg u, Arg v, Object object) {
  auto [uit, ufound] = impl().find(u);
  auto [vit, vfound] = impl().find(v);
  assert(ufound && vfound);
  auto nbr = uit->second->neighbors.find(v);
  if (nbr != uit->second->neighbors.end()) {
    return { {u, v, nbr->second}, false };
  }

  Object* ptr = new Object(std::move(object));
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
  impl().data.erase(u);
}

void MarkovNetwork::remove_edge(Arg u, Arg v) {
  AdjacencyMap& neighbors_u = data(u).neighbors;
  AdjacencyMap& neighbors_v = data(v).neighbors;

  // Look up the edge data
  auto it = neighbors_u.find(v);
  assert(it != neighbors_u.end());

  // delete the edge data and the edge itself
  delete it->second;
  neighbors_u.erase(it);
  neighbors_v.erase(u);
  --impl().num_edges;
}

void MarkovNetwork::remove_edges(Arg u) {
  // Look up the vertex data
  AdjacencyMap& neighbors = data(u).neighbors;

  // Delete the edge data and mirror edges
  for (auto [v, object] : neighbors) {
    if(u != v) impl().data[v]->neighbors.erase(u);
    delete object;
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
  impl().data.clear();
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
