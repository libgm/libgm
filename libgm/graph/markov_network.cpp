#include "markov_network.hpp"

namespace libgm {

struct MarkovNetowrk::Vertex {
  Object property;
  NeighborMap neighbors;
};

struct MarkovNetwork::Impl : Object::Impl {
  VertexDataMap data;
  // boost::object_pool<Object> pool;
  size_t num_edges = 0;

  Impl* clone() const override {
    Impl* result = new Impl(*this);
    for (auto& [_, vertex] : result->data) {
      vertex = new Vertex(*vertex);
    }
    for (auto& [u, vertex] : result->data) {
      for (auto& [v, property] : vertex->neighbors) {
        if (u <= v) {
          property = new Object(*property);
          if (u != v) data.at(v)[u] = property;
        }
      }
    }
    return result;
  }

  bool compare(const Object::Impl& other) const override {
    const Impl& impl = static_cast<const Impl&>(other);
    if (data.size() != impl.data.size()) return false;
    if (num_edges != impl.num_edges) return false;

    for (auto [u, vertex] : data) {
      auto uit = impl.data.find(u);
      if (uit == impl.data.end()) return false;
      if (vertex->property != uit->second->property) return false;

      const NeighborMap& neighbors = uit->second->neighbors;
      for (auto [v, object] : vertex->neighbors) {
        auto eit = neighbors.find(v);
        if (eit == neighbors.end() || *object != *eit->second) return false;
      }
    }
    return true;
  }

  void save(oarchive& ar) const override {
    ar << data.size();
    ar << num_edges;
    for (auto [u, vertex] : data) {
      ar << u << vertex.property;
    }
    for (auto [u, vertex] : data) {
      for (auto [v, object] : vertex->neighbors) {
        if (u <= v) ar << u << v << *object;
      }
    }
  }

  void load(iarchive& ar) override {
    data.clear(); // TODO free edges
    size_t num_vertices, num_edges;
    Arg u, v;
    Object* object;
    ar >> num_vertices;
    ar >> num_edges;
    while (num_vertices-- > 0) {
      ar >> v;
      ar >> data[v].property;
    }
    while (num_edges-- > 0) {
      ar >> u >> v >> object;
      data[u].neighbors[v] = object;
      data[v].neighbors[u] = object;
    }
  }

  void print(std::ostream& out) const override {
    for (auto [v, vertex] : data) {
      out << v << ": " << vertex->property << "--" << sorted(v->neighbors) << std::endl;
    }
  }
};

MarkovNetwork()
  : Object(new Impl) {}

boost::iterator_range<vertex_iterator> MarkovNetwork::vertices() const {
  return { impl().data.begin(), impl().data.end() };
}

/// Returns the range of all edges in the graph.
boost::iterator_range<edge_iterator> MarkovNetwork::edges() const {
  return { { data_.begin(), data_.end() }, { data_.end(), data_.end() } };
}

boost::iterator_range<out_edge_iterator> MarkovNetwork::out_edges(Arg u) const {
  const Vertex& vertex = data(u);
  return { { vertex.neighbors.begin(), u }, { vertex.neighbors.end(), u } };
}

boost::iterator_range<in_edge_iterator> MarkovNetwork::in_edges(Arg u) const {
  const Vertex& vertex = data(u);
  return { { vertex.neighbors.begin(), u }, { vertex.neighbors.end(), u } };
}

boost::iterator_range<adjacency_iterator> MarkovNetwork::adjacent_vertices(Arg u) const {
  const Vertex& vertex = data(u);
  return { vertex.neighbors.begin(), vertex.neighbors.end() };
}

bool MarkovNetwork::contains(Arg u) const {
  return find(u).second;
}

bool MarkovNetwork::contains(Arg u, Arg v) const {
  auto [it, found] = find(u);
  return found && it->second.neighbors.count(v);
}

bool MarkovNetwork::contains(UndirectedEdge<Arg, Object> e) const {
  return contains(e.source(), e.target());
}

UndirectedEdge<Arg, Object> MarkovNetwork::edge(Arg u,  Arg v) const {
  return { u, v, data_.at(u).neighbors.at(v) };
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

Object& MarkovNetwork::operator[](UndirectedEdge<Arg, Object> e) {
  return *e.property;
}

const Object& MarkovNetwork::operator[](UndirectedEdge<Arg, Object> e) const {
  return *e.property;
}

Object& MarkovNetwork::operator()(Arg u, Arg v) {
  return *edge(u, v).first.property;
}
const Object& MarkovNetwork::operator()(Arg u, Arg v) const {
  return *edge(u, v).first.property;
}

bool MarkovNetwork::add_vertex(Arg u, Object object) {
  assert(u);
  if (contains(u)) {
    return false;
  } else {
    impl().data[u].property = std::move(object);
    return true;
  }
}

std::pair<UndirectedEdge<Arg, Object>, bool>
MarkovNetwork::add_edge(Arg u, Arg v, Object object = Object()) {
  assert(u);
  assert(v);
  auto [uit, ufound] = find(u);
  auto [vit, vfound] = find(v);
  assert(ufound && vfound);
  auto nbr = uit->second.neighbors.find(v);
  if (nbr != uit->second.neighbors.end()) {
    return { {u, v, nbr->second}, false };
  }
}

  Object* ptr = new Object(std::move(object));
  uit->second_>neighbors[v] = ptr;
  vit->second->neighbors[u] = ptr;
  ++impl().num_edges;
  return {{u,v, ptr}, true};
}

void MarkovNetwork::add_clique(const std::vector<Arg>& vertices) {
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
  // Look up the vertex data
  auto [uit, ufound] = find(u);
  auto [vit, vfound] = find(v);
  assert(ufound && vfound);

  // Look up the edge data
  auto eit = uit->neighbors.find(v);
  assert(eit != uit->neighbors.end());

  // delete the edge data and the edge itself
  delete it->second;
  uit->second->neighbors.erase(eit);
  vit->second->negihbors.erase(u);
  --impl().num_edges;
}

void MarkovNetwork::remove_edges(Arg u) {
  // Look up the vertex data
  auto [uit, ufound] = find(u);
  assert(ufound);

  // Delete the edge data and mirror edges
  for (auto [v, object] : uit->second->neighbors) {
    if(u != v) impl().data[v].neighbors.erase(u);
    delete object;
  }

  // Clear the neighbors
  impl().num_edges -= uit->second->neighbors.size();
  ut->second->neighbors.clear();
}

void MarkovNetwork::cremove_edges() {
  free_edge_data();
  for (auto [_, vertex] : impl().data) {
    vertex.neighbors.clear();
  }
  impl().num_edges = 0;
}

void MarkovNetwork::clear() {
  free_edge_data();
  impl().data.clear();
  impl().num_edges = 0;
}

void MarkovNetwork::eliminate(const EliminationStrategy& strategy, VertexVisitor visitor) {
  // Initialize the queue.
  MutableQueue<Arg, ptrdiff_t> queue;
  for (Arg u : vertices()) {
    queue.push(u, strategy.priority(u, *this));
  }

  // Reuse the affected vertices vector.
  std::vector<Arg> affected_vertices;
  while (!queue.empty()) {
    // The next vertex to be eliminated.
    Arg u = queue.pop().first;

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
        queue.update(v, strategy.priority(v, *this));
      }
    }
  }
}

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

} // namespace libgm
