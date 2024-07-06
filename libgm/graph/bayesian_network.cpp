
#include "bayesian_network.hpp"

namespace libgm {

struct BayesianNetwork::Vertex {
  Object property;
  Domain parents;
  AdjacencySet children;

  bool equals(const Vertex& other) const {
    return parents == other.parents && property == other.property;
    // Intentionally omitting children
  }
};

struct BayesianNetwork::Impl : Object::Impl {
  VertexDataMap data;
  size_t num_edges;

  Impl(size_t count) : data(count), num_edges(0) {}
  ~Impl() { clear(); }

  Impl* clone() const override {
    Impl* result = new Impl(*this);
    for (auto& [_, ptr] : result->data) {
      ptr = new Vertex(*ptr);
    }
  }

  bool equals(const Object::Impl& other) const override {
    const Impl& impl = static_cast<const Impl&>(other);
    if (data.size() != impl.data.size()) return false;
    if (num_edges != impl.num_edges) return false;
    for (const [u, ptr] : data) {
      auto it = impl.data.find(u);
      if (it == impl.data.end()) return false;
      if (!it->second->equals(*ptr)) return false;
    }
    return true;
  }

  void save(oarchive& ar) const override {
    ar << data.size() << num_edges;
    for (const auto [u, ptr] : impl().data) {
      ar << u << ptr->parents << ptr->property;
    }
  }

  void load(iarchive& ar) override {
    clear();
    size_t num_vertices;
    Arg u;
    ar >> num_vertices;
    ar >> num_edges;
    while (num_vertices-- > 0) {
      ar >> v;
      auto [it, inserted] = data.insert(v, new VertexData);
      assert(inserted);
      ar >> it->second->parents >> it->second->property;
      for (Arg u : it->second->parents) {
        assert(u != v);
        data[u].children.insert(v);
      }
    }
  }

  void print(std::ostream& out) const override {
    for (auto [u, ptr] : sorted(data)) {
      out << u << ": " << ptr->parents << " " << ptr->property << std::endl;
    }
  }

  void remove_in_edges(VertexDataMap::iterator it) {
    num_edges -= it->second->parents.size();
    for (Arg u : it->second->parents) {
      data.at(u).children.erase(it->first);
    }
    it->second->parents.clear();
  }

  void clear() {
    for (auto [u, ptr] : data) {
      delete ptr;
    }
    data.clear();
    num_edges = 0;
  }
};

BayesianNetwork::Impl& BayesianNetwork::impl() {
  return static_cast<Impl&>(*impl_);
}

const BayesianNetwork::Impl& BayesianNetwork::impl() const {
  return static_cast<Impl&>(*impl_);
}

BayesianNetwork::Vertex& BayesianNetwork::data(Arg arg) {
  return *impl().data.arg(arg);
}

const BayesianNetwork::Vertex& BayesianNetwork::data(Arg arg) const {
  return *impl().data.arg(arg);
}

const BayesianNetwork::Impl& BayesianNetwork::impl() const {
  return static_cast<Impl&>(*impl_);
}

BayesianNetwork::BayesianNetwork(size_t count)
  : Object(new Impl(count)) {}

boost::iterator_range<BayesianNetwork::vertex_iterator> BayesianNetwork::vertices() const {
  return { impl().data.begin(), impl().data_.end() };
}

boost::iterator_range<BayesianNetwork::edge_iterator> BayesianNetwork::edges() const {
  return { { data_.begin(), data_.end(), &VertexDataMap::children },
            { data_.end(), data_.end(), &VertexDataMap::children } };
}

/// Returns the edges incoming to a vertex.
boost::iterator_range<BayesianNetwork::in_edge_iterator> BayesianNetwork::in_edges(Arg u) const {
  const Domain& parents = data(u).parents;
  return { { parents.begin(), u }, { parents.end(), u } };
}

/// Returns the outgoing edges from a vertex.
boost::iterator_range<BayesianNetwork::out_edge_iterator> BayesianNetwork::out_edges(Arg u) const {
  const ArgSet& children = data(u).children;
  return { { children.begin(), u }, { children.end(), u } };
}

boost::iterator_range<BayesianNetwork::adjacency_iterator>
BayesianNetwork::adjacent_vertices(Arg u) const {
  const ArgSet& children = data(u).children;
  return { children.begin(), children.end() };
}

bool BayesianNetwork::contains(Arg u) const {
  return impl().data.find(u) != impl().data.end();
}

bool BayesianNetwork::contains(Arg u, Arg v) const {
  return bool(edge(u, v));
}

bool BayesianNetwork::contains(const DirectedEdge<Arg> e) const {
  return contains(e.source(), e.target());
}

DirectedEdge<Arg> BayesianNetwork::edge(Arg u, Arg v) const {
  auto it = impl().data.find(u);
  if (it != impl().data.end()) {
    return {u, v};
  } else {
    return {};
  }
}

size_t BayesianNetwork::in_degree(Arg u) const {
  return data(u).parents.size();
}

size_t BayesianNetwork::out_degree(Arg u) const {
  return data(u).children.size();
}

size_t BayesianNetwork::degree(Arg u) const {
  const VertexData& vertex_data = data(u);
  return vertex_data.parents.size() + vertex_data.children.size();
}

bool BayesianNetwork::empty() const {
  return impl().data.empty();
}

size_t BayesianNetwork::num_vertices() const {
  return impl().data.size();
}

size_t BayesianNetwork::num_edges() const {
  return impl().num_edges;
}

const Domain& BayesianNetwork::parents(Arg u) const {
  return data(u).parents;
}

Object& BayesianNetwork::operator[](Arg u) {
  return data(u).property;
}

const Object& BayesianNetwork::operator[](Arg u) const {
  return data(u).property;
}

MarkovNetwork<void, void> BayesianNetwork::markov_network() const {
  MarkovNetwork<void, void> mn(data_.size());
  for (Arg u :vertices()) {
    mn.add_vertex(u);
  }
  for (Arg u : vertices()) {
    mn.make_clique(u, parents(u));
  }
  return mn;
}

bool BayesianNetwork::add_vertex(Arg v, Domain parents, Object object = Object()) {
  auto [it, found] = find(v);

  // Insert new vertex data or clear in-edges of the old one
  if (found) {
    impl().remove_in_edges(it);
  } else {
    it = impl().data.emplace(u, new VertexData).first;
  }

  // Update the edge count
  impl().num_edges += parents.size();

  // Insert the out-edges
  for (Arg u : parents) {
    data(u).children.insert(v);
  }

  // Update the vertex data
  it->second->parents = std::move(parents);
  it->second->property = std::move(object);

  return !found;
}

void BayesianNetwork::remove_vertex(Arg u) {
  auto [it, found] = find(u);
  assert(found && it->children.empty());
  impl().remove_in_edges(it);
  delete it->second;
  impl().data.erase(it);
}

void BayesianNetwork::remove_in_edges(Arg u) {
  auto [it, found]= find(u);
  assert(found);
  impl().remove_in_edges(it);
}

void BayesianNetwork::clear() {
  impl().clear();
}

} // namespace libgm
