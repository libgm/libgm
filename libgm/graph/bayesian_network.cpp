
#include "bayesian_network.hpp"

#include <algorithm>
#include <cassert>
#include <new>
#include <stdexcept>
#include <vector>

namespace libgm {

struct BayesianNetwork::VertexData {
  Domain parents;
  AdjacencySet children;

#if 0
  bool equals(const VertexData& other) const {
    return parents == other.parents && property == other.property;
    // Intentionally omitting children
  }
#endif
};

struct BayesianNetwork::Impl {
  VertexDataMap data;
  size_t num_edges = 0;
  PropertyLayout property_layout;

  VertexData* allocate_vertex() const {
    return property_layout.allocate<VertexData>();
  }

  void free_vertex(VertexData* ptr) const {
    property_layout.free(ptr);
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::make_size_tag(data.size()));
    for (auto [u, ptr] : data) {
      ar(CEREAL_NVP(u), cereal::make_nvp("parents", ptr->parents));
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    clear();

    cereal::size_type n;
    ar(cereal::make_size_tag(n));
    for (size_t i = 0; i < n; ++i) {
      Arg u;
      Domain parents;
      ar(CEREAL_NVP(u), cereal::make_nvp("parents", parents));

      VertexData* ptr = allocate_vertex();
      ptr->parents = std::move(parents);
      data.emplace(u, ptr);
    }

    // Recreate the out-edges, and recompute the edge count.
    num_edges = 0;
    for (auto [v, ptr] : data) {
      num_edges += ptr->parents.size();
      for (Arg u : ptr->parents) {
        assert(u != v);
        data.at(u)->children.insert(v);
      }
    }
  }

  Impl() = default;
  explicit Impl(size_t count, PropertyLayout layout = {})
    : data(count),
      num_edges(0),
      property_layout(layout) {}

  ~Impl() { clear(); }

  std::unique_ptr<Impl> clone() const {
    auto result = std::make_unique<Impl>(data.size(), property_layout);
    result->num_edges = num_edges;
    for (auto [u, src] : data) {
      VertexData* dst = result->allocate_vertex();
      dst->parents = src->parents;
      dst->children = src->children;
      property_layout.destroy_and_copy_construct(dst, src);
      result->data.emplace(u, dst);
    }
    return result;
  }

#if 0
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
#endif

  void print(std::ostream& out) const {
    std::vector<std::pair<Arg, VertexData*>> values = data.values();
    std::sort(values.begin(), values.end());
    for (auto [u, ptr] : values) {
      out << u << ": " << ptr->parents << std::endl;
    }
  }

  void remove_in_edges(VertexDataMap::const_iterator it) {
    num_edges -= it->second->parents.size();
    for (Arg u : it->second->parents) {
      data.at(u)->children.erase(it->first);
    }
    it->second->parents.clear();
  }

  void clear() {
    for (auto [u, ptr] : data) {
      free_vertex(ptr);
    }
    data.clear();
    num_edges = 0;
  }

  std::pair<VertexDataMap::const_iterator, bool> find(Arg arg) const {
    auto it = data.find(arg);
    return {it, it != data.end()};
  }
};

BayesianNetwork::Impl& BayesianNetwork::impl() {
  return *impl_;
}

const BayesianNetwork::Impl& BayesianNetwork::impl() const {
  return *impl_;
}

BayesianNetwork::VertexData& BayesianNetwork::data(Arg arg) {
  return *impl().data.at(arg);
}

const BayesianNetwork::VertexData& BayesianNetwork::data(Arg arg) const {
  return *impl().data.at(arg);
}

BayesianNetwork::BayesianNetwork(size_t count)
  : impl_(std::make_unique<Impl>(count)) {}

BayesianNetwork::BayesianNetwork(size_t count, PropertyLayout layout)
  : impl_(std::make_unique<Impl>(count, layout)) {}

BayesianNetwork::BayesianNetwork(const BayesianNetwork& other)
  : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

BayesianNetwork::BayesianNetwork(BayesianNetwork&& other) noexcept = default;

BayesianNetwork& BayesianNetwork::operator=(const BayesianNetwork& other) {
  if (this != &other) {
    impl_ = other.impl_ ? other.impl_->clone() : nullptr;
  }
  return *this;
}

BayesianNetwork& BayesianNetwork::operator=(BayesianNetwork&& other) noexcept = default;

BayesianNetwork::~BayesianNetwork() = default;

std::ranges::subrange<BayesianNetwork::out_edge_iterator> BayesianNetwork::out_edges(Arg u) const {
  const AdjacencySet& children = data(u).children;
  return { out_edge_iterator(children.begin(), u), out_edge_iterator(children.end(), u) };
}

std::ranges::subrange<BayesianNetwork::in_edge_iterator> BayesianNetwork::in_edges(Arg u) const {
  const Domain& parents = data(u).parents;
  return { in_edge_iterator(parents.begin(), u), in_edge_iterator(parents.end(), u) };
}

std::ranges::subrange<BayesianNetwork::adjacency_iterator>
BayesianNetwork::adjacent_vertices(Arg u) const {
  const AdjacencySet& children = data(u).children;
  return { children.begin(), children.end() };
}

std::ranges::subrange<BayesianNetwork::vertex_iterator> BayesianNetwork::vertices() const {
  return { impl().data.begin(), impl().data.end() };
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
  if (it != impl().data.end() && it->second->children.contains(v)) {
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

OpaqueRef BayesianNetwork::property(Arg u) {
  return impl().property_layout.get(impl().data.at(u));
}

OpaqueCref BayesianNetwork::property(Arg u) const {
  return impl().property_layout.get(static_cast<const VertexData*>(impl().data.at(u)));
}

MarkovNetwork BayesianNetwork::markov_network() const {
  MarkovNetwork mn(num_vertices());
  for (auto [u, ptr] : impl().data) {
    mn.add_vertex(u);
    mn.add_clique(ptr->parents);
    mn.add_edges(u, ptr->parents);
  }
  return mn;
}

bool BayesianNetwork::add_vertex(Arg v, Domain parents) {
  // Check for self-loops and missing parent vertices.
  for (Arg u : parents) {
    if (u == v) {
      throw std::invalid_argument("BayesianNetwork::add_vertex: self-parent is not allowed");
    }
    if (!contains(u)) {
      throw std::out_of_range("BayesianNetwork::add_vertex: parent not found");
    }
  }

  // Check for duplicate parents
  Domain parents_copy(parents);
  parents_copy.unique();
  if (parents_copy.size() != parents.size()) {
    throw std::invalid_argument("BayesianNetwork::add_vertex: duplicate parent");
  }

  // Insert new vertex data or clear in-edges of the old one.
  auto [it, found] = impl().find(v);
  if (found) {
    impl().remove_in_edges(it);
  } else {
    it = impl().data.emplace(v, impl().allocate_vertex()).first;
  }

  // Update the edge count
  impl().num_edges += parents.size();

  // Insert the out-edges
  for (Arg u : parents) {
    data(u).children.insert(v);
  }

  // Update the vertex data
  it->second->parents = std::move(parents);
  return !found;
}

size_t BayesianNetwork::remove_vertex(Arg u) {
  auto [it, found] = impl().find(u);
  if (!found) {
    return 0;
  }
  if (!it->second->children.empty()) {
    throw std::logic_error("BayesianNetwork::remove_vertex: vertex has outgoing edges");
  }
  impl().remove_in_edges(it);
  impl().free_vertex(it->second);
  impl().data.erase(it);
  return 1;
}

void BayesianNetwork::remove_in_edges(Arg u) {
  auto [it, found]= impl().find(u);
  if (!found) {
    throw std::out_of_range("BayesianNetwork::remove_in_edges: vertex does not exist");
  }
  impl().remove_in_edges(it);
}

void BayesianNetwork::clear() {
  impl().clear();
}

} // namespace libgm
