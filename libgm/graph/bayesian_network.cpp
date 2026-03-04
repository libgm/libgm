
#include "bayesian_network.hpp"

#include <libgm/archives.hpp>

#include <algorithm>
#include <cassert>
#include <new>
#include <vector>

namespace libgm {

struct BayesianNetwork::VertexData {
  Domain parents;
  AdjacencySet children;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(parents);
  }

#if 0
  bool equals(const VertexData& other) const {
    return parents == other.parents && property == other.property;
    // Intentionally omitting children
  }
#endif
};

struct BayesianNetwork::Impl : Object::Impl {
  VertexDataMap data;
  size_t num_edges = 0;
  PropertyLayout property_layout;
  size_t property_offset = sizeof(VertexData);
  size_t vertex_allocation_size = sizeof(VertexData);

  void initialize_layout() {
    property_offset = property_layout.align_up(sizeof(VertexData));
    vertex_allocation_size = property_offset + property_layout.size;
  }

  void* property(VertexData* ptr) const {
    return reinterpret_cast<char*>(ptr) + property_offset;
  }

  const void* property(const VertexData* ptr) const {
    return reinterpret_cast<const char*>(ptr) + property_offset;
  }

  VertexData* allocate_vertex() const {
    void* buffer = ::operator new(vertex_allocation_size);
    VertexData* vertex_data = new (buffer) VertexData;
    if (property_layout.size != 0) {
      assert(property_layout.default_constructor);
      property_layout.default_constructor(property(vertex_data));
    }
    return vertex_data;
  }

  void destroy_property(VertexData* ptr) const {
    if (property_layout.size != 0) {
      assert(property_layout.deleter);
      property_layout.deleter(property(ptr));
    }
  }

  void free_vertex(VertexData* ptr) const {
    destroy_property(ptr);
    ptr->~VertexData();
    ::operator delete(ptr);
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
      property_layout(layout) {
    initialize_layout();
  }

  ~Impl() { clear(); }

  Impl* clone() const override {
    Impl* result = new Impl(data.size(),
                            property_layout);
    result->num_edges = num_edges;
    for (auto [u, src] : data) {
      VertexData* dst = result->allocate_vertex();
      dst->parents = src->parents;
      dst->children = src->children;
      if (property_layout.size != 0) {
        assert(property_layout.copy_constructor);
        result->destroy_property(dst);
        property_layout.copy_constructor(result->property(dst), property(src));
      }
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

  void print(std::ostream& out) const override {
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
  return static_cast<Impl&>(*impl_);
}

const BayesianNetwork::Impl& BayesianNetwork::impl() const {
  return static_cast<Impl&>(*impl_);
}

BayesianNetwork::VertexData& BayesianNetwork::data(Arg arg) {
  return *impl().data.at(arg);
}

const BayesianNetwork::VertexData& BayesianNetwork::data(Arg arg) const {
  return *impl().data.at(arg);
}

BayesianNetwork::BayesianNetwork(size_t count)
  : Object(std::make_unique<Impl>(count)) {}

BayesianNetwork::BayesianNetwork(size_t count, PropertyLayout layout)
  : Object(std::make_unique<Impl>(count, layout)) {}

SubRange<BayesianNetwork::out_edge_iterator> BayesianNetwork::out_edges(Arg u) const {
  const AdjacencySet& children = data(u).children;
  return { out_edge_iterator(children.begin(), u), out_edge_iterator(children.end(), u) };
}

SubRange<BayesianNetwork::in_edge_iterator> BayesianNetwork::in_edges(Arg u) const {
  const Domain& parents = data(u).parents;
  return { in_edge_iterator(parents.begin(), u), in_edge_iterator(parents.end(), u) };
}

SubRange<BayesianNetwork::adjacency_iterator>
BayesianNetwork::adjacent_vertices(Arg u) const {
  const AdjacencySet& children = data(u).children;
  return { children.begin(), children.end() };
}

SubRange<BayesianNetwork::vertex_iterator> BayesianNetwork::vertices() const {
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

void* BayesianNetwork::property(Arg u) {
  return impl().property(impl().data.at(u));
}

const void* BayesianNetwork::property(Arg u) const {
  return impl().property(impl().data.at(u));
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
  auto [it, found] = impl().find(v);

  // Insert new vertex data or clear in-edges of the old one.
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

void BayesianNetwork::remove_vertex(Arg u) {
  auto [it, found] = impl().find(u);
  assert(found && it->second->children.empty());
  impl().remove_in_edges(it);
  impl().free_vertex(it->second);
  impl().data.erase(it);
}

void BayesianNetwork::remove_in_edges(Arg u) {
  auto [it, found]= impl().find(u);
  assert(found);
  impl().remove_in_edges(it);
}

void BayesianNetwork::clear() {
  impl().clear();
}

} // namespace libgm

CEREAL_REGISTER_TYPE(libgm::BayesianNetwork::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::BayesianNetwork::Impl);
