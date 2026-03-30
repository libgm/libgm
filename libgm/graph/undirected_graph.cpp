#include "undirected_graph.hpp"

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>

#include <ankerl/unordered_dense.h>

#include <cassert>
#include <stdexcept>
#include <vector>

namespace libgm {

struct UndirectedGraph::Vertex {
  const Impl* impl;
  size_t index = static_cast<size_t>(-1);
  boost::default_color_type color = boost::white_color;
  bool marked = false;
  size_t degree = 0;
  IntrusiveList<Edge> adjacency;
  IntrusiveList<Vertex>::Hook hook;

  template <typename Archive>
  void serialize(Archive&) {}

  explicit Vertex(Impl* impl)
    : impl(impl) {}
};

std::ostream& operator<<(std::ostream& out, UndirectedGraph::Vertex* v) {
  out << static_cast<void*>(v) << '(' << v->marked << ')';
  return out;
}

struct UndirectedGraph::Edge {
  IntrusiveEdge<Vertex, Edge>::Connectivity connectivity;
  Impl* impl;
  bool marked = false;
  IntrusiveList<Edge>::Hook hook;

  template <typename Archive>
  void save(Archive& ar) {
    ar(cereal::make_nvp("u", u()->index));
    ar(cereal::make_nvp("v", v()->index));
  }

  Edge(Vertex* u, Vertex* v, Impl* impl)
    : connectivity{u, v},
      impl(impl) {}

  ~Edge() {
    --u()->degree;
    --v()->degree;
  }

  Vertex* u() const {
    return connectivity.vertex[0];
  }

  Vertex* v() const {
    return connectivity.vertex[1];
  }
};

std::ostream& operator<<(std::ostream& out, UndirectedGraph::Edge* e) {
  out << static_cast<void*>(e) << '(' << e->marked << ')';
  return out;
}

static_assert(std::is_standard_layout_v<UndirectedGraph::Vertex>);
static_assert(std::is_standard_layout_v<UndirectedGraph::Edge>);

struct UndirectedGraph::Impl {
  IntrusiveList<Vertex> vertices;
  IntrusiveList<Edge> edges;
  size_t num_vertices = 0;
  size_t num_edges = 0;
  PropertyLayout vertex_property_layout;
  PropertyLayout edge_property_layout;

  Vertex* allocate_vertex() const {
    return vertex_property_layout.allocate<Vertex>(const_cast<Impl*>(this));
  }

  Edge* allocate_edge(Vertex* u, Vertex* v) const {
    return edge_property_layout.allocate<Edge>(u, v, const_cast<Impl*>(this));
  }

  Vertex* add_vertex() {
    Vertex* v = allocate_vertex();
    ++num_vertices;
    vertices.push_back(v, v->hook);
    return v;
  }

  Edge* add_edge(Vertex* u, Vertex* v) {
    Edge* e = allocate_edge(u, v);
    ++num_edges;
    edges.push_back(e, e->hook);
    u->adjacency.push_back(e, e->connectivity.adjacency_hook[0]);
    v->adjacency.push_back(e, e->connectivity.adjacency_hook[1]);
    ++u->degree;
    ++v->degree;
    return e;
  }

  void free_vertex(Vertex* v) const {
    vertex_property_layout.free(v);
  }

  void free_edge(Edge* e) const {
    edge_property_layout.free(e);
  }

  void compute_indices() {
    size_t i = 0;
    for (Vertex* vertex : vertices) {
      vertex->index = i++;
    }
  }

  template <typename Archive>
  void save(Archive& ar) {
    compute_indices();

    ar(cereal::make_size_tag(num_vertices));
    for (Vertex* vertex : vertices) {
      ar(*vertex);
    }

    ar(cereal::make_size_tag(num_edges));
    for (Edge* edge : edges) {
      ar(*edge);
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    clear();

    cereal::size_type size;
    ar(cereal::make_size_tag(size));
    num_vertices = size;
    for (size_t i = 0; i < num_vertices; ++i) {
      Vertex* vertex = allocate_vertex();
      ar(*vertex);
      vertices.push_back(vertex, vertex->hook);
    }

    std::vector<Vertex*> indexed_vertices(vertices.begin(), vertices.end());
    ar(cereal::make_size_tag(size));
    num_edges = size;
    for (size_t i = 0; i < num_edges; ++i) {
      size_t u_index, v_index;
      ar(cereal::make_nvp("u", u_index));
      ar(cereal::make_nvp("v", v_index));
      assert(u_index < indexed_vertices.size());
      assert(v_index < indexed_vertices.size());
      Edge* edge = allocate_edge(indexed_vertices[u_index], indexed_vertices[v_index]);
      edges.push_back(edge, edge->hook);
      edge->u()->adjacency.push_back(edge, edge->connectivity.adjacency_hook[0]);
      edge->v()->adjacency.push_back(edge, edge->connectivity.adjacency_hook[1]);
      ++edge->u()->degree;
      ++edge->v()->degree;
    }
  }

  Impl() = default;

  Impl(PropertyLayout vertex_layout, PropertyLayout edge_layout)
    : vertex_property_layout(vertex_layout),
      edge_property_layout(edge_layout) {}

  ~Impl() {
    clear();
  }

  std::unique_ptr<Impl> clone() const {
    auto result = std::make_unique<Impl>(vertex_property_layout, edge_property_layout);

    ankerl::unordered_dense::map<Vertex*, Vertex*> vmap;
    vmap.reserve(num_vertices);

    for (Vertex* src : vertices) {
      Vertex* dst = result->add_vertex();
      dst->color = src->color;
      dst->marked = src->marked;
      vertex_property_layout.destroy_and_copy_construct(dst, src);
      vmap.emplace(src, dst);
    }

    for (Edge* src : edges) {
      Vertex* u = vmap.at(src->u());
      Vertex* v = vmap.at(src->v());
      Edge* dst = result->add_edge(u, v);
      dst->marked = src->marked;
      edge_property_layout.destroy_and_copy_construct(dst, src);
    }

    return result;
  }

  void clear() {
    num_edges = 0;
    for (auto it = edges.begin(); it != edges.end();) {
      free_edge(*it++);
    }

    num_vertices = 0;
    for (auto it = vertices.begin(); it != vertices.end();) {
      free_vertex(*it++);
    }
  }
};

UndirectedGraph::UndirectedGraph()
  : impl_(std::make_unique<Impl>()) {}

UndirectedGraph::UndirectedGraph(PropertyLayout vertex_layout, PropertyLayout edge_layout)
  : impl_(std::make_unique<Impl>(vertex_layout, edge_layout)) {}

UndirectedGraph::UndirectedGraph(const UndirectedGraph& other)
  : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

UndirectedGraph::UndirectedGraph(UndirectedGraph&& other) noexcept = default;

UndirectedGraph& UndirectedGraph::operator=(const UndirectedGraph& other) {
  if (this != &other) {
    impl_ = other.impl_ ? other.impl_->clone() : nullptr;
  }
  return *this;
}

UndirectedGraph& UndirectedGraph::operator=(UndirectedGraph&& other) noexcept = default;

UndirectedGraph::~UndirectedGraph() = default;

UndirectedGraph::Impl& UndirectedGraph::impl() {
  return *impl_;
}

const UndirectedGraph::Impl& UndirectedGraph::impl() const {
  return *impl_;
}

UndirectedGraph::Vertex& UndirectedGraph::data(Vertex* u) {
  assert(u);
  return *u;
}

const UndirectedGraph::Vertex& UndirectedGraph::data(Vertex* u) const {
  assert(u);
  return *u;
}

UndirectedGraph::Edge& UndirectedGraph::data(edge_descriptor e) {
  assert(e);
  return *e.get();
}

const UndirectedGraph::Edge& UndirectedGraph::data(edge_descriptor e) const {
  assert(e);
  return *e.get();
}

void UndirectedGraph::set_marked(Vertex* v, bool value) {
  v->marked = value;
}

void UndirectedGraph::set_marked(edge_descriptor e, bool value) {
  e->marked = value;
}

UndirectedGraph::VertexIndexMap UndirectedGraph::vertex_index_map() {
  impl().compute_indices();
  return {};
}

std::ranges::subrange<UndirectedGraph::out_edge_iterator> UndirectedGraph::out_edges(Vertex* u) const {
  return cast_subrange<out_edge_iterator>(u->adjacency.entries());
}

std::ranges::subrange<UndirectedGraph::in_edge_iterator> UndirectedGraph::in_edges(Vertex* u) const {
  return cast_subrange<in_edge_iterator>(out_edges(u));
}

std::ranges::subrange<UndirectedGraph::adjacency_iterator> UndirectedGraph::adjacent_vertices(Vertex* u) const {
  return cast_subrange<adjacency_iterator>(out_edges(u));
}

std::ranges::subrange<UndirectedGraph::vertex_iterator> UndirectedGraph::vertices() const {
  return {impl().vertices.begin(), impl().vertices.end()};
}

std::ranges::subrange<UndirectedGraph::edge_iterator> UndirectedGraph::edges() const {
  return {impl().edges.begin(), impl().edges.end()};
}

bool UndirectedGraph::empty() const {
  return impl().vertices.empty();
}

bool UndirectedGraph::contains(Vertex* u) const {
  return u && u->impl == &impl();
}

bool UndirectedGraph::contains(Vertex* u, Vertex* v) const {
  if (!contains(u) || !contains(v)) {
    return false;
  }
  for (edge_descriptor e : out_edges(u)) {
    if (e.target() == v) {
      return true;
    }
  }
  return false;
}

bool UndirectedGraph::contains(edge_descriptor e) const {
  return e && e->impl == &impl();
}

size_t UndirectedGraph::num_vertices() const {
  return impl().num_vertices;
}

size_t UndirectedGraph::num_edges() const {
  return impl().num_edges;
}

size_t UndirectedGraph::in_degree(Vertex* u) const {
  return u->degree;
}

size_t UndirectedGraph::out_degree(Vertex* u) const {
  return u->degree;
}

size_t UndirectedGraph::degree(Vertex* u) const {
  return u->degree;
}

UndirectedGraph::Vertex* UndirectedGraph::root() const {
  return empty() ? nullptr : *vertices().begin();
}

bool UndirectedGraph::marked(Vertex* v) const {
  return v->marked;
}

bool UndirectedGraph::marked(edge_descriptor e) const {
  return e->marked;
}

OpaqueRef UndirectedGraph::property(Vertex* u) {
  return impl().vertex_property_layout.get(u);
}

OpaqueCref UndirectedGraph::property(Vertex* u) const {
  return impl().vertex_property_layout.get(static_cast<const Vertex*>(u));
}

OpaqueRef UndirectedGraph::property(edge_descriptor e) {
  return impl().edge_property_layout.get(e.get());
}

OpaqueCref UndirectedGraph::property(edge_descriptor e) const {
  return impl().edge_property_layout.get(static_cast<const Edge*>(e.get()));
}

bool UndirectedGraph::is_connected() {
  if (empty()) {
    return true;
  }

  size_t count = 0;
  struct Visitor : boost::default_bfs_visitor {
    size_t& count;

    explicit Visitor(size_t& count)
      : count(count) {}

    void discover_vertex(Vertex*, const UndirectedGraph&) {
      ++count;
    }
  } visitor(count);

  boost::queue<Vertex*> queue;
  boost::breadth_first_search(*this, *vertices().begin(), queue, visitor, vertex_color_map());
  return count == num_vertices();
}

bool UndirectedGraph::is_tree() {
  return num_edges() == num_vertices() - 1 && is_connected();
}

void UndirectedGraph::pre_order_traversal(Vertex* start, EdgeVisitor edge_visitor) {
  reset_color();

  struct Visitor : boost::default_dfs_visitor {
    EdgeVisitor visit_edge;

    explicit Visitor(EdgeVisitor visit_edge)
      : visit_edge(std::move(visit_edge)) {}

    void tree_edge(edge_descriptor e, const UndirectedGraph&) {
      visit_edge(e);
    }

    void black_target(edge_descriptor, const UndirectedGraph&) {
      throw std::invalid_argument("UndirectedGraph::pre_order_traversal: detected a loop");
    }
  } visitor(std::move(edge_visitor));

  boost::depth_first_visit(*this, start, visitor, vertex_color_map());
}

void UndirectedGraph::post_order_traversal(Vertex* start, EdgeVisitor edge_visitor) {
  reset_color();

  struct Visitor : boost::default_dfs_visitor {
    EdgeVisitor visit_edge;

    explicit Visitor(EdgeVisitor visit_edge)
      : visit_edge(std::move(visit_edge)) {}

    void finish_edge(edge_descriptor e, const UndirectedGraph&) {
      if (e.target()->color != boost::gray_color) {
        visit_edge(e.reverse());
      }
    }

    void black_target(edge_descriptor, const UndirectedGraph&) {
      throw std::invalid_argument("UndirectedGraph::post_order_traversal: detected a loop");
    }
  } visitor(std::move(edge_visitor));

  boost::depth_first_visit(*this, start, visitor, vertex_color_map());
}

void UndirectedGraph::mpp_traversal(Vertex* start, EdgeVisitor edge_visitor) {
  post_order_traversal(start, edge_visitor);
  pre_order_traversal(start, edge_visitor);
}

UndirectedGraph::Vertex* UndirectedGraph::add_vertex() {
  return impl().add_vertex();
}

UndirectedGraph::edge_descriptor UndirectedGraph::add_edge(Vertex* u, Vertex* v) {
  assert(u != v);
  return impl().add_edge(u, v);
}

void UndirectedGraph::remove_vertex(Vertex* u) {
  --impl().num_vertices;
  clear_vertex(u);
  impl().free_vertex(u);
}

size_t UndirectedGraph::remove_edge(edge_descriptor e) {
  if (!e || !contains(e)) {
    return 0;
  }
  --impl().num_edges;
  impl().free_edge(e.get());
  return 1;
}

size_t UndirectedGraph::remove_edge(Vertex* u, Vertex* v) {
  if (!contains(u) || !contains(v)) {
    return 0;
  }
  for (edge_descriptor e : out_edges(u)) {
    if (e.target() == v) {
      return remove_edge(e);
    }
  }
  return 0;
}

void UndirectedGraph::clear_vertex(Vertex* u) {
  impl().num_edges -= u->degree;
  for (auto it = u->adjacency.begin(); it != u->adjacency.end();) {
    impl().free_edge(*it++);
  }
}

void UndirectedGraph::remove_edges() {
  impl().num_edges = 0;
  for (auto it = impl().edges.begin(); it != impl().edges.end();) {
    impl().free_edge(*it++);
  }
}

void UndirectedGraph::clear() {
  remove_edges();

  impl().num_vertices = 0;
  for (auto it = impl().vertices.begin(); it != impl().vertices.end();) {
    impl().free_vertex(*it++);
  }
}

void UndirectedGraph::reset_color() {
  for (Vertex* v : vertices()) {
    v->color = boost::white_color;
  }
}

std::ostream& operator<<(std::ostream& out, const UndirectedGraph& graph) {
  std::cout << "UndirectedGraph([" << std::endl;
  for (UndirectedGraph::Vertex* v : graph.vertices()) {
    out << v << std::endl;
  }
  std::cout << "], [" << std::endl;
  for (UndirectedGraph::Edge* e : graph.edges()) {
    out << e << std::endl;
  }
  std::cout << "])" << std::endl;
  return out;
}

boost::default_color_type get(const UndirectedGraph::VertexColorMap&, UndirectedGraph::Vertex* v) {
  return v->color;
}

void put(const UndirectedGraph::VertexColorMap&, UndirectedGraph::Vertex* v, boost::default_color_type c) {
  v->color = c;
}

size_t get(const UndirectedGraph::VertexIndexMap&, UndirectedGraph::Vertex* v) {
  assert(v);
  return v->index;
}

} // namespace libgm
