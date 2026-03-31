#include "vector_graph.hpp"

#include <algorithm>
#include <iterator>
#include <ranges>

namespace libgm {

namespace {

class encoded_inserter {
public:
  using difference_type = std::ptrdiff_t;

  explicit encoded_inserter(std::vector<size_t>& output)
    : output_(&output) {}

  encoded_inserter& operator*() { return *this; }
  encoded_inserter& operator++() { return *this; }
  encoded_inserter operator++(int) { return *this; }

  encoded_inserter& operator=(size_t v) {
    output_->push_back((v << 1) | 1);
    return *this;
  }

private:
  std::vector<size_t>* output_;
};

} // namespace

std::ranges::subrange<VectorGraph::out_edge_iterator>
VectorGraph::out_edges(size_t u) const {
  auto neighbors = adjacent_vertices(u);
  return {
    out_edge_iterator(neighbors.begin(), u),
    out_edge_iterator(neighbors.end(), u)
  };
}

std::ranges::subrange<VectorGraph::in_edge_iterator>
VectorGraph::in_edges(size_t u) const {
  auto neighbors = adjacent_vertices(u);
  return {
    in_edge_iterator(neighbors.begin(), u),
    in_edge_iterator(neighbors.end(), u)
  };
}

std::ranges::subrange<VectorGraph::adjacency_iterator>
VectorGraph::adjacent_vertices(size_t u) const {
  const auto& neighbors = adjacency_.at(u);
  if (neighbors.empty()) {
    return {adjacency_iterator(), adjacency_iterator()};
  }
  return {
    adjacency_iterator(neighbors.begin()),
    adjacency_iterator(std::prev(neighbors.end()))
  };
}

void VectorGraph::add_clique(std::vector<size_t> vertices) {
  if (vertices.size() <= 1) {
    return;
  }

  std::ranges::sort(vertices);
  vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
  if (vertices.size() <= 1) {
    return;
  }

  for (size_t u : vertices) {
    const auto& current = adjacency_.at(u);
    std::vector<size_t> merged;
    merged.reserve(current.size() + vertices.size());

    auto clique = vertices | std::views::filter([u](size_t v) { return v != u; });
    std::ranges::set_union(adjacent_vertices(u), clique, encoded_inserter(merged));

    if (!merged.empty()) {
      merged.push_back(null_vertex());
    }
    adjacency_.at(u) = std::move(merged);
  }
}

void VectorGraph::mark_erased(size_t u, size_t v) {
  auto& neighbors = adjacency_.at(u);
  auto it = std::lower_bound(neighbors.begin(), neighbors.end(), encode_present(v));
  assert(it != neighbors.end() && *it == encode_present(v));
  *it = encode_erased(v);
}

size_t VectorGraph::clear_vertex(size_t u) {
  size_t removed = 0;
  for (size_t v : adjacent_vertices(u)) {
    mark_erased(v, u);
    ++removed;
  }
  adjacency_.at(u).clear();
  return removed;
}

size_t VectorGraph::remove_edge(size_t u, size_t v) {
  if (!contains(u, v)) {
    return 0;
  }
  mark_erased(u, v);
  mark_erased(v, u);
  return 1;
}

void VectorGraph::eliminate(const EliminationStrategy& strategy,
                            VertexVisitor visitor) {
  using Value = std::pair<ptrdiff_t, size_t>;
  using Heap = boost::heap::fibonacci_heap<Value>;

  Heap heap;
  std::vector<Heap::handle_type> handles(num_vertices());
  std::vector<size_t> affected_vertices;
  std::vector<bool> updated_vertices(num_vertices(), false);

  for (size_t u : vertices()) {
    handles[u] = heap.emplace(strategy.priority(u, *this), u);
  }

  while (!heap.empty()) {
    size_t u = heap.top().second;
    heap.pop();

    affected_vertices.clear();
    strategy.updated(u, *this, affected_vertices);

    visitor(u);
    std::vector<size_t> neighbors(adjacent_vertices(u).begin(), adjacent_vertices(u).end());
    add_clique(std::move(neighbors));
    clear_vertex(u);

    for (size_t v : affected_vertices) {
      if (!updated_vertices[v]) {
        updated_vertices[v] = true;
        if (v != u) {
          heap.update(handles[v], {strategy.priority(v, *this), v});
        }
      }
    }
    for (size_t v : affected_vertices) {
      updated_vertices[v] = false;
    }
  }
}

} // namespace libgm
