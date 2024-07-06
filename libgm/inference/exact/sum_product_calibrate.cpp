#include "sum_product_calibrate.hpp"

namespace libgm {

SumProductCalibrate::SumProductCalibrate()
  : calibrated_(true) { }

void SumProductCalibrate::reset(const ClusterGraph& cg) {
  calibrated_ = false;
  assert(cg.tree());
  jt_.clear();

  // initialize the cliques and edges
  ankerl::unordered_dense::map<vertex_descriptor, vertex_descriptor> map;
  for (vertex_descriptor v : cg.vertices()) {
    map.emplace(v, jt_.add_vertex(cg.cluster(v), cg[v].cast<Factor>(vtable)));
  }
  for (edge_descriptor e : cg.edges()) {
    edge_descriptor f = jt_.add_edge(map.at(e.source()), map.at(e.target()));
  }
}

void SumProductCalibrate::reset(
    MarkovNetworkT<>& mn, const EliminationStrategy& strategy, const ShapeMap& shape_map) {
  calibrated_ = false;

  // initialize the junction tree
  jt_.triangulated(mn, strategy);

  // Initialize the clique Factors
  for (id_t v : jt_.vertices()) {
    jt_[v] = Factor::ones(jt_.shape(v, shape_map), vt_);
  }
}

void SumProductCalibrate::multiply_in(const Domain& domain, const Factor& factor) {
  vertex_descriptor v = jt_.find_cluster_cover(domain);
  assert(v);
  jt_[v].multiply_in(factor, jt_.dims(v, domain));
  calibrated_ = false;
}

void SumProductCalibrate::calibrate() {
  jt_.mpp_traversal(nullptr, [&](edge_descriptor e) {
    Factor product = jt_[e.source()];
    for (edge_descriptor in : jt_.in_edges(e.source())) {
      if (in.source() != e.target()) {
        product.multiply_in(jt_[in](in), jt_.target_dims(in));
      }
    }
    jt_[e](e) = product.marginal(jt_.source_dims(e));
  });
  calibrated_ = true;
}

void SumProductCalibrate::normalize() {
  assert(calibrated_ && !jt_.empty());

  // Compute the normalization constant z, and normalize the root
  // and every message in the direction from the root
  vertex_descriptor root = jt_.root();
  Normalizer normalize = normalizer(root);
  normalize(jt_[root]);
  jt_.pre_order_traversal(root, [&normalize, this](edge_descriptor e) {
    normalize(jt_[e](e));
  });
}

void SumProductCalibrate::condition(const Assignment& a) {
  // Extract the restricted arguments
  Domain args = a.keys();

  // Update the factors and messages
  jt_.intersecting_clusters(args, [&](vertex_descriptor v) {
    Domain y, x; // restricted, retained
    jt_.cluster(v).partition(a, y, x);
    jt_[v] = jt_[v].restrict(jt_.dims(v, y), a.values(y));
    jt_.update_cluster(v, x);
  });
  jt_.intersecting_separators(args, [&](edge_descriptor e) {
    Domain y, x; // restricted, retained
    jt_.separator(e).partition(a, y, x);
    Dims dims = jt_.dims(e, y);
    Values values = a.values(y);
    jt_[e].forward = jt_[e].forward.restrict(dims, values);
    jt_[e].reverse = jt_[e].reverse.restrict(dims, values);
    jt_.update_separator(e, x);
  });

  // The junction tree needs to be calibrated afterwards
  calibrated_ = false;
}

Factor SumProductCalibrate::belief(vertex_descriptor v) const {
  assert(calibrated_);
  Factor result = jt_[v];
  for (edge_descriptor in : jt_.in_edges(v)) {
    result.multiply_in(jt_[in](in), in->target_dims(in));
  }
  return result;
}

Factor SumProductCalibrate::belief(edge_descriptor e) const {
  assert(calibrated_);
  return jt_[e].forward * jt_[e].reverse;
}

Factor SumProductCalibrate::belief(const Domain& domain) const {
  assert(calibrated_);

  // Try to find a separator that covers the variables
  edge_descriptor e = jt_.find_separator_cover(domain);
  if (e) {
    return belief(e).marginal(jt_.dims(e, domain));
  }

  // Next, look for a clique that covers the variables
  vertex_descriptor v = jt_.find_cluster_cover(domain);
  if (v) {
    return belief(v).marginal(jt_.dims(v, domain));
  }

  // Did not find a suitable clique / separator
  throw std::invalid_argument(
    "SumProductCalibrate::belief: the domain is not covered by any clique or separator"
  );
}

} // namespace libgm
