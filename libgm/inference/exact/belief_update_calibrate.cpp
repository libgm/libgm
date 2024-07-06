namespace libgm {

void BeliefUpdateCalibrate::BeliefUpdateCalibrate(const ClusterGraphT<Potential, Potential>& jt) {
  jt_ = jt;
}

void BeliefUpdateCalibrate::reset(
    MarkovNetwork& mn, const EliminationStrategy& strategy, const ShapeMap& shape_map) {
  // compute the junction tree for the given factors
  jt_.triangulated(mn, strategy);

  // intialize the clique and separator potentials to unity
  for (vertex_descriptor v : jt_.vertices()) {
    jt_[v] = Potential::one(jt_.shape(v, shape_map));
  }
  for (edge_descriptor e : jt_.edges()) {
    jt_[e] = Potential::one(jt_.shape(e, shape_map));
  }
}

void BeliefUpdateCalibrate::multiply_in(const Domain& domain, Potential factor) {
  vertex_descriptor v = jt_.find_cluster_cover(domain);
  assert(v);
  jt_[v].multiply_in(jt_.dims(v, domain), factor);
}

void BeliefUpdateCalibrate::calibrate() {
  jt_.mpp_traversal(nullptr, [&](edge_descriptor e) {
    jt_[e.target()].divide_in(jt_[e], jt_.target_dims(e));
    jt_[e] = jt_[e.source()].marginal(jt_.source_dims(e));
    jt_[e.target()].multiply_in(jt_[e], jt_.target_dims(e));
  });
}

void BeliefUpdateCalibrate::normalize() {
  for (vertex_descriptor v : jt_.vertices()) {
    jt_[v].normalize();
  }
  for (edge_descriptor e : jt_.edges()) {
    jt_[e].normalize();
  }
}

void BeliefUpdateCalibrate::condition(const Assignment& a) {
  // Extract the restricted arguments
  Domain args = a.keys();

  // Update the factors and messages
  jt_.intersecting_clusters(args, [&](vertex_descriptor v) {
    Domain y, x; // restricted, retained
    jt_.cluster(v).partition(a, y, x);
    jt_[v] = jt_[v].restrict(jt_.dims(v, y), a.values(y));
    jt_.update_cluster(v, x);
  });
  jt_.intersecting_separators(vars, [&](edge_descriptor e) {
    Domain y, x; // restricted, retained
    jt_.separator(e).partition(a, y, x);
    jt_[e] = jt_[e].restrict(jt_.dims(e, y), a.values(y));
    jt_.update_separator(e, x);
  });
}

/// Returns the belief associated with a vertex.
const Potential& belief(vertex_descriptor v) const {
  return jt_[v];
}

/// Returns the belief associated with an edge.
const Potential& belief(edge_descriptor e) const {
  return jt_[e];
}

Potential belief(const Domain& domain) const {
  // Try to find a separator that covers the variables
  edge_descriptor e = jt_.find_separator_cover(domain);
  if (e) {
    return jt_[e].marginal(jt_.dims(e, domain));
  }

  // Next, look for a clique that covers the variables
  vertex_descriptor v = jt_.find_cluster_cover(domain);
  if (v) {
    return jt_[v].marginal(jt_.dims(v, domain));
  }

  // Did not find a suitable clique / separator
  throw std::invalid_argument(
    "BeliefUpdateCalibrate::belief: the domain is not covered by any clique or separator."
  );
}

} // namespace libgm
