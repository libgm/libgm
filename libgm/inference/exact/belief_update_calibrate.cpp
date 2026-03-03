#include "belief_update_calibrate.hpp"

namespace libgm {

using Potential = BeliefUpdateCalibrate::Potential;

void BeliefUpdateCalibrate::reset(
    MarkovNetwork& mn, const EliminationStrategy& strategy, const ShapeMap& shape_map) {
  // compute the junction tree for the given factors
  jt_.triangulated(mn, strategy);

  // intialize the clique and separator potentials to unity
  for (vertex_descriptor v : jt_.vertices()) {
    belief(v) = one(jt_.shape(v, shape_map));
  }
  for (edge_descriptor e : jt_.edges()) {
    belief(e) = one(jt_.shape(e, shape_map));
  }
}

void BeliefUpdateCalibrate::multiply_in(const Domain& domain, Potential factor) {
  vertex_descriptor v = jt_.find_cluster_cover(domain);
  assert(v);
  belief(v).multiply_in(jt_.dims(v, domain), factor);
}

void BeliefUpdateCalibrate::calibrate() {
  jt_.mpp_traversal(nullptr, [&](edge_descriptor e) {
    belief(e.target()).divide_in(belief(e), jt_.target_dims(e));
    belief(e) = belief(e.source()).marginal(jt_.source_dims(e));
    belief(e.target()).multiply_in(belief(e), jt_.target_dims(e));
  });
}

void BeliefUpdateCalibrate::normalize() {
  for (vertex_descriptor v : jt_.vertices()) {
    belief(v).normalize();
  }
  for (edge_descriptor e : jt_.edges()) {
    belief(e).normalize();
  }
}

void BeliefUpdateCalibrate::condition(const Assignment& a) {
  // Extract the restricted arguments
  Domain args = a.keys();

  // Update the factors and messages
  jt_.intersecting_clusters(args, [&](vertex_descriptor v) {
    Domain y, x; // restricted, retained
    a.partition(jt_.cluster(v), y, x);
    belief(v) = belief(v).restrict(jt_.dims(v, y), a.values(y));
    jt_.update_cluster(v, x);
  });
  jt_.intersecting_separators(vars, [&](edge_descriptor e) {
    Domain y, x; // restricted, retained
    a.partition(jt_.separator(e), y, x);
    belief(e) = belief(e).restrict(jt_.dims(e, y), a.values(y));
    jt_.update_separator(e, x);
  });
}

Potential BeliefUpdateCalibrate::belief(const Domain& domain) const {
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
    "BeliefUpdateCalibrate::belief: the domain is not covered by any clique or separator."
  );
}

} // namespace libgm
