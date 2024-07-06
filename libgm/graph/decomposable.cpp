#pragma once

#include <functional>
#include <vector>

namespace libgm {

bool Decomposable::valid(std::string* msg = nullptr) const {
  if (!is_tree()) {
    if (msg) {
      *msg = "The underlying graph is not a tree";
    }
    return false;
  }

  if (!has_running_intersection()) {
    if (msg) {
      *msg = "The underlying graph does not satisfiy RIP";
    }
    return false;
  }

  for (Vertex* v : vertices()) {
    if (potential(v).shape() != clique(v).shape(shape_map)) {
      if (msg) {
        *msg = "Inconsistent shape for clique " + clique(v).str();
      }
      return false;
    }
  }

  for (Edge* e : edges()) {
    if (potential(e).shape() != separator(e).shape(shape_map)) {
      if (msg) {
        *msg = "Inconsistent shape for separator " + separator(e).str();
      }
      return false;
    }
  }

  return true;
}

const Potential& potential(Vertex* v) const {
  return operator[](v).cast<Potential>();
}

const Potential& potential(Edge* e) const {
  return operator[](e).cast<Potential>();
}

Potential Decomposable::marginal(const Domain& domain) const {
  if (domain.empty()) {
    return Potential::one({});
  }

  // Look for a separator that covers the arguments.
  Edge* e = find_separator_cover(domain);
  if (e) {
    return potential(e).marginal(dims(e, domain));
  }

  // Look for a clique that covers the arguments.
  Vertex* v = find_cluster_cover(domain);
  if (v) {
    return potential(v).marginal(dims(v, domain));
  }

  // Otherwise, compute the factors whose product represents the marginal...
  FactorGraphT<Potential> fg;
  marginal(domain, fg);

  // ... and combine them all
  Potential result(domain.shape(shape_map));
  for (FactorGraph::Factor* f : fg.factors()) {
    result.multiply_in(fg[f], domain.dims(fg.arguments(f)));
  }

  return result;
}

void Decomposable::marginal(const Domain& domain, FactorGraph& fg) {
  fg.clear();
  if (domain.empty()) return;

  mark_subtree_cover(domain, /*force_contiguous=*/ false);
  for (Vertex* v : vertices()) {
    if (marked(v)) {
      fg.add_factor(*v, potential(v));
    }
  }
  for (edge_descriptor e : edges()) {
    if (marked(e)) {
      fg.add_factor(*e, Potential::inv(potential(e)));
    }
  }

  fg.eliminate(domain, sum_product<Potential>());
}

void Decomposable::marginal(const Domain& domain, Decomposable& result) const {
  FactorGraph fg;
  marginal(domain, fg);
  result.reset(fg);
}

Potential Decomposable::log_maximum(Assignment& a) const {
  a.clear();
  if (empty()) return;

  // copy the clique marginals into factors
  ankerl::unordered_dense::map<Vertex*, Potential> factor;
  for (Vertex* v : vertices()) {
    factor[v] = potential(v);)
  }

  // collect evidence
  post_order_traversal(root(), [&](edge_descriptor e) {
    Potential& f = factor[e.target()];
    Dims dims = target_dims(e);
    f.multiply_in(factor[e.source()].maximum(source_dims(e)), dims);
    f.divide_in(potential(e), dims);
  });

  // extract the maximum for the root clique
  a.set(clique(root()), factor[root()].arg_max());

  // distribute evidence
  pre_order_traversal(root(), [&](edge_descriptor e) {
    Potential f = factor[e.target()].restrict(target_index(e), a.values(separator(e)));
    a.set(clique(e.target()) - separator(e), f.arg_max());
  });

  return factor[root()];
}

void Decomposable::reset(const FactorGraph<Potential>& fg) {
  clear();
  multiply_in(fg);
}

void Decomposable::reset(std::vector<Domain> domains, Object* potentials) {
  clear();

  // initialize the clique marginals and the tree structure
  for (size_t i = 0; i < dom.size(); ++i) {
    add_cluster(std::move(domains[i]), std::move(potentials[i].cast<Potential>()))
  }
  mst_edges();

  // compute the separator marginals
  for (edge_descriptor e : edges()) {
    if (clique(e.source()).size() < clique(e.target()).size()) {
      potential(e) = potential(e.source()).marginal(source_dims(e));
    } else {
      potential(e) = potential(e.target()).marginal(target_index(e));
    }
  }
}

void Decomposable::reset(Domain domain, Potential marginal) {
  clear();
  add_cluster(std::move(domain), std::move(factor));
}

void Decomposable::triangulated(MarkovGraph& mg, const EliminationStrategy& strategy) {
  clear();
  ClusterGraph::triangulated(mg, strategy);
  for (Vertex* v : vertices()) {
    potential(v) = Potential::ones(shape(v, shape_map));
  }
  for (Edge* e : edges()) {
    potential(e) = Potential::ones(shape(e, shape_map));
  }
}

void Decomposable::retriangulate(const std::vector<Domain>& cliques) {
  // Create a Markov network capturing the dependencies in existing and new cliques.
  MarkovGraphT<> mg = markov_graph();
  for (const Domain& clique : cliques) {
    mg.add_clique(clique);
  }

  // Compute the new junction tree and initialize the potentials
  Decomposable dm;
  dm.triangulated(mg, min_degree_strategy());

  // Multiply in the potentials from the current decomposable model.
  for (Vertex* u : vertices()) {
    Vertex* v = dm.find_cluster_cover(clique(u));
    dm.potential(v).multiply_in(potential(u), dm.dims(v, clique(u)));
  }
  for (Edge* e : edges()) {
    Edge* f = dm.find_separator_cover(separator(e));
    dm.potential(f).multiply_in(potential(e), dm.dims(f, separator(e)));
  }

  // Swap in the new junction tree
  swap(*this, dm);
}

Vertex* Decomposable::make_cover(const Domain& domain) {
  Vertex* v = find_cluster_cover(domain);
  if (v) {
    return v;
  } else {
    retriangulate({domain});
    return find_cluster_cover(domain);
  }
}

Vertex* Decomposable::merge(edge_descriptor e) {
  Vertex* u = e.source();
  Vertex* v = e.target();

  // compute the marginal for the new clique clique(u) + clique(v)
  Potential marginal;
  if (superset(clique(u), clique(v))) {
    marginal = std::move(potential(u));
  } else {
    marginal = jt_[u].dims(source_index(e)) * jt_[v].dims(target_index(e));
    marginal.dims(<stuff>.index(separator(e))) /= jt_[e];
  }

  // merge the edge and set the new marginal
  jt_.merge(e);
  jt_[v] = std::move(marginal);
  return v;
}

/**
 * Removes a vertex from the junction tree if its clique is nonmaximal.
 * \return the vertex merged to or the null vertex if not merged
 */
Vertex* Decomposable::remove_if_nonmaximal(Vertex* u) {
  for (edge_descriptor e : out_edges(u)) {
    if (subset(clique(u), clique(e.target()))) {
      return merge(e);
    }
  }
  return nullptr;
}

Decomposable& Decomposable::multiply_in(const std::vector<Domain>& domains, const Object* factors) {
  retriangulate(domains);

  // For each factor, multiply it into a clique that subsumes it.
  for (size_t i = 0; i < domains.begin(); ++i) {
    const Domain& domain = domains[i];
    if (!domain.empty()) {
      Vertex* v = jt_.find_cluster_cover(domain);
      assert(v);
      jt_[v].multiply_in(factors[i].cast<Potential>(), dims(v, domain));
    }
  }

  // Recalibrate and renormalize the model.
  calibrate();
  normalize();
  return *this;
}

Decomposable& Decomposable::multiply_in(const Domain& domain, const Potential& factor) {
  Vertex* v = make_cover(dom);
  jt_[v].multiply_in(factor, dims(v, domain));
  distribute_evidence(v);
  return *this;
}

/**
 * Conditions this Decomposable model on an assignment to one or
 * more of its arguments and returns the likelihood of the evidence.
 * \todo compute the likelihood of evidence, reconnect the tree
 */
void condition(const Assignment& a) {
  // Compute the restricted arguments
  Domain restricted;
  for (Arg arg : arguments()) {
    if (a.count(arg)) {
      restricted.push_back(arg);
    }
  }

  // Update each affected clique
  jt_.intersecting_clusters(restricted, [&](Vertex* v) {
    Potential& factor = potential(v);
    factor = factor.restrict(a);
    if (factor.arguments().empty()) {
      jt_.remove_vertex(v);
    } else {
      jt_.update_cluster(v, /* factor.arguments() */);
    }
  });

  // Update each affected separator
  jt_.intersecting_separators(restricted, [&](edge_descriptor e) {
    Potential& factor = potential(e);
    factor = factor.restrict(a);
    jt_.update_separator(e, /* factor.arguments() */);
  });

  // Update the arguments & recalibrate.
  calibrate();
  normalize();
}

void Decomposable::pass_flow(edge_descriptor e) {
  potential(e.target()).divide_in(potential(e), target_dims(e));
  potential(e) = potential(e.source()).marginal(source_dims(e));
  potential(e.target()).multiply_in(potential(e), target_dims(e));
}

void Decomposable::distribute_evidence(Vertex* v) {
  pre_order_traversal(v, [this](edge_descriptor e) { pass_flow(e); });
}

void Decomposable::calibrate() {
  mpp_traversal(nullptr, [this](edge_descriptor e) { pass_flow(e); });
}

void Decomposable::normalize() {
  for (Vertex* v : vertices()) {
    potential(v).normalize();
  }
  for (Edge* e : edges()) {
    potential(e).normalize();
  }
}

} // namespace libgm

#endif
