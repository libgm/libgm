#ifndef LIBGM_CHOW_LIU_HPP
#define LIBGM_CHOW_LIU_HPP

#include <libgm/graph/undirected_graph.hpp>
#include <libgm/graph/algorithm/mst.hpp>
#include <libgm/model/decomposable.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>

namespace libgm {

  /**
   * Class for learning the Chow-Liu tree over a set of variables.
   * Models the Learner concept.
   *
   * \tparam F type of factor in the model
   *
   * \ingroup learning_structure
   */
  template <typename F>
  class chow_liu {
  public:
    // Learner concept types
    typedef decomposable<F>       model_type;
    typedef typename F::real_type real_type;

    // The algorithm parameters
    typedef typename factor_mle<F>::regul_type param_type;

    // Additional types
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type   domain_type;
    typedef undirected_edge<variable_type> edge_type;

    /**
     * Constructs the Chow-Liu learner using the given parameters.
     */
    explicit chow_liu(const param_type& param = param_type())
      : param_(param) { }

    /**
     * Fits a model using the supplied dataset for the given variables.
     */
    template <typename Dataset>
    chow_liu& fit(const Dataset& ds, const domain_type& vars) {
      factor_mle<F> mle(param_);
      model_.clear();
      score_.clear();

      // handle the edge cases first
      if (vars.size() <= 1) {
        if (vars.size() == 1) {
          model_.reset_marginal(mle(ds, {*vars.begin()}));
        }
        return *this;
      }

      // g will hold factor F and weight (mutual information) for each edge
      // this part could be optimized to eliminate copies
      undirected_graph<variable_type, void_, std::pair<F, real_type> > g;
      for (variable_type u : vars) {
        for (variable_type v : vars) {
          if (u < v) {
            F f = mle(ds, {u, v});
            real_type mi = f.mutual_information({u}, {v});
            edge_type e = g.add_edge(u, v, {f, mi}).first;
            score_[e] = mi;
          }
        }
      }

      // Compute the MST; the edges are the cliques to be kept
      std::vector<edge_type> edges;
      kruskal_minimum_spanning_tree(
        g,
        [&g](const edge_type& e) { return -g[e].second; },
        std::back_inserter(edges));

      // Construct the model
      std::vector<F> marginals;
      objective_ = real_type(0);
      for (const edge_type& e : edges) {
        marginals.push_back(std::move(g[e].first));
        objective_ += g[e].second;
      }
      model_.reset_marginals(marginals);
      return *this;
    }

    //! Returns the trained model.
    decomposable<F>& model() {
      return model_;
    }

    //! Returns the objective value.
    real_type objective() const {
      return objective_;
    }

    //! Returns a map from edges (pairs of variables) to scores.
    const std::unordered_map<edge_type, real_type>& scores() const {
      return score_;
    }

    // Private data
    // =========================================================================
  private:
    //! The regularization parameters used in estimating the factors.
    param_type param_;

    //! The trained model.
    model_type model_;

    //! The objective value.
    real_type objective_;

    //! A map from edges (unordered pairs of variables) to mutual information.
    std::unordered_map<edge_type, real_type> score_;

  }; // class chow_liu

} // namespace libgm

#endif
