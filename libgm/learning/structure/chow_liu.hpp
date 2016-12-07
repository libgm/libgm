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
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam F
   *         A type of factors in the model.
   *
   * \ingroup learning_structure
   */
  template <typename Arg, typename F>
  class chow_liu {
    using edge_type = undirected_edge<Arg>;

  public:
    // Learner concept types
    using model_type = decomposable<Arg, F>;
    using real_type  = typename F::real_type;

    // The algorithm parameters
    using param_type = typename F::mle_type::regul_type;

    /**
     * Constructs the Chow-Liu learner using the given parameters.
     */
    explicit chow_liu(const param_type& param = param_type())
      : param_(param) { }

    /**
     * Fits a model using the supplied dataset for the given variables.
     */
    chow_liu& fit(const dataset<Arg, real_type>& ds, const domain<Arg>& args) {
      typename F::mle_type mle(param_);
      model_.clear();
      score_.clear();

      // handle the edge cases first
      if (args.empty()) {
        return *this;
      } else if (args.size() == 1) {
        Arg v = args.front();
        model_.reset_marginal({v}, mle(ds.project(v), F::shape(v)));
        return *this;
      }

      // g will hold factor F and weight (mutual information) for each edge
      // this part could be optimized to eliminate copies
      using vertex_property = std::pair<std::annotated<Arg, F>, real_type>;
      undirected_graph<Arg, void_, vertex_property> g;
      for (Arg u : args) {
        for (Arg v : args) {
          if (u < v) {
            annotated<Arg, F> f;
            f.domain = { u, v };
            f.factor = mle(ds.samples(f.domain), F::shape(f.domain));
            std::size_t nu = argument_arity(u), nv = argument_arity(v);
            real_type mi = f.factor.mutual_information(0, nu, nu, nv);
            edge_type e = g.add_edge(u, v, {std::move(f), mi}).first;
            score_[e] = mi;
          }
        }
      }

      // Compute the MST; the edges are the cliques to be kept
      std::vector<edge_type> edges;
      kruskal_minimum_spanning_tree(
        g,
        [&g](edge_type e) { return -g[e].second; },
        std::back_inserter(edges));

      // Construct the model
      std::vector<annotated<Arg, F> > marginals;
      objective_ = real_type(0);
      for (edge_type e : edges) {
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
    //--------------------------------------------------------------------------
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
