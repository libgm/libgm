#ifndef LIBGM_MEAN_FIELD_BIPARTITE_HPP
#define LIBGM_MEAN_FIELD_BIPARTITE_HPP

#include <libgm/factor/traits.hpp>
#include <libgm/graph/bipartite_graph.hpp>
#include <libgm/parallel/vector_processor.hpp>

#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace libgm {

  /**
   * A class that runs the mean field algorithm for a bipartite graph.
   * The computation is performed synchronously, first for all type-1
   * vertices and then for all type-2 vertices. The number of worker
   * threads is controlled by a parameter to the constructor.
   *
   * \tparam Vertex1 the type that represents a type-1 vertex
   * \tparam Vertex2 the type that represents a type-2 vertex
   * \tparam NodeF the factor type associated with vertices
   * \tparam EdgeF the factor type associated with edges
   */
  template <typename Vertex1,
            typename Vertex2,
            typename NodeF,
            typename EdgeF = NodeF>
  class mean_field_bipartite {
    static_assert(are_pairwise_compatible<NodeF, EdgeF>::value,
                  "The node & edge factor types are not pairwise compatible");
    // Public types
    //==========================================================================
  public:
    // Factorized Inference types
    typedef typename NodeF::real_type        real_type;
    typedef typename NodeF::result_type      result_type;
    typedef typename NodeF::argument_type    argument_type;
    typedef typename NodeF::assignment_type  assignment_type;
    typedef typename NodeF::probability_type belief_type;

    typedef bipartite_graph<Vertex1, Vertex2, NodeF, NodeF, EdgeF> model_type;
    typedef typename model_type::edge_type edge_type;

    // Public functions
    //==========================================================================
  public:
    /**
     * Creates a mean field engine for the given graph.
     * The graph vertices must not change after initialization
     * (the potentials may).
     *
     * \param num_threads the number of worker threads
     */
    explicit mean_field_bipartite(const model_type* model,
                                  std::size_t nthreads = 1)
      : model_(*model), nthreads_(nthreads) {
      vertices1_.reserve(model_.num_vertices1());
      vertices2_.reserve(model_.num_vertices2());
      beliefs1_.reserve(model_.num_vertices1());
      beliefs2_.reserve(model_.num_vertices2());

      for (Vertex1 v : model_.vertices1()) {
        vertices1_.push_back(v);
        beliefs1_[v] = belief_type(model_[v].arguments(), real_type(1));
      }

      for (Vertex2 v : model_.vertices2()) {
        vertices2_.push_back(v);
        beliefs2_[v] = belief_type(model_[v].arguments(), real_type(1));
      }
    }

    /**
     * Returns the vector of type-1 vertices.
     */
    const std::vector<Vertex1>& vertices1() const {
      return vertices1_;
    }

    /**
     * Returns the vector of type-1 vertices.
     */
    const std::vector<Vertex2>& vertices2() const {
      return vertices2_;
    }

    /**
     * Performs a single iteration of mean field.
     */
    real_type iterate() {
      real_type diff1 = update_all(vertices1_);
      real_type diff2 = update_all(vertices2_);
      return (diff1 + diff2) / model_.num_vertices();
    }

    /**
     * Returns the belief for a type-1 vertex.
     */
    const belief_type& belief(Vertex1 v) const {
      return beliefs1_.at(v);
    }

    /**
     * Returns the belief for a type-2 vertex.
     */
    const belief_type& belief(Vertex2 v) const {
      return beliefs2_.at(v);
    }

    // Private members
    //==========================================================================
  private:
    /**
     * Updates the given range of vertices and returns the sum of the
     * factor differences.
     * \tparam Vertex the vertex type
     */
    template <typename Vertex>
    real_type update_all(std::vector<Vertex>& vertices) {
      std::vector<real_type> sums(nthreads_, real_type(0));
      vector_processor<Vertex, real_type> process([&](Vertex v, real_type& sum){
          sum += update(v);
        });
      process(vertices, sums);
      return std::accumulate(sums.begin(), sums.end(), real_type(0));
    }

    /**
     * Updates a single vertex.
     * \tparam Vertex the vertex type
     */
    template <typename Vertex>
    real_type update(Vertex v) {
      NodeF result = model_[v];
      for (edge_type e : model_.in_edges(v)) {
        if (e.forward()) {
          model_[e].exp_log_multiply(belief(e.v1()), result);
        } else {
          model_[e].exp_log_multiply(belief(e.v2()), result);
        }
      }
      result /= result.maximum();
      belief_type new_belief(result);
      new_belief.normalize();
      swap(const_cast<belief_type&>(belief(v)), new_belief);
      return sum_diff(new_belief, belief(v));
    }

    //! The underlying graphical model.
    const model_type& model_;

    //! The number of worker threads.
    std::size_t nthreads_;

    //! A vector of type-1 vertices for quick access.
    std::vector<Vertex1> vertices1_;

    //! A vector of type-2 vertices for quick access.
    std::vector<Vertex2> vertices2_;

    //! A map of current beliefs for type-1 vertices
    std::unordered_map<Vertex1, belief_type> beliefs1_;

    //! A map of current beliefs for type-2 vertices
    std::unordered_map<Vertex2, belief_type> beliefs2_;

  };

} // namespace libgm

#endif
