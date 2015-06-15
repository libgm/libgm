#ifndef LIBGM_CONSTRAINED_ELIM_STRATEGY_HPP
#define LIBGM_CONSTRAINED_ELIM_STRATEGY_HPP

#include <utility>

namespace libgm {

  /**
   * Implements an elimination strategy subject to an elimination
   * order constraint. The elimination (partial) order is specified
   * by an intrinsic priority associated with each vertex. For
   * example, if some vertices have intrinsic priority 0 and others
   * have intrinsic priority 1, then all vertices with intrinsic
   * priority 1 will be eliminated before those with intrinsic
   * priority 0. This type models the EliminationStrategy concept.
   *
   * \tparam IntrinsicPriority
   *         A type that represents the intrinsic priority.
   * \tparam Function
   *         A unary function that accepts a vertex and returns
   *         the intrinsic priority for that vertex.
   * \tparam Strategy
   *         A type that models the EliminationStrategy concept.
   *         It is used to choose the relative elimination order
   *         of vertices with the same intrinsic priority.
   *
   * \ingroup graph_types
   */
  template <typename IntrinsicPriority, typename Function, typename Strategy>
  struct constrained_elim_strategy {

    /**
     * The type of the priority of this constrained elimination
     * strategy. Note that the ordering defined on std::pair
     * implements the correct semantics for ordering vertices given
     * the primary and secondary priorities.
     */
    typedef std::pair<IntrinsicPriority,
                      typename Strategy::priority_type> priority_type;

    /**
     * Default constructor.
     * Only applicable when Function and Strategy are default-constructible.
     */
    constrained_elim_strategy() { }

    /**
     * Constructor.
     */
    constrained_elim_strategy(Function intrinsic_priority,
                              Strategy secondary_strategy)
      : intrinsic_priority_(intrinsic_priority),
        secondary_strategy_(secondary_strategy) { }

    /**
     * Computes the priority of a vertex using the intrinsic priority
     * and the secondary strategy.
     */
    template <typename Graph>
    priority_type priority(typename Graph::vertex_type v, const Graph& g) {
      return std::make_pair(intrinsic_priority_(v),
                            secondary_strategy_.priority(v, g));
    }

    /**
     * Computes the set of vertices whose priority may change if a
     * designated vertex is eliminated.  Since the intrinsic priority
     * does not change during the elimination process, this is the
     * same set of vertices as reported by the secondary strategy.
     */
    template <typename Graph, typename OutIt>
    void updated(typename Graph::vertex_type v, const Graph& g, OutIt out) {
      secondary_strategy_.updated(v, g, out);
    }

  private:
    //! The functor that computes the primary priority.
    Function intrinsic_priority_;

    //! The secondary strategy.
    Strategy secondary_strategy_;

  }; // struct constrained_elim_strategy

  /**
   * Utility function that constructs a constrained elimination strategy
   * from its arguments, inferring the contrained_elim_strategy template
   * arguments.
   */
  template <typename Result, typename Function, typename Strategy>
  constrained_elim_strategy<Result, Function, Strategy>
  constrained_strategy(Function fn, Strategy s) {
    return { fn, s };
  }

} // namespace libgm

#endif
