#pragma once

#include <libgm/object.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/intrusive_list.hpp>
#include <libgm/datastructure/subrange.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/markov_network.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/map_key_iterator.hpp>

#include <ankerl/unordered_dense.h>

#include <new>
#include <type_traits>
#include <utility>

namespace libgm {

/**
 * A class that represents an undirected bipartite graph. The graph contains
 * two types of vertices (type 1 and type 2), which correspond to the two
 * sides of the partition. These vertices are represented by the template
 * arguments Vertex1 and Vertex2, respectively. These two types _must_ be
 * distinct and not convertible to each other. This requirement is necessary
 * to allow for overload resolution to work in functions, such as neighbors().
 *
 * \ingroup graph_types
 */
class FactorGraph : public Object {
protected:
  struct Argument;
  using ArgumentMap = ankerl::unordered_dense::map<Arg, Argument*>;

  // Public types
  //--------------------------------------------------------------------------
public:
  /// Factor class (Factor* is the handle).
  struct Factor;

  // Iterators (the exact types are implementation detail).
  using argument_iterator = MapKeyIterator<ArgumentMap>;
  using factor_iterator = IntrusiveList<Factor>::iterator;

  // Constructors, destructors, and related functions
  //--------------------------------------------------------------------------
public:
  /// Creates an empty graph.
  FactorGraph();

protected:
  FactorGraph(PropertyLayout argument_layout, PropertyLayout factor_layout);

  /// Swap with another graph in constant time
  friend void swap(FactorGraph& a, FactorGraph& b);

  // Accessors
  //--------------------------------------------------------------------------
public:
  /// Returns the range of all arguments.
  SubRange<argument_iterator> arguments() const;

  /// Returns the range of all factors.
  SubRange<factor_iterator> factors() const;

  /// Returns the arguments associated with a factor.
  const Domain& arguments(Factor* u) const;

  /// Returns the factors containing an argument.
  const IntrusiveList<Factor>& factors(Arg u) const;

  /// Returns true if the graph contains the given argument.
  bool contains(Arg u) const;

  /// Returns true if the graph contains the given factor.
  bool contains(Factor* u) const;

  /// Returns true if the graph contains an undirected edge {u, v}.
  bool contains(Arg u, Factor* v) const;

  /// Returns the number of factors containing the given argument.
  size_t degree(Arg u) const;

  /// Returns the number of arguments of the given factor.
  size_t degree(Factor* u) const;

  /// Returns true if the graph has no arguments or factors.
  bool empty() const;

  /// Returns the number of (unique) arguments.
  size_t num_arguments() const;

  /// Returns the number of factors.
  size_t num_factors() const;

  /// Returns the raw pointer to the property associated with an argument.
  void* property(Arg u);

  /// Returns the raw pointer to the property associated with an argument.
  const void* property(Arg u) const;

  /// Returns the raw pointer to the property associated with a factor.
  void* property(Factor* u);

  /// Returns the raw pointer to the property associated with a factor.
  const void* property(Factor* u) const;

  /// Prints the graph to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const FactorGraph& g);

  // Queries
  //--------------------------------------------------------------------------
  /// Returns the Markov graph capturing the cliques in this factor graph.
  MarkovNetwork markov_network() const;

  // Modifications
  //--------------------------------------------------------------------------

  /// Adds an argument to this graph.
  bool add_argument(Arg u);

  /// Adds a factor to this graph.
  Factor* add_factor(Domain args);

  /**
   * Removes an argument from the graph.
   * The argument must not be present in any factor.
   */
  void remove_argument(Arg u);

  /**
   * Removes a factor from this graph and all edges adhacent to it.
   * Does not remove any argument (even if it is no longer present in any remaining factors).
   */
  void remove_factor(Factor* u);

  /// Removes all vertices and edges from the graph.
  void clear();

  // Eliminates all variables other than the specified ones.
  void eliminate(const Domain& retain, const CommutativeSemiring& csr, const ShapeMap& shape_map,
                 const EliminationStrategy& strategy);

  // Private members
  //--------------------------------------------------------------------------
private:
  struct Impl;

  Impl& impl();
  const Impl& impl() const;

  Argument& argument(Arg u);
  const Argument& argument(Arg u) const;

}; // class FactorGraph

/**
 * A factor graph with strongly typed argument and factor properties.
 *
 * \tparam AP
 *         The type of data associated with arguments.
 * \tparam FP
 *         The type of data associated with factors.
 */
template <typename AP, typename FP>
struct FactorGraphT : FactorGraph {
  static_assert(!std::is_void_v<AP>, "AP must be a non-void property type.");
  static_assert(!std::is_void_v<FP>, "FP must be a non-void property type.");

  using FactorGraph::add_argument;
  using FactorGraph::add_factor;

  FactorGraphT()
    : FactorGraph(property_layout<AP>(), property_layout<FP>()) {}

  AP& operator[](Arg u) {
    return *static_cast<AP*>(FactorGraph::property(u));
  }

  const AP& operator[](Arg u) const {
    return *static_cast<const AP*>(FactorGraph::property(u));
  }

  FP& operator[](Factor* f) {
    return *static_cast<FP*>(FactorGraph::property(f));
  }

  const FP& operator[](Factor* f) const {
    return *static_cast<const FP*>(FactorGraph::property(f));
  }

  /// Adds an argument and associates it with a strongly-typed property.
  bool add_argument(Arg u, AP property) {
    bool inserted = FactorGraph::add_argument(u);
    if (inserted) {
      static_cast<AP*>(FactorGraph::property(u))->~AP();
      new (FactorGraph::property(u)) AP(std::move(property));
    }
    return inserted;
  }

  /// Adds a factor and associates it with a strongly-typed property.
  Factor* add_factor(Domain args, FP property) {
    Factor* factor = FactorGraph::add_factor(std::move(args));
    static_cast<FP*>(FactorGraph::property(factor))->~FP();
    new (FactorGraph::property(factor)) FP(std::move(property));
    return factor;
  }
};

} // namespace libgm
