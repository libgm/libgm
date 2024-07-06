#pragma once

#include <libgm/iterator/bind1_iterator.hpp>
#include <libgm/iterator/bind2_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <unordered_map>

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
class FactorGraph {
protected:
  struct Argument;
  struct Factor;

  using ArgumentMap = ankerl::unordered_dense::map<Arg, Argument*>;
  using FactorList = boost::intrusive::list<VertexBase>

  // Public types
  //--------------------------------------------------------------------------
public:
  // A set of factors.
  using FactorSet = ankerl::unordered_dense::set<Factor*>;

  // Iterators (the exact types are implementation detail).
  using argument_iterator = MapKeyIterator<ArgumentMap>;
  using factor_iterator = MemberIterator<FactorList::const_iterator, &VertexBase::cast<Factor>>;

  // Constructors, destructors, and related functions
  //--------------------------------------------------------------------------
public:
  /// Creates an empty graph.
  FactorGraph();

  /// Swap with another graph in constant time
  friend void swap(FactorGraph& a, FactorGraph& b);

  // Accessors
  //--------------------------------------------------------------------------
public:
  /// Returns the range of all arguments.
  boost::iterator_range<argument_iterator> arguments() const;

  /// Returns the range of all factors.
  boost::iterator_range<factor_iterator> factors() const;

  /// Returns the arguments associated with a factor.
  const Domain& arguments(Factor* u) const;

  /// Returns the factors containing an argument.
  const FactorSet& factors(Arg u) const;

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

  /// Returns the property associated with an argument.
  const Object& operator[](Arg u) const;

  /// Returns the property associated with a factor.
  const Object& operator[](Factor* u) const;

  /// Returns the property associated with an argument.
  Object& operator[](Arg u);

  /// Returns the property associated with a factor.
  Object& operator[](Factor* u);

  /// Prints the graph to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const FactorGraph& g);

  // Queries
  //--------------------------------------------------------------------------
  /// Returns the Markov graph capturing the cliques in this factor graph.
  MarkovGraphT<> markov_graph() const;

  // Modifications
  //--------------------------------------------------------------------------

  /**
   * Adds an argument to this graph and associates a property with it.
   * If the vertex is already present, its property is not overwritten.
   * \return true if the insertion took place (i.e., vertex was not present).
   */
  bool add_argument(Arg u, Object property = Object());

  /**
   * Adds a factor to this graph and associates a property with it.
   * \return The descriptor of the newly added vertex.
   */
  Factor* add_factor(Domain args, Object property = Object());

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

  /// Saves the graph to an archive.
  void save(oarchive& ar) const;

  /// Loads the graph from an archive.
  void load(iarchive& ar);

  // Eliminates all variables other than the specified ones.
  void eliminate(const Domain& retain, EliminationStrategy strategy);

  // Private members
  //--------------------------------------------------------------------------
private:
  const Argument& argument(Arg u) const;
  Argument& argument(Arg u);

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
  /// Returns the strongly-typed property associated with an argument.
  std::add_lvalue_reference_t<AP> operator[](Arg u) {
    if constexpr (!std::is_same_v<AP, void>) {
      return static_cast<AP&>(FactorGraph::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with an argument.
  std::add_lvalue_reference_t<const AP> operator[](Arg u) const {
    if constexpr (!std::is_same_v<AP, void>) {
      return static_cast<const AP&>(FactorGraph::operator[](u));
    }
  }

  /// Returns the strongly-typed property associated with a factor.
  std::add_lvalue_reference_t<FP> operator[](Factor* f) {
    if constexpr (!std::is_same_v<FP, void>) {
      return static_cast<FP&>(FactorGraph::operator[](f));
    }
  }

  /// Returns the strongly-typed property associated with an edge.
  std::add_lvalue_reference_t<const FP> operator[](Factor* f) const {
    if constexpr (!std::is_same_v<FP, void>) {
      return static_cast<const FP&>(FactorGraph::operator[](f));
    }
  }

  /// Adds an argument and associates it with a strongly-typed property.
  bool add_argument(Arg u, Nullable<AP> property = Nullable<AP>()) {
    return FactorGraph::add_argument(u, std::move(property));
  }

  /// Adds a factor and associates it with a strongly-typed property.
  Factor* add_factor(Domain args, Nullable<FP> property = Nullable<FP>()) {
    return FactorGraph::add_factor(std::move(args), std::move(property));
  }
};

} // namespace libgm
