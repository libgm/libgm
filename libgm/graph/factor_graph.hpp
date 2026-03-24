#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/intrusive_list.hpp>
#include <libgm/datastructure/subrange.hpp>
#include <libgm/graph/markov_network.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/opaque.hpp>

#include <ankerl/unordered_dense.h>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

#include <cassert>
#include <memory>
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
class FactorGraph {
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
  FactorGraph(const FactorGraph& other);
  FactorGraph(FactorGraph&& other) noexcept;
  FactorGraph& operator=(const FactorGraph& other);
  FactorGraph& operator=(FactorGraph&& other) noexcept;
  ~FactorGraph();

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

  /// Returns an opaque reference to the property associated with an argument.
  OpaqueRef property(Arg u);

  /// Returns an opaque const reference to the property associated with an argument.
  OpaqueCref property(Arg u) const;

  /// Returns an opaque reference to the property associated with a factor.
  OpaqueRef property(Factor* u);

  /// Returns an opaque const reference to the property associated with a factor.
  OpaqueCref property(Factor* u) const;

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
   * Removes an argument from the graph and all adjacent factors.
   */
  void remove_argument(Arg u);

  /**
   * Removes a factor from this graph and all edges adhacent to it.
   * Does not remove any argument (even if it is no longer present in any remaining factors).
   */
  void remove_factor(Factor* u);

  /// Removes all vertices and edges from the graph.
  void clear();

  // Private members
  //--------------------------------------------------------------------------
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  Impl& impl();
  const Impl& impl() const;

  Argument& argument(Arg u);
  const Argument& argument(Arg u) const;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(impl_);
  }

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

  explicit FactorGraphT(const MarkovNetworkT<FP, FP>& mn)
    : FactorGraphT() {
    for (Arg u : mn.vertices()) {
      add_argument(u);
    }
    for (Arg u : mn.vertices()) {
      add_factor(Domain{u}, mn[u]);
    }
    for (Arg u : mn.vertices()) {
      for (UndirectedEdge<Arg> e : mn.out_edges(u)) {
        if (e.source() <= e.target()) {
          add_factor(Domain{e.source(), e.target()}, mn[e]);
        }
      }
    }
  }

  AP& operator[](Arg u) {
    return opaque_cast<AP>(property(u));
  }

  const AP& operator[](Arg u) const {
    return opaque_cast<AP>(property(u));
  }

  FP& operator[](Factor* f) {
    return opaque_cast<FP>(property(f));
  }

  const FP& operator[](Factor* f) const {
    return opaque_cast<FP>(property(f));
  }

  /// Adds an argument and associates it with a strongly-typed property.
  bool add_argument(Arg u, AP property) {
    bool inserted = FactorGraph::add_argument(u);
    if (inserted) {
      (*this)[u] = std::move(property);
    }
    return inserted;
  }

  /// Adds a factor and associates it with a strongly-typed property.
  Factor* add_factor(Domain args, FP property) {
    Factor* factor = FactorGraph::add_factor(std::move(args));
    (*this)[factor] = std::move(property);
    return factor;
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::base_class<const FactorGraph>(this));

    ar(cereal::make_size_tag(num_arguments()));
    for (Arg u : arguments()) {
      ar(CEREAL_NVP(u), cereal::make_nvp("property", operator[](u)));
    }

    ar(cereal::make_size_tag(num_factors()));
    for (Factor* f : factors()) {
      ar(operator[](f));
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(cereal::base_class<FactorGraph>(this));

    cereal::size_type argument_count;
    ar(cereal::make_size_tag(argument_count));
    assert(argument_count == num_arguments());
    for (size_t i = 0; i < argument_count; ++i) {
      Arg u;
      ar(CEREAL_NVP(u));
      assert(contains(u));
      ar(cereal::make_nvp("property", operator[](u)));
    }

    cereal::size_type factor_count;
    ar(cereal::make_size_tag(factor_count));
    assert(factor_count == num_factors());
    for (Factor* f : factors()) {
      ar(operator[](f));
    }
  }
};

} // namespace libgm
