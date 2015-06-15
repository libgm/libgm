#ifndef LIBGM_FACTOR_GRAPH_HPP
#define LIBGM_FACTOR_GRAPH_HPP

#include <libgm/graph/bipartite_graph.hpp>
#include <libgm/graph/id.hpp>
#include <libgm/graph/property_fn.hpp>
#include <libgm/math/logarithmic.hpp>

#include <boost/iterator/transform_iterator.hpp>

#include <vector>

namespace libgm {

  /**
   * This class represents a factor graph model. A factor graph is a bipartite
   * graph, where type-1 vertices correspond to factors, and type-2 vertices
   * correspond to variables. There is an undirected edge between a factor
   * and a variable if the variable is in the domain of the factor.
   * This model represents (an unnormalized) distribution over the
   * variables as a product of all the contained factors.
   *
   * \tparam F the factor type stored in this model
   * \ingroup model
   */
  template <typename F>
  class factor_graph
    : public bipartite_graph<id_t, typename F::variable_type, F> {

    typedef bipartite_graph<id_t, typename F::variable_type, F> base;

    // Public type declarations
    // =========================================================================
  public:
    // FactorizedModel types
    typedef typename F::real_type       real_type;
    typedef logarithmic<real_type>      result_type;
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef F                           value_type;

    typedef boost::transform_iterator<
      vertex1_property_fn<base>, typename base::vertex1_iterator
    > iterator;

    typedef boost::transform_iterator<
      vertex1_property_fn<const base>, typename base::vertex1_iterator
    > const_iterator;

    // Shortcuts
    typedef typename base::vertex1_iterator vertex1_iterator;
    typedef typename base::vertex2_iterator vertex2_iterator;

    // bring functions from base
    using base::print_degree_distribution;

    // Constructors
    //==========================================================================
  public:
    //! Creates an empty factor graph
    factor_graph() : max_id_(0) { }

    //! Swaps two factor graphs in constant time
    friend void swap(factor_graph& a, factor_graph& b) {
      swap(static_cast<base&>(a), static_cast<base&>(b));
      std::swap(a.max_id_, b.max_id_);
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of the model (the range of all type-2 vertices).
    iterator_range<vertex2_iterator> arguments() const {
      return this->vertices2();
    }

    //! Returns the arguments of the factor with the given id.
    const domain_type& arguments(id_t id) const {
      return (*this)[id].arguments();
    }

    //! Returns the range of ids of all factors in the graph.
    iterator_range<vertex1_iterator>
    factor_ids() const {
      return this->vertices1();
    }

    //! Returns the number of variables in the graph (alias for num_vertices2).
    std::size_t num_arguments() const {
      return this->num_vertices2();
    }

    //! Returns the number of factors in the graph (alias for num_vertices1).
    std::size_t num_factors() const {
      return this->num_vertices1();
    }

    //! Returns the iterator to the first factor.
    iterator begin() {
      return iterator(this->vertices1().begin(),
                      vertex1_property_fn<base>(this));
    }

    //! Returns the iterator to the first factor.
    const_iterator begin() const {
      return const_iterator(this->vertices1().begin(),
                            vertex1_property_fn<const base>(this));
    }

    //! Returns the iterator past the last factor.
    iterator end() {
      return iterator(this->vertices1().end(),
                      vertex1_property_fn<base>(this));
    }

    //! Returns the iterator past the last factor.
    const_iterator end() const {
      return const_iterator(this->vertices1().end(),
                            vertex1_property_fn<const base>(this));
    }

    //! Prints the degree distribution to the given stream
    void print_degree_distribution(std::ostream& out) const {
      out << "Degree distribution of variables:" << std::endl;
      print_degree_distribution(out, arguments());
      out << "Degree distribution of factors:" << std::endl;
      print_degree_distribution(out, factor_ids());
    }

    // Modifiers
    //==========================================================================

    /**
     * Adds a non-empty factor to the factor graph.
     * \returns the corresponding id or null if the factor has no arguments
     */
    id_t add_factor(const F& f) {
      if (f.arguments().empty()) {
        return id_t(); // not added
      }
      while (this->contains(max_id_)) {
        ++max_id_;
      }
      this->add_vertex(max_id_, f);
      for (variable_type var : f.arguments()) {
        this->add_edge(max_id_, var);
      }
      return max_id_;
    }

    /**
     * Updates the factor associated with the given id, modifying the
     * graph structure to reflect the new factor arguments.
     */
    void update_factor(id_t id, const F& f) {
      // if the factor is a constant, just drop the factor
      if (f.arguments().empty()) {
        this->remove_vertex(id);
        return;
      }
      // remove all the old edges
      this->remove_edges(id);
      // add all the new edges
      for (variable_type v : f.arguments()) {
        this->add_edge(id, v);
      }
      (*this)[id] = f;
    }

    /**
     * Normalize all factors
     */
    void normalize() {
      for (F& factor : *this) {
        factor.normalize();
      }
    }

    /**
     * Condition on an assignment.
     * \todo This does not work yet.
     */
    void condition(const assignment_type& a) {
      assert(false); // not working yet
      for (const auto& p : a) {
        variable_type var = p.first;
        if (!this->contains(var)) {
          continue;
        }
        for (id_t id : this->neighbors(var)) {
          if (arguments(id).count(var)) { // not processed yet
            update_factor(id, (*this)[id].restrict(a));
          }
        }
        this->remove_vertex(var);
      }
    }

    /**
     * Simplify the model by merging factors. For each factor f(X),
     * if a factor g(Y) exists such that X \subseteq Y, the factor
     * g is multiplied by f, and f is removed from the model.
     */
    void simplify() {
      std::vector<id_t> ids(factor_ids().begin(), factor_ids().end());
      for (id_t f_id : ids) {
        const domain_type& f_args = arguments(f_id);
        // identify the variable with the fewest connected factors
        std::size_t min_degree = std::numeric_limits<std::size_t>::max();
        variable_type var;
        for (variable_type v : f_args) {
          if (this->degree(v) < min_degree) {
            min_degree = this->degree(v);
            var = v;
          }
        }
        // identify a subsuming factor among the new neighbors of var
        for (id_t g_id : this->neighbors(var)) {
          if (f_id != g_id && subset(f_args, arguments(g_id))) {
            (*this)[g_id] *= (*this)[f_id];
            this->remove_vertex(f_id);
            break;
          }
        }
      }
    }

  private:
    //! The largest ID seen so far.
    id_t max_id_;

  }; // class factor_graph

} // namespace libgm

#endif
