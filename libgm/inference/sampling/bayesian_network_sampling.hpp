#include <random>


    /**
     * Draws a single sample from a Bayesian network.
     * \tparam Generator a type that models UniformRandomNumberGenerator
     */
    template <typename Generator>
    void sample(Generator& rng, assignment_type& a) const {
      partial_order_traversal(*this, [&](vertex_type v) {
          (*this)[v].sample(rng, {v}, a);
        });
    }
