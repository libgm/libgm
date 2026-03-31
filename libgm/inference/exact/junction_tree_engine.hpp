#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/model/markov_structure.hpp>
#include <libgm/graph/elimination_strategy.hpp>

namespace libgm {

template <typename F>
struct JunctionTreeEngine {
  virtual void reset(MarkovStructure mg, const EliminationStrategy& strategy, const ShapeMap& shape_map) = 0;
  virtual void multiply_in(const Domain& domain, const F& factor) = 0;
  virtual void calibrate() = 0;
  virtual void normalize() = 0;
  virtual void condition(const typename F::assignment_type& a) = 0;
};

}
