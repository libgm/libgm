#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_assignment.hpp>
#include <libgm/graph/elimination_strategy.hpp>
#include <libgm/model/markov_structure.hpp>

namespace libgm {

template <Argument Arg, typename F>
struct JunctionTreeEngine {
  using assignment_type = typename F::template assignment_t<Arg>;

  virtual ~JunctionTreeEngine() = default;

  virtual void reset(MarkovStructure<Arg> mg, const EliminationStrategy& strategy, const ShapeMap<Arg>& shape_map) = 0;
  virtual void multiply_in(const Domain<Arg>& domain, const F& factor) = 0;
  virtual void calibrate() = 0;
  virtual void normalize() = 0;
  virtual void condition(const assignment_type& a) = 0;
};

} // namespace libgm
