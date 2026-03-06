#pragma once

#include <libgm/argument/argument.hpp>

#include <vector>

namespace libgm {

class MarkovNetwork;

struct EliminationStrategy {
  virtual ptrdiff_t priority(Arg u, const MarkovNetwork& g) const = 0;
  virtual void updated(Arg u, const MarkovNetwork& g, std::vector<Arg>& output) const = 0;
};

}
