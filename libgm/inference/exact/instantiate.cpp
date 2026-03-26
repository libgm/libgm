#include "belief_update_calibrate.hpp"
#include "sum_product_calibrate.hpp"
#include "variable_elimination.hpp"

#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>

namespace libgm {
  template class BeliefUpdateCalibrate<CanonicalGaussian<double>>;
  template class BeliefUpdateCalibrate<ProbabilityTable<double>>;

  template class SumProductCalibrate<CanonicalGaussian<double>>;
  template class SumProductCalibrate<ProbabilityTable<double>>;

  template class VariableElimination<CanonicalGaussian<double>>;
  template class VariableElimination<ProbabilityTable<double>>;
}