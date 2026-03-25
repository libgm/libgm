#include "belief_update_calibrate_test.hpp"
#include "sum_product_calibrate_test.hpp"
#include "variable_elimination_test.hpp"

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