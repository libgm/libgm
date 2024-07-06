#pragma once

#include <bitset>

namespace libgm {

/**
 * A set of dimensions.
 *
 * This choice hard-codes the maximum number of arguments a factor can have to 64.
 * This should be enough for most reasonable applications (past 64, the inference
 * will be slow). Note that, for example, the multivariate normal factors
 * MomentGaussian and CanonicalGaussian can have more than 64 dimensions, because
 * each argument in those factors can have length > 1.
 *
 * TODO: check the order of bits printed in the operator<< - MSB or LSB first?
 */
using Dims = std::bitset<64>;

} // namespace libgm
