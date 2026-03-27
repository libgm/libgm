#pragma once

#include <bitset>
#include <cstddef>
#include <initializer_list>

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
 * NOTE: the order of bits printed in the operator<< is MSB first, which deviates from Domain / Shape
 */
using Dims = std::bitset<64>;

Dims make_dims(std::initializer_list<size_t> idx);

Dims make_dims_range(size_t begin, size_t end);

} // namespace libgm
