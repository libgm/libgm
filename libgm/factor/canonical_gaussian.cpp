#include <libgm/archives.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/impl/canonical_gaussian.hpp>

namespace libgm {

template class CanonicalGaussian<float>;
template class CanonicalGaussian<double>;
template std::ostream& operator<<(std::ostream&, const CanonicalGaussian<float>&);
template std::ostream& operator<<(std::ostream&, const CanonicalGaussian<double>&);

}
