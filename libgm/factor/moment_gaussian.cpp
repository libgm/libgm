#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/factor/impl/moment_gaussian.hpp>

namespace libgm {

template class MomentGaussian<float>;
template class MomentGaussian<double>;
template std::ostream& operator<<(std::ostream&, const MomentGaussian<float>&);
template std::ostream& operator<<(std::ostream&, const MomentGaussian<double>&);

}
