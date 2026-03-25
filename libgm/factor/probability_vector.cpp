#include <libgm/archives.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/factor/impl/probability_vector.hpp>

namespace libgm {

template class ProbabilityVector<float>;
template class ProbabilityVector<double>;
template std::ostream& operator<<(std::ostream&, const ProbabilityVector<float>&);
template std::ostream& operator<<(std::ostream&, const ProbabilityVector<double>&);

}
