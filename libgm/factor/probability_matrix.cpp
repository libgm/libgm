#include <libgm/archives.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/impl/probability_matrix.hpp>

namespace libgm {

template class ProbabilityMatrix<float>;
template class ProbabilityMatrix<double>;
template std::ostream& operator<<(std::ostream&, const ProbabilityMatrix<float>&);
template std::ostream& operator<<(std::ostream&, const ProbabilityMatrix<double>&);

}
