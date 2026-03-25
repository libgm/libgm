#include <libgm/archives.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/impl/logarithmic_matrix.hpp>

namespace libgm {

template class LogarithmicMatrix<float>;
template class LogarithmicMatrix<double>;
template std::ostream& operator<<(std::ostream&, const LogarithmicMatrix<float>&);
template std::ostream& operator<<(std::ostream&, const LogarithmicMatrix<double>&);

}
