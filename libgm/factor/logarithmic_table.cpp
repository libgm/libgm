#include <libgm/archives.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/impl/logarithmic_table.hpp>

namespace libgm {

template class LogarithmicTable<float>;
template class LogarithmicTable<double>;
template std::ostream& operator<<(std::ostream&, const LogarithmicTable<float>&);
template std::ostream& operator<<(std::ostream&, const LogarithmicTable<double>&);

}
