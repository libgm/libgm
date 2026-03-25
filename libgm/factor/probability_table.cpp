#include <libgm/archives.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/impl/probability_table.hpp>

namespace libgm {

template class ProbabilityTable<float>;
template class ProbabilityTable<double>;
template std::ostream& operator<<(std::ostream&, const ProbabilityTable<float>&);
template std::ostream& operator<<(std::ostream&, const ProbabilityTable<double>&);

}
