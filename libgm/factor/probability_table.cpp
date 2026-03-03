#include <libgm/archives.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/impl/probability_table.hpp>

namespace libgm {

template class ProbabilityTable<float>;
template class ProbabilityTable<double>;

}

CEREAL_REGISTER_TYPE(libgm::ProbabilityTable<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::ProbabilityTable<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::ProbabilityTable<float>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::ProbabilityTable<double>::Impl);
