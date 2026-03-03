#include <libgm/archives.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/factor/impl/probability_vector.hpp>

namespace libgm {

template class ProbabilityVector<float>;
template class ProbabilityVector<double>;

}

CEREAL_REGISTER_TYPE(libgm::ProbabilityVector<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::ProbabilityVector<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::ProbabilityVector<float>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::ProbabilityVector<double>::Impl);
