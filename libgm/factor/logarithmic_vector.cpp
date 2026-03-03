#include <libgm/archives.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/impl/logarithmic_vector.hpp>

namespace libgm {

template class LogarithmicVector<float>;
template class LogarithmicVector<double>;

}

CEREAL_REGISTER_TYPE(libgm::LogarithmicVector<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::LogarithmicVector<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::LogarithmicVector<float>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::LogarithmicVector<double>::Impl);
