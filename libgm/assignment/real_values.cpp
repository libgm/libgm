#include <libgm/assignment/impl/real_values.hpp>
#include <libgm/archives.hpp>

#include <cereal/types/polymorphic.hpp>

namespace libgm {

template class RealValues<float>;
template class RealValues<double>;

}

CEREAL_REGISTER_TYPE(libgm::RealValues<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::RealValues<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::RealValues<float>::Impl)
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::RealValues<double>::Impl)
