#include <libgm/archives.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/impl/canonical_gaussian.hpp>

namespace libgm {

template class CanonicalGaussian<float>;
template class CanonicalGaussian<double>;

}

CEREAL_REGISTER_TYPE(libgm::CanonicalGaussian<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::CanonicalGaussian<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::CanonicalGaussian<float>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::CanonicalGaussian<double>::Impl);
