#include <libgm/archives.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/impl/logarithmic_matrix.hpp>

namespace libgm {

template class LogarithmicMatrix<float>;
template class LogarithmicMatrix<double>;

}

CEREAL_REGISTER_TYPE(libgm::LogarithmicMatrix<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::LogarithmicMatrix<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::LogarithmicMatrix<float>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::LogarithmicMatrix<double>::Impl);
