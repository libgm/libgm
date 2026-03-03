#include <libgm/archives.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/impl/probability_matrix.hpp>

namespace libgm {

template class ProbabilityMatrix<float>;
template class ProbabilityMatrix<double>;

}

CEREAL_REGISTER_TYPE(libgm::ProbabilityMatrix<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::ProbabilityMatrix<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::ProbabilityMatrix<float>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::ProbabilityMatrix<double>::Impl);
