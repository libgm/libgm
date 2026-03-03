#include "vector_assignment.hpp"

#include <libgm/assignment/impl/vector_assignment.hpp>
#include <libgm/archives.hpp>

#include <cereal/types/polymorphic.hpp>

namespace libgm {

template class VectorAssignment<float>;
template class VectorAssignment<double>;

} // namespace libgm

CEREAL_REGISTER_TYPE(libgm::VectorAssignment<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::VectorAssignment<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::VectorAssignment<float>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::VectorAssignment<double>::Impl);
