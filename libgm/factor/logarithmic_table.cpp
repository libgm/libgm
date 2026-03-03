#include <libgm/archives.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/impl/logarithmic_table.hpp>

namespace libgm {

template class LogarithmicTable<float>;
template class LogarithmicTable<double>;

}

CEREAL_REGISTER_TYPE(libgm::LogarithmicTable<float>::Impl);
CEREAL_REGISTER_TYPE(libgm::LogarithmicTable<double>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::LogarithmicTable<float>::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::LogarithmicTable<double>::Impl);
