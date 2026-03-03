#include "discrete_values.hpp"

#include <libgm/archives.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

namespace libgm {

struct DiscreteValues::Impl : Object::Impl {
  std::vector<size_t> values;

  Impl() = default;

  Impl(std::vector<size_t> values)
    : values(std::move(values)) {}

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(values);
  }

  // Object operations
  //--------------------------------------------------------------------------

  Impl* clone() const override {
    return new Impl(*this);
  }

  void print(std::ostream& out) const override {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) out << ", ";
      out << values[i];
    }
    out << ']';
  }
};

DiscreteValues::DiscreteValues(std::initializer_list<size_t> list)
  : Object(std::make_unique<Impl>(list)) {}

DiscreteValues::DiscreteValues(std::vector<size_t> values)
  : Object(std::make_unique<Impl>(std::move(values))) {}

DiscreteValues& DiscreteValues::operator=(std::vector<size_t> other) {
  if (impl_) {
    impl().values = std::move(other);
  } else {
    impl_.reset(new Impl(std::move(other)));
  }
  return *this;
}

size_t DiscreteValues::size() const {
  return impl().values.size();
}

size_t DiscreteValues::operator[](size_t pos) const {
  return impl().values[pos];
}

size_t& DiscreteValues::operator[](size_t pos) {
  return impl().values[pos];
}

const size_t* DiscreteValues::data() const {
  return impl().values.data();
}

const std::vector<size_t>& DiscreteValues::vec() const {
  return impl().values;
}

DiscreteValues::Impl& DiscreteValues::impl() {
  return static_cast<Impl&>(*impl_);
}

const DiscreteValues::Impl& DiscreteValues::impl() const {
  return static_cast<Impl&>(*impl_);
}

} // namespace libgm

CEREAL_REGISTER_TYPE(libgm::DiscreteValues::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::DiscreteValues::Impl)
