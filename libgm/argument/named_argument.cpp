#include "named_argument.hpp"

#include <stdexcept>
#include <tuple>

namespace libgm {

std::unordered_map<std::string, NamedFactory*> NamedFactory::registry_;

NamedArgument::NamedArgument()
  : factory(&NamedFactory::default_factory()) {}

NamedArgument::NamedArgument(std::string name, NamedFactory& factory)
  : name(std::move(name))
  , factory(&factory) {}

bool NamedArgument::less(const Argument& other) const {
  const auto& rhs = static_cast<const NamedArgument&>(other);
  return std::tie(*factory, name) < std::tie(*rhs.factory, rhs.name);
}

void NamedArgument::print(std::ostream& out) const {
  out << name;
}

NamedFactory::NamedFactory(std::string name_space)
  : namespace_(std::move(name_space)) {
  auto [it, inserted] = registry_.emplace(namespace_, this);
  if (!inserted) {
    throw std::invalid_argument("NamedFactory namespace already registered: " + namespace_);
  }
}

NamedFactory::~NamedFactory() {
  auto it = registry_.find(namespace_);
  if (it != registry_.end() && it->second == this) {
    registry_.erase(it);
  }
}

Arg NamedFactory::make(std::string name) {
  if (name.empty()) {
    return Arg();
  }
  auto it = storage_.find(name);
  if (it == storage_.end()) {
    auto argument = std::make_shared<NamedArgument>(std::move(name), *this);
    it = storage_.emplace(argument->name, std::move(argument)).first;
  }
  return Arg(it->second.get());
}

void NamedFactory::register_argument(const std::shared_ptr<NamedArgument>& argument) {
  if (!argument) {
    return;
  }
  if (argument->factory != this) {
    throw std::invalid_argument("NamedArgument factory mismatch during registration");
  }
  auto [it, inserted] = storage_.emplace(argument->name, argument);
  if (!inserted) {
    throw std::invalid_argument("NamedArgument already registered: " + argument->name);
  }
}

void NamedFactory::clear() {
  storage_.clear();
}

bool NamedFactory::operator<(const NamedFactory& other) const {
  if (namespace_ != other.namespace_) {
    return namespace_ < other.namespace_;
  }
  return this < &other;
}

NamedFactory& NamedFactory::default_factory() {
  static NamedFactory factory("");
  return factory;
}

NamedFactory* NamedFactory::find(const std::string& name_space) {
  auto it = registry_.find(name_space);
  if (it == registry_.end()) {
    return nullptr;
  }
  return it->second;
}

} // namespace libgm

CEREAL_REGISTER_TYPE(libgm::NamedArgument);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Argument, libgm::NamedArgument);
