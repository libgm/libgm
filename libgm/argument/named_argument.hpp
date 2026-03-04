#pragma once

#include <libgm/argument/argument.hpp>

#include <libgm/archives.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace libgm {

class NamedFactory;

class NamedArgument final : public Argument {
public:
  NamedArgument();
  explicit NamedArgument(std::string name, NamedFactory& factory);

  bool less(const Argument& other) const override;
  void print(std::ostream& out) const override;

  template <typename Archive>
  void save(Archive& ar) const;

  template <typename Archive>
  void load(Archive& ar);

  std::string name;
  NamedFactory* factory;
};

class NamedFactory {
public:
  explicit NamedFactory(std::string name_space = "");
  ~NamedFactory();

  NamedFactory(const NamedFactory&) = delete;
  NamedFactory& operator=(const NamedFactory&) = delete;
  NamedFactory(NamedFactory&&) = delete;
  NamedFactory& operator=(NamedFactory&&) = delete;

  Arg make(std::string name);
  void register_argument(const std::shared_ptr<NamedArgument>& argument);
  void clear();
  bool operator<(const NamedFactory& other) const;

  const std::string& name_space() const {
    return namespace_;
  }

  static NamedFactory* find(const std::string& name_space);
  static NamedFactory& default_factory();

private:
  std::string namespace_;
  std::unordered_map<std::string, std::shared_ptr<NamedArgument>> storage_;
  static std::unordered_map<std::string, NamedFactory*> registry_;
};

template <typename Archive>
void NamedArgument::save(Archive& ar) const {
  ar(name, factory->name_space());
}

template <typename Archive>
void NamedArgument::load(Archive& ar) {
  std::string name_space;
  ar(name, name_space);

  NamedFactory* decoded_factory = name_space.empty()
      ? &NamedFactory::default_factory()
      : NamedFactory::find(name_space);
  if (!decoded_factory) {
    throw std::invalid_argument("No NamedFactory registered for namespace: " + name_space);
  }
  factory = decoded_factory;
  factory->register_argument(std::static_pointer_cast<NamedArgument>(shared_from_this()));
}

} // namespace libgm
