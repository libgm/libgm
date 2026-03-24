#pragma once

#include <libgm/argument/domain.hpp>

namespace libgm {

template <typename Factor>
struct Annotated {
  Factor factor;
  Domain domain;
};

template <typename Factor>
Annotated<std::decay_t<Factor>> annotate(Factor&& factor, Domain domain) {
  return {std::forward<Factor>(factor), std::move(domain)};
}

}
