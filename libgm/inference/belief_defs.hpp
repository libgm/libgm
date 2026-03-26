#pragma once

#include <functional>

namespace libgm {

template <typename F>
using BeliefUpdate = std::function<void(F&, F)>;

template <typename F>
using BeliefDiff = std::function<typename F::real_type(const F&, const F&)>;

}