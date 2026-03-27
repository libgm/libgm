#pragma once

#include <chrono>

struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point start = clock::now();

  void restart() {
    start = clock::now();
  }

  double elapsed() const {
    return std::chrono::duration<double>(clock::now() - start).count();
  }
};
