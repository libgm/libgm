#ifndef LIBGM_FUNCTIONS_HPP
#define LIBGM_FUNCTIONS_HPP

namespace libgm {

  // Arguments
  void concat();
  void intersect();
  void disjoint();
  void equivalent();
  void subset();
  void supserset();
  void compatible();

  // Factors
  void pow();
  void max();
  void min();
  void weighted_update();
  void cross_entropy();
  void kl_divergence();
  void js_divergence();
  void sum_diff();
  void max_diff();

  // Optimization
  void update();

} // namespace libgm

#endif
