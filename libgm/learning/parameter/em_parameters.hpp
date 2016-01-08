#ifndef LIBGM_EM_PARAMETERS_HPP
#define LIBGM_EM_PARAMETERS_HPP

namespace libgm {

  /**
   * A struct that represents the parameters of an EM algorithm.
   * \tparam Regul a type that represents regularization parameters
   */
  template <typename Regul = double, typename RealType = double>
  struct em_parameters {
    Regul regul;
    std::size_t max_iter;
    RealType tol;
    bool verbose;
    unsigned seed;
    em_parameters(const Regul& regul = Regul(),
                  size_t max_iter = 100,
                  RealType tol = 1e-6,
                  bool verbose = false,
                  unsigned seed = 0)
      : regul(regul),
        max_iter(max_iter),
        tol(tol),
        verbose(verbose),
        seed(seed) { }

  }; // struct em_parameters

} // namespace libgm

#endif
