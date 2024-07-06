namespace libgm {

template <typename R>
struct ModelEntropy {
  /**
   * Computes the entropy over a subset of arguments.
   */
  virtual R entropy(const Domain& domain) const = 0;

  /**
   * Computes the conditional entropy H(Y | X), where Y, X are subsets
   * of the arguments of this model.
   * \todo see if we can optimize this
   */
  R conditional_entropy(const Domain& y, const Domain& x) const {
    return entropy(x + y) - entropy(x);
  }

  /**
   * Computes the mutual information I(A ; B) between two subsets of*
   * arguments of this model.
   */
  R mutual_information(const Domain& a, const Domain& b) const {
    return entropy(a) + entropy(b) - entropy(a + b);
  }

  /**
   * Computes the conditional mutual information I(A; B | C),
   * where A,B,C must be subsets of the arguments of this model.
   * This is computed using I(A; B | C) = H(A | C) - H(A | B, C).
   *
   * @param base   Base of logarithm.
   * @return double representing the conditional mutual information.
   */
  R conditional_mutual_information(const Domain& a, const Domain& b, const Domain& c) const {
    return conditional_entropy(a, c) - conditional_entropy(a, b + c);
  }
};

} // namespace libgm
