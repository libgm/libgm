#ifndef LIBGM_EIGEN_HYBRID_HPP
#define LIBGM_EIGEN_HYBRID_HPP

#include <libgm/datastructure/hybrid_index.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/eigen/uint.hpp>
#include <libgm/math/eigen/subvector.hpp>
#include <libgm/parser/range_io.hpp>

#include <algorithm>

namespace libgm {

  /**
   * A class that represents a dense vector with an integral and
   * a real component.
   *
   * \tparam T the real type representing the vector elements
   */
  template <typename T = double>
  struct hybrid_vector
    : public uint_vector, real_vector<T> {

    //! Default constructor; constructs empty vector.
    hybrid_vector() { }

    /**
     * Constructs an vector with the integral and real components
     * of specified lengths. The integral component is initialized
     * to a 0, but the vector one is not.
     */
    hybrid_vector(std::size_t nint, std::size_t nreal)
      : uint_vector(nint),
        real_vector<T>(nreal) { }

    //! Constructs a vector with the given integral and real components.
    hybrid_vector(const uint_vector& uint,
                  const real_vector<T>& real)
      : uint_vector(uint),
        real_vector<T>(real) { }

    //! Constructs an vector with the given finite and vector components.
    hybrid_vector(std::initializer_list<std::size_t> uint,
                  std::initializer_list<T> real)
      : uint_vector(uint),
        real_vector<T>(real.size()) {
      std::copy(real.begin(), real.end(), this->real().data());
    }

    //! Returns the integral component..
    uint_vector& uint() {
      return *this;
    }

    //! Returns the integral component.
    const uint_vector& uint() const {
      return *this;
    }

    //! Returns the real component.
    real_vector<T>& real() {
      return *this;
    }

    //! Returns the real component.
    const real_vector<T>& real() const {
      return *this;
    }

    //! Returns the length of the integral component.
    std::size_t uint_size() const {
      return uint().size();
    }

    //! Returns the length of the real component.
    std::size_t real_size() const {
      return real().size();
    }

    //! Swaps the contents of two hybrid indices.
    friend void swap(hybrid_vector& a, hybrid_vector& b) {
      a.uint().swap(b.uint());
      a.real().swap(b.real());
    }

    //! Resizes the vector to the given lengths. The original values may be lost.
    void resize(std::size_t nint, std::size_t nreal) {
      uint().resize(nint);
      real().resize(nreal);
    }

    //! Returns true if the two hybrid indices are equal
    bool operator==(const hybrid_vector& b) const {
      return uint() == b.uint() && real() == b.real();
    }

    //! Returns true if the two hybrid indices are not equal.
    bool operator!=(const hybrid_vector& b) const {
      return !(*this == b);
    }

    //! Prints a hybrid vector to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const hybrid_vector& v) {
      print_range(out, v.uint().begin(), v.uint().end(), '[', ' ', ']');
      out << ' ';
      print_range(out, v.real().data(), v.real().data() + v.real().size(),
                  '[', ' ', ']');
      return out;
    }

  }; // class hybrid_vector

  /**
   * A class that represents a dense matrix with an integral and
   * a real component.
   *
   * \tparam T the real type representing the vector elements
   */
  template <typename T = double>
  struct hybrid_matrix
    : public uint_matrix, real_matrix<T> {

    //! Default constructor; constructs empty matrix
    hybrid_matrix() { }

    /**
     * Constructs an vector with the integral and real components
     * of specified lengths. The components may not be initialized.
     */
    hybrid_matrix(std::size_t uint_rows, std::size_t uint_cols,
                  std::size_t real_rows, std::size_t real_cols)
      : uint_matrix(uint_rows, uint_cols),
        real_matrix<T>(real_rows, real_cols) { }

    //! Constructs a matrix with the given integral and real components.
    hybrid_matrix(Eigen::Ref<const uint_matrix> uint,
                  Eigen::Ref<const real_matrix<T> > real)
      : uint_matrix(uint),
        real_matrix<T>(real) { }

    //! Returns the integral component..
    uint_matrix& uint() {
      return *this;
    }

    //! Returns the integral component.
    const uint_matrix& uint() const {
      return *this;
    }

    //! Returns the real component.
    real_matrix<T>& real() {
      return *this;
    }

    //! Returns the real component.
    const real_matrix<T>& real() const {
      return *this;
    }

    /**
     * Returns the number of columns of the matrix, provided that the
     * discrete and continuous components have the same number of columns.
     */
    std::size_t cols() const {
      assert(uint().cols() == real().cols());
      return uint().cols();
    }

    //! Swaps the contents of two hybrid indices.
    friend void swap(hybrid_matrix& a, hybrid_matrix& b) {
      a.uint().swap(b.uint());
      a.real().swap(b.real());
    }

    //! Returns true if the two hybrid indices are equal
    bool operator==(const hybrid_matrix& b) const {
      return uint() == b.uint() && real() == b.real();
    }

    //! Returns true if the two hybrid indices are not equal.
    bool operator!=(const hybrid_matrix& b) const {
      return !(*this == b);
    }

    //! Prints a hybrid matrix to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const hybrid_matrix& v) {
      out << v.uint() << std::endl
          << v.real();
      return out;
    }

  }; // class hybrid_matrix

  /**
   * A class that represents the elements of a hybrid matrix.
   */
  template <typename T>
  class hybrid_subvector {
  public:
    hybrid_subvector(const hybrid_matrix<T>& m,
                     const hybrid_index& rows,
                     std::size_t start_col)
      : uint_(elements(m.uint(), rows.uint, start_col)),
        real_(elements(m.real(), rows.real, start_col)) { }

    void eval_to(hybrid_vector<T>& result) const {
      uint_.eval_to(result.uint());
      real_.eval_to(result.real());
    }

  private:
    subvector<const Eigen::Matrix<std::size_t, Eigen::Dynamic, 1> > uint_;
    subvector<const real_vector<T> > real_;
  };

  /**
   * Selects the elements of a hybrid matrix.
   */
  template <typename T>
  hybrid_subvector<T> elements(const hybrid_matrix<T>& m,
                               const hybrid_index& rows,
                               std::size_t start_col) {
    return { m, rows, start_col };
  }

} // namespace libgm

#endif
