#ifndef LIBGM_HYBRID_INDEX_HPP
#define LIBGM_HYBRID_INDEX_HPP

#include <libgm/global.hpp>
#include <libgm/datastructure/finite_index.hpp>
#include <libgm/math/eigen/dynamic.hpp>

namespace libgm {

  /**
   * A class that represents a dense index with a finite and vector
   * component. It can be viewed as an assignment to a vector of
   * finite and vector arguments.
   *
   * \tparam T the real type representing the vector elements
   */
  template <typename T>
  struct hybrid_index
    : public finite_index, dynamic_vector<T> {

    //! Default constructor; constructs empty index.
    hybrid_index() { }

    /**
     * Constructs an index with the specified finite and vector lengths.
     * The finite elements are initialized to 0, but the vector elements
     * are not.
     */
    hybrid_index(size_t nfinite, size_t nvector)
      : finite_index(nfinite),
        dynamic_vector<T>(nvector) { }

    //! Constructs an index with the given finite and vector component.
    hybrid_index(const finite_index& finite,
                 const dynamic_vector<T>& vector)
      : finite_index(finite),
        dynamic_vector<T>(vector) { }

    //! Returns the finite component of the index.
    finite_index& finite() {
      return *this;
    }

    //! Returns the finite component of the index.
    const finite_index& finite() const {
      return *this;
    }

    //! Returns the vector component of the index.
    dynamic_vector<T>& vector() {
      return *this;
    }

    //! Returns the vector component of the index.
    const dynamic_vector<T>& vector() const {
      return *this;
    }

    //! Returns the length of the finite component.
    size_t finite_size() const {
      return finite().size();
    }

    //! Returns the length fo the vector component.
    size_t vector_size() const {
      return vector().size();
    }

    //! Swaps the contents of two hybrid indices.
    friend void swap(hybrid_index& a, hybrid_index& b) {
      a.finite().swap(b.finite());
      a.vector().swap(b.vector());
    }

    //! Resizes the index to the given lengths. The original values may be lost.
    void resize(size_t nfinite, size_t nvector) {
      finite().resize(nfinite);
      vector().resize(nvector);
    }

    //! Returns true if the two hybrid indices are equal
    friend bool operator==(const hybrid_index& a, const hybrid_index& b) {
      return a.finite() == b.finite() && a.vector() == b.vector();
    }

    //! Returns true if the two hybrid indices are not equal.
    friend bool operator!=(const hybrid_index& a, const hybrid_index& b) {
      return !(a == b);
    }

  }; // struct hybrid_index

  /**
   * Prints a hybrid vector to an output stream.
   * \relates hybrid_index
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const hybrid_index<T>& v) {
    out << "[";
    for (size_t i = 0; i < v.finite().size(); ++i) {
      if (i > 0) { out << ' '; }
      out << v.finite()[i];
    }
    out << "] [";
    for (size_t i = 0; i < v.vector().size(); ++i) {
      if (i > 0) { out << ' '; }
      out << v.vector()[i];
    }
    out << "]";
    return out;
  }

} // namespace libgm

#endif
