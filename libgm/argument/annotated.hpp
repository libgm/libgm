#ifndef LIBGM_ANNOTATED_HPP
#define LIBGM_ANNOTATED_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <utility>

namespace libgm {

  /**
   * A class that represents an object annotated with a domain.
   */
  template <typename Arg, typename T>
  struct annotated {

    //! The annotating domain.
    libgm::domain<Arg> domain;

    //! The annotated object.
    T object;

    //! Default constructor
    annotated()
      : domain(), object() { }

    //! Sets the annotating domain an annotated object.
    annotated(const libgm::domain<Arg>& domain, const T& object)
      : domain(domain), object(object) { }

    //! Moves the annotating domain and annotated object.
    annotated(libgm::domain<Arg>&& domain, T&& object)
      : domain(std::move(domain)), object(std::move(object)) { }

    //! Serializes cluster and property.
    void save(oarchive& ar) const {
      ar << domain << object;
    }

    //! Deserializes cluster and property.
    void load(iarchive& ar) {
      ar >> domain >> object;
    }

    friend bool operator==(const annotated& a, const annotated& b) {
      return a.domain == b.domain && a.object == b.object;
    }

    friend bool operator!=(const annotated& a, const annotated& b) {
      return a.domain != b.domain || a.object != b.object;
    }

    friend std::ostream& operator<<(std::ostream& out, const annotated& a) {
      out << a.domain << ' ' << a.object;
      return out;
    }

  }; // struct annotated

} // namespace libgm

#endif
