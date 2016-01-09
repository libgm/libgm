#ifndef LIBGM_ID_HPP
#define LIBGM_ID_HPP

#include <libgm/functional/hash.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

namespace libgm {

  /**
   * A struct that represents an unsigned integer identifier. This effectively
   * acts as a strong typedef for std::size_t, and its needed in datastructures
   * such as cluster and factor graphs, in order not to confuse the ID with
   * the arguments (which could be an integer as well). Typically, the user
   * will not create the IDs; they will be assigned automatically in the graph
   * structures. However, if needed, it is possible to explicitly convert an
   * std::size_t to id_t.
   */
  struct id_t {
    //! The underlying identifier value.
    std::size_t val;

    //! Default constructor. Creates an ID with value 0.
    id_t() : val(0) { }

    //! Conversion from std::size_t.
    explicit id_t(std::size_t val) : val(val) { }

    //! Returns true if the identifier is not 0.
    explicit operator bool() const { return val != 0; }

    //! Saves the ID to an archive.
    void save(oarchive& ar) const { ar << val; }

    //! Loads the ID from an archive.
    void load(iarchive& ar) { ar >> val; }

    //! Advances the ID.
    id_t& operator++() { ++val; return *this; }

    //! Compares two IDs
    friend bool operator==(id_t x, id_t y) { return x.val == y.val; }

    //! Compares two IDs
    friend bool operator!=(id_t x, id_t y) { return x.val != y.val; }

    //! Compares two IDs
    friend bool operator<(id_t x, id_t y) { return x.val < y.val; }

    //! Compares two IDs
    friend bool operator<=(id_t x, id_t y) { return x.val <= y.val; }

    //! Compares two IDs
    friend bool operator>(id_t x, id_t y) { return x.val > y.val; }

    //! Compares two IDs
    friend bool operator>=(id_t x, id_t y) { return x.val >= y.val; }

    //! Computes the hash value of an ID.
    friend std::size_t hash_value(id_t id) { return id.val; }

    //! Prints the ID to an output stream
    friend std::ostream&
    operator<<(std::ostream& out, id_t id) { return out << id.val; }

  }; // struct id_t

  // Traits
  //============================================================================

  /**
   * A specialization of vertex_traits for id_t.
   */
  template <>
  struct vertex_traits<id_t> {
    //! Returns the default-constructed id_t.
    static id_t null() { return id_t(); }

    //! Returns a special "deleted" id_t.
    static id_t deleted() { return id_t(-1); }

    //! Prints the id.
    static void print(std::ostream& out, id_t id) {
      if (id.val == 0) {
        out << "null";
      } else if (id.val == -1) {
        out << "deleted";
      } else {
        out << id.val;
      }
    }

    //! The id_t uses the default hasher.
    typedef std::hash<id_t> hasher;
  };

} // namespace libgm


namespace std {

  template <>
  struct hash<libgm::id_t>
    : libgm::default_hash<libgm::id_t> { };

} // namespace std


#endif
