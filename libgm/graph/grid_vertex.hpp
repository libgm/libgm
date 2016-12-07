#ifndef LIBGM_GRID_VERTEX_HPP
#define LIBGM_GRID_VERTEX_HPP

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <cstdlib>
#include <iosfwd>
#include <utility>

namespace libgm {

  /**
   * A vertex of a 2D grid.
   *
   * \tparam Index
   *         The type representing the grid index. Must be a signed integer.
   */
  template <typename Index = int>
  struct grid_vertex {
    Index x;
    Index y;

    //! Default constructor; sets the vertex to 0.
    grid_vertex()
      : x(), y() { }

    //! Constructs the vertex with given coordinates.
    grid_vertex(Index x, Index y)
      : x(x), y(y) { }

    //! Saves the vertex to an output stream.
    void save(oarchive& ar) const {
      ar << x << y;
    }

    //! Loads the vertex from an input stream.
    void load(iarchive& ar) {
      ar >> x >> y;
    }

    //! Swaps two vertices.
    friend void swap(grid_vertex& u, grid_vertex& v) {
      using std::swap;
      swap(u.x, v.x);
      swap(u.y, v.y);
    }

    //! Returns the sum of the coordinates.
    Index sum() const {
      return x + y;
    }

    //! Returns the vertex (x+1, y).
    grid_vertex right() const {
      return { x + 1, y };
    }

    //! Returns the vertex (x, y+1).
    grid_vertex above() const {
      return { x, y + 1 };
    }

    //! Returns the linear index of a vertex given the grid size.
    std::size_t linear(grid_vertex size) const {
      return std::size_t(x) * std::size_t(size.y) + y;
    }

    //! Returns the coordinate-wise sum of two vertices.
    friend grid_vertex operator+(grid_vertex u, grid_vertex v) {
      return { u.x + v.x, u.y + v.y };
    }

    //! Returns the coordinate-wise difference of two vertices.
    friend grid_vertex operator-(grid_vertex u, grid_vertex v) {
      return { u.x - v.x, u.y - v.y };
    }

    //! Returns the grid_vertex, whose coordinates are offset by a value.
    friend grid_vertex operator+(grid_vertex u, Index a) {
      return { u.x + a, u.y + a };
    }

    //! Returns the grid_vertex, whose coordinates are offset by a value.
    friend grid_vertex operator-(grid_vertex u, Index a) {
      return { u.x - a, u.y - a };
    }

    //! Returns the element-wise absolute value of a vertex.
    friend grid_vertex abs(grid_vertex u) {
      using std::abs;
      return { abs(u.x), abs(u,y) };
    }
    //! Returns true if two vertices are equal.
    friend bool operator==(grid_vertex u, grid_vertex v) {
      return u.x == v.x && u.y == v.y;
    }

    //! Returns true if two vertices are not equal.
    friend bool operator!=(grid_vertex u, grid_vertex v) {
      return u.x != v.x || u.y != v.y;
    }

    //! Returns true if both coordinates of u are less than or equal to v.
    friend bool operator<=(grid_vertex u, grid_vertex v) {
      return u.x <= v.x && u.y <= v.y;
    }

    //! Returns true if both coordinates of u are greater than or equal to v.
    friend bool operator>=(grid_vertex u, grid_vertex v) {
      return u.x >= v.x && u.y >= v.y;
    }

    //! Returns true if both coordinates of u are less than those of v.
    friend bool operator<(grid_vertex u, grid_vertex v) {
      return u.x < v.x && u.y < v.y;
    }

    //! Returns true if both coordinates of u are greater than those of v.
    friend bool operator>(grid_vertex u, grid_vertex v) {
      return u.x > v.x && u.y > v.y;
    }

    //! Returns the number of coordinates where u is les than v.
    friend std::size_t count_less(grid_vertex u, grid_vertex v) {
      return std::size_t(u.x < v.x) + std::size_t(u.y < v.y);
    }

    //! Returns the number of coordinates where u is les than v.
    friend std::size_t count_greater(grid_vertex u, grid_vertex v) {
      return std::size_t(u.x > v.x) + std::size_t(u.y > v.y);
    }

    //! Prints a vertex to an output stream.
    friend std::ostream& operator<<(std::ostream& out, grid_vertex v) {
      out << '(' << v.x << ',' << v.y << ')';
      return out;
    }

  } // namespace grid_vertex

} // namespace libgm

#endif
