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
    Index row;
    Index col;

    //! Default constructor; sets the vertex to 0.
    grid_vertex()
      : row(), col() { }

    //! Constructs the vertex with given coordinates.
    grid_vertex(Index row, Index col)
      : row(row), col(col) { }

    //! Saves the vertex to an output stream.
    void save(oarchive& ar) const {
      ar << row << col;
    }

    //! Loads the vertex from an input stream.
    void load(iarchive& ar) {
      ar >> row >> col;
    }

    //! Swaps two vertices.
    friend void swap(grid_vertex& u, grid_vertex& v) {
      using std::swap;
      swap(u.row, v.row);
      swap(u.col, v.col);
    }

    //! Returns the sum of the coordinates.
    Index sum() const {
      return row + col;
    }

    //! Returns the vertex (row - 1, col).
    grid_vertex above() const {
      return { row - 1, col };
    }

    //! Returns the vertex (row + 1, col).
    grid_vertex below() const {
      return { row + 1, col };
    }

    //! Returns the vertex (row, col - 1).
    grid_vertex left() const {
      return { row, col - 1 };
    }

    //! Returns the vertex (row, col + 1).
    grid_vertex right() const {
      return { row, col + 1 };
    }

    //! Returns the coordinate-wise sum of two vertices.
    friend grid_vertex operator+(grid_vertex u, grid_vertex v) {
      return { u.row + v.row, u.col + v.col };
    }

    //! Returns the coordinate-wise difference of two vertices.
    friend grid_vertex operator-(grid_vertex u, grid_vertex v) {
      return { u.row - v.row, u.col - v.col };
    }

    //! Returns the grid_vertex, whose coordinates are offset by a value.
    friend grid_vertex operator+(grid_vertex u, Index a) {
      return { u.row + a, u.col + a };
    }

    //! Returns the grid_vertex, whose coordinates are offset by a value.
    friend grid_vertex operator-(grid_vertex u, Index a) {
      return { u.row - a, u.col - a };
    }

    //! Returns the element-wise absolute value of a vertex.
    friend grid_vertex abs(grid_vertex u) {
      using std::abs;
      return { abs(u.row), abs(u.col) };
    }

    //! Returns true if two vertices are equal.
    friend bool operator==(grid_vertex u, grid_vertex v) {
      return u.row == v.row && u.col == v.col;
    }

    //! Returns true if two vertices are not equal.
    friend bool operator!=(grid_vertex u, grid_vertex v) {
      return u.row != v.row || u.col != v.col;
    }

    //! Returns true if both coordinates of u are less than or equal to v.
    friend bool operator<=(grid_vertex u, grid_vertex v) {
      return u.row <= v.row && u.col <= v.col;
    }

    //! Returns true if both coordinates of u are greater than or equal to v.
    friend bool operator>=(grid_vertex u, grid_vertex v) {
      return u.row >= v.row && u.col >= v.col;
    }

    //! Returns true if both coordinates of u are less than those of v.
    friend bool operator<(grid_vertex u, grid_vertex v) {
      return u.row < v.row && u.col < v.col;
    }

    //! Returns true if both coordinates of u are greater than those of v.
    friend bool operator>(grid_vertex u, grid_vertex v) {
      return u.row > v.row && u.col > v.col;
    }

    //! Returns the number of coordinates where u is les than v.
    friend std::size_t count_less(grid_vertex u, grid_vertex v) {
      return std::size_t(u.row < v.row) + std::size_t(u.col < v.col);
    }

    //! Returns the number of coordinates where u is les than v.
    friend std::size_t count_greater(grid_vertex u, grid_vertex v) {
      return std::size_t(u.row > v.row) + std::size_t(u.col > v.col);
    }

    //! Prints a vertex to an output stream.
    friend std::ostream& operator<<(std::ostream& out, grid_vertex v) {
      out << '(' << v.row << ',' << v.col << ')';
      return out;
    }

  } // namespace grid_vertex

} // namespace libgm

#endif
