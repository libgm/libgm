#pragma once

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <iostream>
#include <memory>

namespace libgm {

/**
 * The base class of all the objects using a pointer-to-implementation (PImpl) pattern.
 *
 * The classes D derived from Object should specify their own implementation, by deriving from
 * the Object::Impl class (typically as a forward declared class D::Impl, implemented separately).
 * The classes D driving from Object must not add any data members directly. Instead, they should
 * add data to D::Impl. Furthermore, the subclasses D should not have any virtual members; instead
 * the functions inside D::Impl can be virtual.
 */
class Object {
public:
  struct Impl {
    /// Virtual destructor.
    virtual ~Impl();

    /// Returns the deep copy (clone) of this object.
    virtual Impl* clone() const = 0;

    /// Prints this object ot an output stream.
    virtual void print(std::ostream& out) const = 0;

    /// Saves this object to an output archive.
    virtual void save(oarchive& ar) const = 0;

    /// Load this object from an input archive.
    virtual void load(iarchive& ar) = 0;
  };

  /// Default constructor. Constructs an empty object.
  Object() = default;

  /// Constructs an object with the given implementation.
  explicit Object(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

  /// Copy constructor. Delegates to Impl::clone.
  Object(const Object& other);

  /// Move constructor.
  Object(Object&& other) = default;

  /// Copy assignment. Delegates to Impl::clone.
  Object& operator=(const Object& other);

  /// Move assignment.
  Object& operator=(Object&& other) = default;

  /// Returns true if the object contains data (its impl is not null).
  explicit operator bool() const { return bool(impl_); }

  /// Saves the object to an output archive.
  void save(oarchive& ar) const;

  /// Load the object from an input archive.
  void load(iarchive& ar);

  /// Prints the object to an output stream. Delegates to Impl::print.
  friend std::ostream& operator<<(std::ostream& out, const Object& object);

protected:
  std::unique_ptr<Impl> impl_;

}; // class Object

/// A unique pointer to the (weakly typed) implementation of an object.
using ImplPtr = std::unique_ptr<Object::Impl>;

} // namespace libgm
