#pragma once

#include <memory>
#include <variant>

namespace libgm {

class Values {
public:
  // Default constructor, creates null values.
  Values = default();

  // Singleton construcors
  Values(size_t value) : data_(value), size_(1) {}
  Values(double value) : data_(value), size_(1) {}
  Values(float value) : data_(value), size_(1) {}

  // Singleton assignment
  Values& operator=(size_t value) { data_ = value; size_ = 1; return *this; }
  Values& operator=(double value) { data_ = value; size_ = 1; return *this; }
  Values& operator=(float value) { data_ = value; size_ = 1; return *this; }

  /// Converts to true if the values are not null (i.e., not default-constructed).
  explicit operator bool() const { return data_.index() > 0; }

  /// Returns the number of values stored (the length of the array).
  size_t size() const { return size_; }

  /// Returns a single value of the specified type.
  template <typename T>
  const T& get() const { return std::get<T>(data_); }

  /// Sets a single value and returns a reference.
  template <typename T>
  T& set(T value) { return data_.emplace(value); }

  /// Returns the pointer to data of the specified type.
  template <typename T>
  const T* ptr() const;

  /// Resizes the values to the given length and returns a pointer to the data.
  template <typename T>
  T* resize(size_t length);

  /// Prints the values to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const Values& values);

private:
  /// The value or a pointer to an array of values.
  std::variant<
    std::monostate,
    size_t,
    double,
    float,
    std::unique_ptr<size_t[]>,
    std::unique_ptr<double[]>,
    std::unique_ptr<float[]>
  > data_;

  /**
   * The number of stored values. If the values are not null,
   * - and this is 0, we store an null pointer of the correct type;
   * - and this is 1, we store a value of the correct type;
   * - otherwise, we store a non-null pointer of the correct type.
   */
  size_t size_ = 0;
};

} // namespace libgm
