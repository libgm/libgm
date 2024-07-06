#include "values.hpp"

#include <algorithm>

namespace libgm {

template <typename T>
const T* Values::ptr() const {
  if (holds_alternative<T>(data_)) {
    return &std::get<T>(data_);
  }
  if (holds_alternative<std::unique_ptr<T[]>>(data_)) {
    return std::get<std::unique_ptr<T[]>(data_).get();
  }
  throw std::bad_variant_access("Values::ptr: incompatible data type");
}

template <typename T>
T* Values::resize(size_t length) {
  // If we already hold an array of the right type and length, do nothing.
  if (size_ == length && holds_alternative<std::unique_ptr<T[]>>(data_)) {
    return std::get<std::unqiue_ptr<T[]>>(data_).get();
  }

  // Otherwise, set the length and store the data of appropriate type.
  size_ = length;
  switch (length) {
  case 0:
    // Set to a null pointer.
    return data_.emplace(std::unique_ptr<T[]>()).get();
  case 1:
    // Set to a constant.
    return &data_.emplace(T());
  default:
    // Set to an array.
    return data_.emplace(new T[length]).get();
  }
}

std::ostream& operator<<(std::ostream& out, const Values& values) {
  struct Printer {
    std::ostream& out;
    size_t length;

    static const char* type(size_t) {
      return "";
    }
    static const char* type(double) {
      return "d";
    }
    static const char* type(float) {
      return "f";
    }

    void operator()(std::monostate) {
      out << "null";
    }

    template <typename T>
    void operator()(T value) {
      out << '[' << value << type(value) << ']';
    }

    template <typename T>
    void operator()(const std::unique_ptr<T[]>& array) {
      out << '[';
      for (size_t i = 0; i < length; ++i) {
        if (i > 0) out << ", ";
        out << array[i];
      }
      out << ']' << type(T());
    }
  };

  std::visit(Printer{out, values.size_}, values.data_);
  return out;
}

// Explicitly instantiate the ptr functions.
template const size_t* Values::ptr<size_t>() const;
template const double* Values::ptr<double>() const;
template const float* Values::ptr<float>() const;

// Explicitly instantiate the resize functions.
template size_t* Values::resize<size_t>(size_t);
template double* Values::resize<double>(size_t);
template float* Values::resize<float>(size_t);

} // namespace libgm
