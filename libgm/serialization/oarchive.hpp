#ifndef LIBGM_OARCHIVE_HPP
#define LIBGM_OARCHIVE_HPP

#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <unordered_set>

#include <boost/noncopyable.hpp>

#define LIBGM_SERIALIZE_CHAR(src_type)      \
  oarchive& operator<<(const src_type c) { \
    serialize_char(c);                     \
    return *this;                          \
  }

#define LIBGM_SERIALIZE_INT(src_type)        \
  oarchive& operator<<(const src_type x) {  \
    serialize_int(x);                       \
    return *this;                           \
  }

namespace libgm {

  /**
   * A class for serializing data in the native binary format.
   * The class is used in conjunction with operator<<. By default,
   * operator<< throws an exception if the write operation fails.
   */
  class oarchive : boost::noncopyable {
  public:
    //! Constructs an output archive with the given stream.
    oarchive(std::ostream& out)
      : out_(&out), bytes_(0) {}
    
    //! Sets the stream and resets the bytes to 0.
    void reset(std::ostream& out) {
      out_ = &out;
      bytes_ = 0;
    }

    //! Returns the number of bytes written.
    size_t bytes() const {
      return bytes_;
    }

    //! Throws an exception if the output stream indicates failure.
    void check() {
      if (out_->fail()) {
        throw std::runtime_error("oarchive: Stream operation failed!");
      }
    }

    //! Serializes a single character using 1 byte.
    void serialize_char(const char c) {
      out_->put(c);
      ++bytes_;
      check();
    }

    //! Serializes a 64-bit integer.
    void serialize_int(const int64_t x) {
      out_->write(reinterpret_cast<const char*>(&x), sizeof(int64_t));
      bytes_ += sizeof(int64_t);
      check();
    }

    //! Serializes a buffer of known length in bytes.
    void serialize_buf(const void* buf, const size_t length) {
      if (length == 0) { return; }
      out_->write(reinterpret_cast<const char*>(buf), length);
      bytes_ += length;
      check();
    }

    //! Serializes a range, automatically computing its length.
    template <typename ForwardIterator>
    void serialize_range(ForwardIterator it, ForwardIterator end) {
      serialize_int(std::distance(it, end));
      for (; it != end; ++it) {
        *this << *it;
      }
    }

    //! Serializes a range with known length.
    template <typename InputIterator>
    void serialize_range(InputIterator it, InputIterator end, size_t len) {
      serialize_int(len);
      size_t count = 0;
      for (; it != end; ++it) {
        *this << *it;
        ++count;
      }
      assert(count == len);
    }

    //! Serializes a dynamically allocated object.
    template <typename Object>
    void serialize_dynamic(const Object* obj) {
      if (obj == nullptr) {
        serialize_int(0);
      } else {
        auto p = dynamic_.emplace(obj, dynamic_.size() + 1);
        serialize_int(p.first->second);   // the ID (existing or new)
        if (p.second) { obj->save(*this); } // if newly inserted, serialize obj
      }
    }

    LIBGM_SERIALIZE_CHAR(char);
    LIBGM_SERIALIZE_CHAR(unsigned char);
    LIBGM_SERIALIZE_CHAR(bool);

    LIBGM_SERIALIZE_INT(int);
    LIBGM_SERIALIZE_INT(long);
    LIBGM_SERIALIZE_INT(long long);
    LIBGM_SERIALIZE_INT(unsigned long);
    LIBGM_SERIALIZE_INT(unsigned int);
    LIBGM_SERIALIZE_INT(unsigned long long);

    oarchive& operator<<(const float x) {
      out_->write(reinterpret_cast<const char*>(&x), sizeof(float));
      bytes_ += sizeof(float);
      check();
      return *this;
    }

    oarchive& operator<<(const double x) {
      out_->write(reinterpret_cast<const char*>(&x), sizeof(double));
      bytes_ += sizeof(double);
      check();
      return *this;
    }

  private:
    //! The stream to which we write data.
    std::ostream* out_; 

    //! The number of serialized bytes.
    size_t bytes_;

    //! A map that stores for each dynamically allocated object its ID in archive.
    std::unordered_map<const void*, size_t> dynamic_;

  }; // class oarchive

  /**
   * Catch all serializer that invokes save() member of the class T.
   * \relates oarchive
   */
  template <typename T>
  oarchive& operator<<(oarchive& a, const T& t) {
    t.save(a);
    return a;
  }

} // namespace libgm

#endif
