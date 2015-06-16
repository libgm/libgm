#ifndef LIBGM_TEST_PREDICATES_HPP
#define LIBGM_TEST_PREDICATES_HPP

#include <algorithm>
#include <fstream>
#include <cstdio>

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

std::string temp_filename() {
  char filename[L_tmpnam+1];
#ifdef WIN32
  filename[0] = '.';
  tmpnam(filename + 1);
#else
  tmpnam(filename);
#endif
  return std::string(filename);
}

//! Serializes and deserializes a value and verifies they are equal
template <typename T>
boost::test_tools::predicate_result
serialize_deserialize(const T& value) {
  std::string filename = temp_filename();
  using std::ios_base;

  std::ofstream fout(filename.c_str(), ios_base::binary | ios_base::out);
  assert(fout);
  libgm::oarchive oa(fout);
  oa << value;
  fout.close();

  std::ifstream fin(filename.c_str(), ios_base::binary | ios_base::in);
  assert(fin);
  libgm::iarchive ia(fin);
  T value2;
  ia >> value2;
  fin.close();

  remove(filename.c_str());

  if(value != value2) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Serialization did not preserve the value [\n"
                     << value << "!="
                     << value2 << "]";
    return result;
  }
  return true;
}

//! Serializes and deserializes a value and verifies they are equal
template <typename T>
boost::test_tools::predicate_result
serialize_deserialize(const T& value, libgm::universe& u) {
  std::string filename = temp_filename();
  using std::ios_base;

  std::ofstream fout(filename.c_str(), ios_base::binary | ios_base::out);
  assert(fout);
  libgm::oarchive oa(fout);
  oa << value;
  fout.close();

  std::ifstream fin(filename.c_str(), ios_base::binary | ios_base::in);
  assert(fin);
  libgm::iarchive ia(fin);
  ia.universe(&u);
  T value2;
  ia >> value2;
  fin.close();

  remove(filename.c_str());

  if(value != value2) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Serialization did not preserve the value [\n"
                     << value << "!="
                     << value2 << "]";
    return result;
  }
  return true;
}

template <typename Range1, typename Range2>
boost::test_tools::predicate_result
range_equal(const Range1& x, const Range2& y) {
  auto x_it = x.begin(), x_end = x.end();
  auto y_it = y.begin(), y_end = y.end();
  std::size_t pos = 0;
  while (x_it != x_end && y_it != y_end) {
    if (*x_it != *y_it) {
      boost::test_tools::predicate_result result(false);
      result.message() << "the ranges differ at position " << pos;
      return result;
    }
    ++x_it;
    ++y_it;
  }
  if ((x_it == x_end) ^ (y_it == y_end)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "the ranges have different lengths ["
                     << std::distance(x.begin(), x.end()) << " != "
                     << std::distance(y.begin(), y.end()) << "]";
    return result;
  }
  return true;
}

#endif
