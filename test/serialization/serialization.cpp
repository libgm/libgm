#define BOOST_TEST_MODULE serialization
#include <boost/test/unit_test.hpp>

#include <libgm/serialization/list.hpp>
#include <libgm/serialization/map.hpp>
#include <libgm/serialization/serialize.hpp>
#include <libgm/serialization/set.hpp>
#include <libgm/serialization/vector.hpp>

#include <cassert>
#include <cstring>
#include <fstream>
#include <string>

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<int>);

BOOST_AUTO_TEST_CASE(basic_datatypes) {
  char t1         = 'z';
  bool t2         = true;
  int t3          = 10;
  int t4          = 18345;
  long t5         = 30921233;
  long long t6    = (long long)(t5)*100;
  float t7        = 10.35;
  double t8       = 3.14156;
  const char* t9  = "hello world";
  const char* t10 = "blue";

  char r1;
  bool r2;
  int r3;
  int r4;
  long r5;
  long long r6;
  float r7;
  double r8;
  char* r9 = NULL;
  char r10[10];

  // serialize t1-10
  std::ofstream f;
  f.open("test.bin", std::fstream::binary);
  oarchive a(f);
  a << t1 << t2 << t3 << t4 << t5 << t6 << t7 << t8 << t9;
  a.serialize_buf(t10, strlen(t10) + 1);
  f.close();

  // deserialize into r1-10
  std::ifstream g;
  g.open("test.bin", std::fstream::binary);
  iarchive b(g);
  b >> r1 >> r2 >> r3 >> r4 >> r5 >> r6 >> r7 >> r8;
  b >> r9;
  b.deserialize_buf(r10, strlen(t10) + 1);
  g.close();

  BOOST_CHECK_EQUAL(t1, r1);
  BOOST_CHECK_EQUAL(t2, r2);
  BOOST_CHECK_EQUAL(t3, r3);
  BOOST_CHECK_EQUAL(t4, r4);
  BOOST_CHECK_EQUAL(t5, r5);
  BOOST_CHECK_EQUAL(t6, r6);
  BOOST_CHECK_EQUAL(t7, r7);
  BOOST_CHECK_EQUAL(t8, r8);
  BOOST_CHECK_EQUAL(strcmp(t9, r9), 0);
  BOOST_CHECK_EQUAL(strcmp(t10, r10), 0);
}

BOOST_AUTO_TEST_CASE(vectors) {
  std::vector<int> v;
  for (int i = 0; i < 10; ++i) {
    v.push_back(i);
  }
  std::ofstream f;
  f.open("test.bin", std::fstream::binary);
  oarchive a(f);
  a << v;
  f.close();

  std::vector<int> w;
  std::ifstream g;
  iarchive b(g);
  g.open("test.bin", std::fstream::binary);
  b >> w;
  g.close();

  BOOST_CHECK_EQUAL(v, w);
}

struct member {
  int z;
  void save(oarchive& a) const {
    a << z;
  }
  void load(iarchive& a) {
    a >> z;
  }
};

struct test_class {
  int i;
  int j;
  std::vector<int> k;
  member l;
  void save(oarchive& a) const {
    a << i << j << k << l;
  }
  void load(iarchive& a) {
    a >> i >> j >> k >> l;
  }
};

BOOST_AUTO_TEST_CASE(classes) {
  // create a test class
  test_class t;
  t.i = 10;
  t.j = 20;
  t.k.push_back(30);

  // serialize
  std::ofstream f;
  f.open("test.bin", std::fstream::binary);
  oarchive a(f);
  a << t;
  f.close();

  // deserialize into t2
  test_class t2;
  std::ifstream g;
  g.open("test.bin", std::fstream::binary);
  iarchive b(g);
  b >> t2;
  g.close();

  // check
  BOOST_CHECK_EQUAL(t.i, t2.i);
  BOOST_CHECK_EQUAL(t.j, t2.j);
  BOOST_CHECK_EQUAL(t.k, t2.k);
}

BOOST_AUTO_TEST_CASE(vector_of_classes) {
  // create a vector of test classes
  std::vector<test_class> vt;
  vt.resize(10);
  for (std::size_t i = 0; i < 10; ++i) {
    vt[i].i = i;
    vt[i].j = i*21;
    vt[i].k.resize(10);
    vt[i].k[i] = i*51;
  }

  // serialize
  std::ofstream f;
  f.open("test.bin", std::fstream::binary);
  oarchive a(f);
  a << vt;
  f.close();

  // deserialize into vt2
  std::vector<test_class> vt2;
  std::ifstream g;
  g.open("test.bin", std::fstream::binary);
  iarchive b(g);
  b >> vt2;
  g.close();

  // check
  BOOST_CHECK_EQUAL(vt.size(), vt2.size());
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(vt[i].i, vt2[i].i);
    BOOST_CHECK_EQUAL(vt[i].j, vt2[i].j);
    BOOST_CHECK_EQUAL(vt[i].k, vt2[i].k);
  }
}

BOOST_AUTO_TEST_CASE(vector_of_strings) {
  std::string x = "Hello world";
  std::string y = "This is a test";
  std::vector<std::string> v;
  v.push_back(x);
  v.push_back(y);

  std::ofstream f;
  f.open("test.bin", std::fstream::binary);
  oarchive a(f);
  a << v;
  f.close();

  std::vector<std::string> v2;
  std::ifstream g;
  g.open("test.bin", std::fstream::binary);
  iarchive b(g);
  b >> v2;
  g.close();

  BOOST_CHECK_EQUAL(v2.size(), 2);
  BOOST_CHECK_EQUAL(v2[0], x);
  BOOST_CHECK_EQUAL(v2[1], y);
}

BOOST_AUTO_TEST_CASE(map) {
  std::map<std::string,int> v;
  v["one"] = 1;
  v["two"] = 2;
  v["three"] = 3;

  std::ofstream f;
  f.open("test.bin", std::fstream::binary);
  oarchive a(f);
  a << v;
  f.close();

  //deserialize into vt2
  std::map<std::string,int> v2;
  std::ifstream g;
  g.open("test.bin", std::fstream::binary);
  iarchive b(g);
  b >> v2;
  g.close();

  BOOST_CHECK_EQUAL(v2.size(), 3);
  BOOST_CHECK_EQUAL(v2["one"], 1);
  BOOST_CHECK_EQUAL(v2["two"], 2);
  BOOST_CHECK_EQUAL(v2["three"], 3);
}
