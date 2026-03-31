#define BOOST_TEST_MODULE domain
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/named_argument.hpp>
#include <boost/container_hash/hash.hpp>
#include <sstream>
#include <unordered_set>

using namespace libgm;

namespace {

using Arg = NamedArg<16>;

libgm::Domain<Arg> sorted(libgm::Domain<Arg> d) {
  d.sort();
  return d;
}

bool equivalent(libgm::Domain<Arg> a, libgm::Domain<Arg> b) {
  a.sort();
  b.sort();
  return a == b;
}

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors) {
  Arg x("x");
  Arg y("y");

  libgm::Domain<Arg> a;
  BOOST_CHECK(a.empty());

  libgm::Domain<Arg> b({x, y});
  BOOST_CHECK_EQUAL(b.size(), 2);
  BOOST_CHECK_EQUAL(b[0], x);
  BOOST_CHECK_EQUAL(b[1], y);

  libgm::Domain<Arg> c(&x, &x + 1);
  BOOST_CHECK_EQUAL(c.size(), 1);
  BOOST_CHECK_EQUAL(c[0], x);
}

BOOST_AUTO_TEST_CASE(test_operations) {
  Arg x("x");
  Arg y("y");
  Arg z("z");
  Arg w("w");

  libgm::Domain<Arg> xyz  = {x, y, z};
  libgm::Domain<Arg> x1   = {x};
  libgm::Domain<Arg> y1   = {y};
  libgm::Domain<Arg> z1   = {z};
  libgm::Domain<Arg> xy   = {x, y};
  libgm::Domain<Arg> xw   = {x, w};
  libgm::Domain<Arg> yx   = {y, x};
  libgm::Domain<Arg> yw   = {y, w};
  libgm::Domain<Arg> yz   = {y, z};
  libgm::Domain<Arg> zw   = {z, w};
  libgm::Domain<Arg> xyw  = {x, y, w};
  libgm::Domain<Arg> yzw  = {y, z, w};
  libgm::Domain<Arg> xyzw = {x, y, z, w};
  libgm::Domain<Arg> xwzy = {x, w, z, y};
  libgm::Domain<Arg> xywx = {x, y, w, x};

  libgm::Domain<Arg> x1y1 = x1;
  x1y1.append(y1);
  BOOST_CHECK_EQUAL(x1y1, xy);
  libgm::Domain<Arg> xyz_concat = xy;
  xyz_concat.append(z1);
  BOOST_CHECK_EQUAL(xyz_concat, xyz);
  BOOST_CHECK(xyzw.has_prefix(xy));
  BOOST_CHECK(!xyzw.has_prefix(xw));
  BOOST_CHECK(xyzw.has_suffix(zw));
  BOOST_CHECK(!xyzw.has_suffix(xw));

  BOOST_CHECK_EQUAL(sorted(xy) | sorted(z1), sorted(xyz));
  BOOST_CHECK_EQUAL(sorted(xy) | sorted(yw), sorted(xyw));
  BOOST_CHECK_EQUAL(sorted(xy) - sorted(z1), sorted(xy));
  BOOST_CHECK_EQUAL(sorted(xy) - sorted(yz), sorted(x1));
  BOOST_CHECK_EQUAL(sorted(xy) & sorted(yw), sorted(y1));

  BOOST_CHECK(are_disjoint(sorted(xy), sorted(z1)));
  BOOST_CHECK(!are_disjoint(sorted(xy), sorted(yzw)));
  BOOST_CHECK(equivalent(xy, yx));
  BOOST_CHECK(!equivalent(yw, zw));
  BOOST_CHECK(is_subset(sorted(yx), sorted(xyz)));
  BOOST_CHECK(!is_subset(sorted(yx), sorted(yw)));
  BOOST_CHECK(is_superset(sorted(xyzw), sorted(yx)));
  BOOST_CHECK(!is_superset(sorted(xyw), sorted(xyz)));

  BOOST_CHECK(xyz.contains(x));
  BOOST_CHECK(!xyz.contains(w));
  xywx.unique();
  BOOST_CHECK_EQUAL(xywx.size(), 3);
  BOOST_CHECK(equivalent(xywx, xyw));
}

BOOST_AUTO_TEST_CASE(test_num_univariate) {
  libgm::Domain<Arg> v;
  v.push_back(Arg("a"));
  v.push_back(Arg("b"));
  v.push_back(Arg("c"));
  v.push_back(Arg("d"));
  v.push_back(Arg("e"));
  v.push_back(Arg("f"));
  v.push_back(Arg("g"));
  v.push_back(Arg("h"));
  v.push_back(Arg("i"));
  v.push_back(Arg("j"));

  BOOST_CHECK_EQUAL(v.size(), 10);

  v.clear();
  v.push_back(Arg("k"));
  v.push_back(Arg("l"));
  v.push_back(Arg("m"));
  v.push_back(Arg("n"));
  BOOST_CHECK_EQUAL(v.size(), 4);
}

BOOST_AUTO_TEST_CASE(test_argset_constructor_and_sort_state) {
  Arg x("x");
  Arg y("y");
  Arg z("z");

  libgm::Domain<Arg> d = {z, x, y};
  d.sort();
  BOOST_CHECK(d.is_sorted());
  BOOST_CHECK_EQUAL(d.size(), 3);
  BOOST_CHECK(d.contains(x));
  BOOST_CHECK(d.contains(y));
  BOOST_CHECK(d.contains(z));
}

BOOST_AUTO_TEST_CASE(test_stream_prefix_suffix_and_bounds) {
  Arg x("x");
  Arg y("y");
  Arg z("z");

  libgm::Domain<Arg> xyz = {x, y, z};
  libgm::Domain<Arg> p2 = xyz.prefix(2);
  libgm::Domain<Arg> s2 = xyz.suffix(2);
  BOOST_CHECK_EQUAL(p2, libgm::Domain<Arg>({x, y}));
  BOOST_CHECK_EQUAL(s2, libgm::Domain<Arg>({y, z}));

  BOOST_CHECK_THROW(xyz.prefix(4), std::invalid_argument);
  BOOST_CHECK_THROW(xyz.suffix(4), std::invalid_argument);

  std::ostringstream out;
  out << libgm::Domain<Arg>({x, y});
  BOOST_CHECK_EQUAL(out.str(), "[x, y]");
}

BOOST_AUTO_TEST_CASE(test_erase_and_set_ops_in_place) {
  Arg x("x");
  Arg y("y");
  Arg z("z");
  Arg w("w");

  libgm::Domain<Arg> xyz = sorted(libgm::Domain<Arg>({x, y, z}));
  libgm::Domain<Arg> yw = sorted(libgm::Domain<Arg>({y, w}));
  libgm::Domain<Arg> yz = sorted(libgm::Domain<Arg>({y, z}));
  libgm::Domain<Arg> xw = sorted(libgm::Domain<Arg>({x, w}));

  libgm::Domain<Arg> inter = xyz;
  inter &= yw;
  BOOST_CHECK_EQUAL(inter, libgm::Domain<Arg>({y}));

  libgm::Domain<Arg> diff = xyz;
  diff -= yz;
  BOOST_CHECK_EQUAL(diff, libgm::Domain<Arg>({x}));

  BOOST_CHECK_EQUAL(intersection_size(xyz, yz), 2);
  BOOST_CHECK_EQUAL(intersection_size(xyz, xw), 1);

  libgm::Domain<Arg> e = sorted(libgm::Domain<Arg>({x, y, z}));
  e.erase(y);
  BOOST_CHECK_EQUAL(e, libgm::Domain<Arg>({x, z}));
  BOOST_CHECK_THROW(e.erase(w), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_shape_dims_and_dims_omit) {
  Arg a("a");
  Arg b("b");
  Arg c("c");
  Arg d("d");
  Arg e("e");

  libgm::Domain<Arg> abcd = sorted(libgm::Domain<Arg>({a, b, c, d}));

  ShapeMap<Arg> shape_map = [&](Arg arg) -> size_t {
    if (arg == a) return 2;
    if (arg == b) return 3;
    if (arg == c) return 5;
    if (arg == d) return 7;
    return 0;
  };

  Shape shape = abcd.shape(shape_map);
  BOOST_CHECK_EQUAL(shape.size(), 4);
  BOOST_CHECK_EQUAL(shape[0], 2);
  BOOST_CHECK_EQUAL(shape[1], 3);
  BOOST_CHECK_EQUAL(shape[2], 5);
  BOOST_CHECK_EQUAL(shape[3], 7);

  libgm::Domain<Arg> ac = {abcd[0], abcd[2]};
  Dims dims = abcd.dims(ac);
  BOOST_CHECK(dims.test(0));
  BOOST_CHECK(!dims.test(1));
  BOOST_CHECK(dims.test(2));
  BOOST_CHECK(!dims.test(3));

  Dims omit = abcd.dims_omit(c);
  BOOST_CHECK(omit.test(0));
  BOOST_CHECK(omit.test(1));
  BOOST_CHECK(!omit.test(2));
  BOOST_CHECK(omit.test(3));

  libgm::Domain<Arg> ca = {c, a};
  BOOST_CHECK_THROW(abcd.dims(ca), std::invalid_argument);
  BOOST_CHECK_THROW(abcd.dims(libgm::Domain<Arg>({a, e})), std::invalid_argument);
  BOOST_CHECK_THROW(abcd.dims_omit(e), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_hashing) {
  Arg x("x");
  Arg y("y");
  Arg z("z");

  libgm::Domain<Arg> a = sorted(libgm::Domain<Arg>({x, y, z}));
  libgm::Domain<Arg> b = sorted(libgm::Domain<Arg>({x, y, z}));
  libgm::Domain<Arg> c = sorted(libgm::Domain<Arg>({x, z}));

  BOOST_CHECK_EQUAL(boost::hash<libgm::Domain<Arg>>()(a), boost::hash<libgm::Domain<Arg>>()(b));
  BOOST_CHECK_EQUAL(std::hash<libgm::Domain<Arg>>()(a), std::hash<libgm::Domain<Arg>>()(b));
  BOOST_CHECK(a == b);
  BOOST_CHECK(a != c);

  std::unordered_set<libgm::Domain<Arg>> set;
  set.insert(a);
  BOOST_CHECK(set.contains(b));
  BOOST_CHECK(!set.contains(c));
}
