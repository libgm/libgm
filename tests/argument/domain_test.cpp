#define BOOST_TEST_MODULE domain
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/named_argument.hpp>
#include <boost/container_hash/hash.hpp>
#include <sstream>
#include <unordered_set>

using namespace libgm;

namespace {

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

Domain sorted(Domain d) {
  d.sort();
  return d;
}

bool equivalent(Domain a, Domain b) {
  a.sort();
  b.sort();
  return a == b;
}

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");

  Domain a;
  BOOST_CHECK(a.empty());

  Domain b({x, y});
  BOOST_CHECK_EQUAL(b.size(), 2);
  BOOST_CHECK_EQUAL(b[0], x);
  BOOST_CHECK_EQUAL(b[1], y);

  Domain c(&x, &x + 1);
  BOOST_CHECK_EQUAL(c.size(), 1);
  BOOST_CHECK_EQUAL(c[0], x);
}

BOOST_AUTO_TEST_CASE(test_operations) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg z = make_arg("z");
  Arg w = make_arg("w");

  Domain xyz  = {x, y, z};
  Domain x1   = {x};
  Domain y1   = {y};
  Domain z1   = {z};
  Domain xy   = {x, y};
  Domain xw   = {x, w};
  Domain yx   = {y, x};
  Domain yw   = {y, w};
  Domain yz   = {y, z};
  Domain zw   = {z, w};
  Domain xyw  = {x, y, w};
  Domain yzw  = {y, z, w};
  Domain xyzw = {x, y, z, w};
  Domain xwzy = {x, w, z, y};
  Domain xywx = {x, y, w, x};

  Domain x1y1 = x1;
  x1y1.append(y1);
  BOOST_CHECK_EQUAL(x1y1, xy);
  Domain xyz_concat = xy;
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
  Domain v;
  v.push_back(make_arg("a"));
  v.push_back(make_arg("b"));
  v.push_back(make_arg("c"));
  v.push_back(make_arg("d"));
  v.push_back(make_arg("e"));
  v.push_back(make_arg("f"));
  v.push_back(make_arg("g"));
  v.push_back(make_arg("h"));
  v.push_back(make_arg("i"));
  v.push_back(make_arg("j"));

  BOOST_CHECK_EQUAL(v.size(), 10);

  v.clear();
  v.push_back(make_arg("k"));
  v.push_back(make_arg("l"));
  v.push_back(make_arg("m"));
  v.push_back(make_arg("n"));
  BOOST_CHECK_EQUAL(v.size(), 4);
}

BOOST_AUTO_TEST_CASE(test_argset_constructor_and_sort_state) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg z = make_arg("z");

  ArgSet set;
  set.insert(z);
  set.insert(x);
  set.insert(y);

  Domain d(set);
  BOOST_CHECK(d.is_sorted());
  BOOST_CHECK_EQUAL(d.size(), 3);
  BOOST_CHECK(d.contains(x));
  BOOST_CHECK(d.contains(y));
  BOOST_CHECK(d.contains(z));
}

BOOST_AUTO_TEST_CASE(test_stream_prefix_suffix_and_bounds) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg z = make_arg("z");

  Domain xyz = {x, y, z};
  Domain p2 = xyz.prefix(2);
  Domain s2 = xyz.suffix(2);
  BOOST_CHECK_EQUAL(p2, Domain({x, y}));
  BOOST_CHECK_EQUAL(s2, Domain({y, z}));

  BOOST_CHECK_THROW(xyz.prefix(4), std::invalid_argument);
  BOOST_CHECK_THROW(xyz.suffix(4), std::invalid_argument);

  std::ostringstream out;
  out << Domain({x, y});
  BOOST_CHECK_EQUAL(out.str(), "[x, y]");
}

BOOST_AUTO_TEST_CASE(test_erase_and_set_ops_in_place) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg z = make_arg("z");
  Arg w = make_arg("w");

  Domain xyz = sorted(Domain({x, y, z}));
  Domain yw = sorted(Domain({y, w}));
  Domain yz = sorted(Domain({y, z}));
  Domain xw = sorted(Domain({x, w}));

  Domain inter = xyz;
  inter &= yw;
  BOOST_CHECK_EQUAL(inter, Domain({y}));

  Domain diff = xyz;
  diff -= yz;
  BOOST_CHECK_EQUAL(diff, Domain({x}));

  BOOST_CHECK_EQUAL(intersection_size(xyz, yz), 2);
  BOOST_CHECK_EQUAL(intersection_size(xyz, xw), 1);

  Domain e = sorted(Domain({x, y, z}));
  e.erase(y);
  BOOST_CHECK_EQUAL(e, Domain({x, z}));
  BOOST_CHECK_THROW(e.erase(w), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_shape_dims_and_dims_omit) {
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  Arg d = make_arg("d");
  Arg e = make_arg("e");

  Domain abcd = sorted(Domain({a, b, c, d}));

  ShapeMap shape_map = [&](Arg arg) -> size_t {
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

  Domain ac = {abcd[0], abcd[2]};
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

  Domain ca = {c, a};
  BOOST_CHECK_THROW(abcd.dims(ca), std::invalid_argument);
  BOOST_CHECK_THROW(abcd.dims(Domain({a, e})), std::invalid_argument);
  BOOST_CHECK_THROW(abcd.dims_omit(e), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_hashing) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg z = make_arg("z");

  Domain a = sorted(Domain({x, y, z}));
  Domain b = sorted(Domain({x, y, z}));
  Domain c = sorted(Domain({x, z}));

  BOOST_CHECK_EQUAL(boost::hash<Domain>()(a), boost::hash<Domain>()(b));
  BOOST_CHECK_EQUAL(std::hash<Domain>()(a), std::hash<Domain>()(b));
  BOOST_CHECK(a == b);
  BOOST_CHECK(a != c);

  std::unordered_set<Domain> set;
  set.insert(a);
  BOOST_CHECK(set.contains(b));
  BOOST_CHECK(!set.contains(c));
}
