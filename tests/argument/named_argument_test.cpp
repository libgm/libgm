#define BOOST_TEST_MODULE named_argument
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>

#include <memory>
#include <stdexcept>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_make_and_identity) {
  NamedFactory f("named_argument_test_make");

  Arg x1 = f.make("x");
  Arg x2 = f.make("x");
  Arg y = f.make("y");

  BOOST_CHECK(x1);
  BOOST_CHECK(x2);
  BOOST_CHECK(y);
  BOOST_CHECK(x1 == x2);
  BOOST_CHECK(x1 != y);
}

BOOST_AUTO_TEST_CASE(test_empty_name_returns_null) {
  NamedFactory f("named_argument_test_empty");
  Arg a = f.make("");
  BOOST_CHECK(!a);
}

BOOST_AUTO_TEST_CASE(test_registry_lookup_and_uniqueness) {
  const std::string ns = "named_argument_test_registry";
  {
    NamedFactory f(ns);
    BOOST_CHECK(NamedFactory::find(ns) == &f);
    BOOST_CHECK_THROW(NamedFactory duplicate(ns), std::invalid_argument);
  }
  BOOST_CHECK(NamedFactory::find(ns) == nullptr);
}

BOOST_AUTO_TEST_CASE(test_cross_factory_ordering) {
  NamedFactory a("named_argument_test_a");
  NamedFactory b("named_argument_test_b");

  Arg xa = a.make("x");
  Arg xb = b.make("x");

  BOOST_CHECK(xa != xb);
  BOOST_CHECK(xa < xb);
}

BOOST_AUTO_TEST_CASE(test_register_argument_validation) {
  NamedFactory a("named_argument_test_register_a");
  NamedFactory b("named_argument_test_register_b");

  auto arg = std::make_shared<NamedArgument>("x", a);
  BOOST_CHECK_THROW(b.register_argument(arg), std::invalid_argument);

  a.register_argument(arg);
  BOOST_CHECK_THROW(a.register_argument(arg), std::invalid_argument);
}
