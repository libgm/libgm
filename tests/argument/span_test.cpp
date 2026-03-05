#define BOOST_TEST_MODULE span
#include <boost/test/unit_test.hpp>

#include <libgm/argument/span.hpp>

#include <sstream>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_span_defaults_and_stream) {
  Span s;
  BOOST_CHECK_EQUAL(s.start, 0);
  BOOST_CHECK_EQUAL(s.length, 0);

  std::ostringstream out;
  out << s;
  BOOST_CHECK_EQUAL(out.str(), "(0, 0)");
}

BOOST_AUTO_TEST_CASE(test_spans_sum_and_stream) {
  Spans spans = {
    Span{0, 2},
    Span{3, 5},
    Span{10, 1},
  };

  BOOST_CHECK_EQUAL(spans.sum(), 8);

  std::ostringstream out;
  out << spans;
  BOOST_CHECK_EQUAL(out.str(), "Spans([(0, 2), (3, 5), (10, 1)])");
}

BOOST_AUTO_TEST_CASE(test_empty_spans_sum) {
  Spans spans;
  BOOST_CHECK_EQUAL(spans.sum(), 0);
}
