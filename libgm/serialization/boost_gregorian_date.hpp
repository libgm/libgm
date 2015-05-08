#ifndef LIBGM_SERIALIZE_BOOST_GREGORIAN_DATE_HPP
#define LIBGM_SERIALIZE_BOOST_GREGORIAN_DATE_HPP

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <boost/date_time/gregorian/gregorian.hpp>

namespace libgm {

  inline oarchive& operator<<(oarchive& ar, const boost::gregorian::date& date) {
    if (date.is_special()) {
      ar.serialize_char(0);
      ar.serialize_char(date.as_special());
    } else {
      ar.serialize_char(date.year() & 0x7f);
      ar.serialize_char(date.year() >> 7);
      ar.serialize_char(date.month());
      ar.serialize_char(date.day());
    }
    return ar;
  }

  inline iarchive& operator>>(iarchive& a, boost::gregorian::date& date) {
    char y1, y2, m, d;
    ar >> y1;
    if (y1 == 0) { // special date
      ar >> y2;
      date = boost::gregorian::date(boost::gregorian::special_values(y2));
    } else {
      ar >> y2 >> m >> d;
      date = boost::gregorian::date(int(y1) | (int(y2) << 7), m, d);
    }
    return ar;
  }

}

#endif
