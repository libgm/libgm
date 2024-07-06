#pragma once

namespace libgm {

template <typename K>
struct ColorMap {
  using value_type = boost::default_color_type;
  using reference = boost::default_color_type;
  using key_type = K;
  using category = boost::read_write_property_map_tag;
};

}