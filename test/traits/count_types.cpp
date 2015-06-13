#include <libgm/traits/count_types.hpp>

#include <string>

int main() {
  using libgm::count_same;
  static_assert(count_same<double>::value == 0, "empty");
  static_assert(count_same<double, char, int, float>::value == 0, "none");
  static_assert(count_same<double, double>::value == 1, "single");
  static_assert(count_same<double, char, double>::value == 1, "single");
  static_assert(count_same<double, double, int, double>::value == 2, "double");

  using libgm::count_convertible;
  using std::string;
  static_assert(count_convertible<double, string, void*>::value == 0, "none");
  static_assert(count_convertible<double, float, void*>::value == 1, "single");
  static_assert(count_convertible<double, string, float>::value == 1, "single");
  static_assert(count_convertible<double, float, double>::value == 2, "double");

  using libgm::all_of_types;
  using std::is_integral;
  static_assert(!all_of_types<is_integral, void*, string>::value, "zero");
  static_assert(!all_of_types<is_integral, double, long>::value, "one");
  static_assert(all_of_types<is_integral, int, long>::value, "two");

  using libgm::any_of_types;
  static_assert(!any_of_types<is_integral, void*, string>::value, "zero");
  static_assert(any_of_types<is_integral, double, long>::value, "one");
  static_assert(any_of_types<is_integral, int, long>::value, "two");

  using libgm::none_of_types;
  static_assert(none_of_types<is_integral, void*, string>::value, "zero");
  static_assert(!none_of_types<is_integral, double, long>::value, "one");
  static_assert(!none_of_types<is_integral, int, long>::value, "two");
}
