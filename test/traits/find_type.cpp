#include <libgm/traits/find_type.hpp>

#include <string>

int main() {
  using libgm::find_same;
  //static_assert(find_same<double, char, int, float>::value != 0, "missing");
  static_assert(find_same<double, double>::value == 0, "index 0");
  static_assert(find_same<double, char, double>::value == 1, "index 1");
  static_assert(find_same<double, char, int, double>::value == 2, "index 2");
  static_assert(find_same<double, double, int, double>::value == 0, "duplicate");

  using libgm::find_convertible;
  using std::string;
  //static_assert(find_convertible<double, string, void*>::value != 0, "mising");
  static_assert(find_convertible<double, float, void*>::value == 0, "index 0");
  static_assert(find_convertible<double, string, float>::value == 1, "index 1");
  static_assert(find_convertible<double, float, double>::value == 0, "duplicate");
}
