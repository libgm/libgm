#include <libgm/traits/at_type.hpp>

#include <type_traits>

int main() {
  using std::is_same;
  using libgm::at_type;
  //static_assert(is_same<at_type<1, double>::type, double>::value, "past end");
  static_assert(is_same<at_type<0, double>::type, double>::value, "first");
  static_assert(is_same<at_type<1, char, int>::type, int>::value, "second");
  static_assert(is_same<at_type<2, char, int, float>::type, float>::value, "third");
}
