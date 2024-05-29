namespace libgm {

class Dims {
public:
  static Dims all() {
    return variant(std::in_place_index_t<0>());
  }

  static Dims front(size_t n) {
    return variant(std::in_place_index_t<1>(), n);
  }

  static Dims back(size_t n) {
    return variant(std::in_place_index_t<2>(), n);
  }

  static Dims list(std::vector<unsigned> indices) {
    return variant(std::in_place_index_t<3>(), std::move(indices));
  }

private:
  using variant = std::variant<std::monostate, size_t, size_t, std::vector<unsigned>>;
};

}  // namespace libgm