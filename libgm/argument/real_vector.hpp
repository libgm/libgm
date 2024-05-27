namespace libgm {

template <typename T>
class RealVector : public Object {
public:
  RealVector(DenseVector<T> vec);

private:
  struct Impl;
};

}
