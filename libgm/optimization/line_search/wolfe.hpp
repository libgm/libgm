#ifndef LIBGM_WOLFE_CONDITIONS_HPP
#define LIBGM_WOLFE_CONDITIONS_HPP

#include <functional>

namespace libgm {

  /**
   * A class can evaluate whether or not Wolfe conditions hold for
   * the given point along the line search, permitting early stopping.
   * The default constructor for the class disables early stopping.
   * Sensible defaults for running the line search depend on the
   * optimization method used, see the static functions.
   *
   * For more information, see http://en.wikipedia.org/wiki/Wolfe_conditions
   */
  template <typename RealType = double>
  class wolfe {
  public:
    typedef RealType real_type;
    typedef line_search_result<RealType> argument_type;
    typedef bool result_type;

    RealType c1;
    RealType c2;
    enum { NONE, WEAK, STRONG } type;

    //! The default constructor; disables early stopping.
    wolfe()
      : type(NONE) { }

    //! The construct with given parameters.
    wolfe(RealType c1, RealType c2, bool strong = true)
      : c1(c1), c2(c2), type(strong ? STRONG : WEAK) { }

    //! The defaults for quasi-Newton methods.
    static wolfe quasi_newton(bool strong = true) {
      return wolfe(1e-4, 0.9, strong);
    }
      
    //! The defaults for conjugate gradient descent.
    static wolfe conjugate_gradient(bool strong = true) {
      return wolfe(1e-4, 0.1, strong);
    }

    //! Sets the condition type based on a string.
    void parse_type(const std::string& str) {
      if (str == "none") {
        type = NONE;
      } else if (str == "weak") {
        type = WEAK;
      } else if (str == "strong") {
        type = STRONG;
      } else {
        throw std::invalid_argument("Invalid type of Wolfe conditions");
      }
    }

    //! Returns true if the parameters are valid.
    bool valid() const {
      return (type == NONE) || (0.0 < c1 && c1 <= c2 && c2 <= 1.0);
    }

    //! Returns true if the conditions are empty.
    bool empty() const {
      return type == NONE;
    }

    //! Evaluates the Wolfe conditions for value/slope at step 0 and t.
    bool operator()(const argument_type& start,
                    const argument_type& test) const {
      switch (type) {
      case NONE:
        return false;
      case WEAK:
        return test.value <= start.value + c1 * test.step * start.slope
          && test.slope >= c2 * start.slope;
      case STRONG:
        return test.value <= start.value + c1 * test.step * start.slope
          && std::abs(test.slope) <= c2 * std::abs(start.slope);
      default:
        throw std::logic_error("Invalid type of Wolfe conditions");
      }
    }

    //! Prints the Wolf condition parameters to an output stream
    friend std::ostream& operator<<(std::ostream& out, const wolfe& w) {
      switch (w.type) {
      case NONE:
        out << "(none)"; break;
      case WEAK:
        out << w.c1 << " " << w.c2 << " strong"; break;
      case STRONG:
        out << w.c1 << " " << w.c2 << " weak"; break;
      default:
        throw std::logic_error("Invalid type of Wolfe conditions");
      }
      return out;
    }
    
  }; // class wolfe

} // namespace libgm

#endif
