#ifndef LIBGM_UNIVERSE_HPP
#define LIBGM_UNIVERSE_HPP

#include <memory>
#include <type_traits>
#include <vector>

namespace libgm {

  /**
   * A universe stores pointers to dynamically-allocated objects and deletes
   * the stored objects when the universe goes out of scope. This class is
   * used primarily by the argument types, including var and vec.
   */
  class universe {
  public:
    //! The base class of managed objects.
    struct managed {
      virtual ~managed() { }
    };

    //! Default constructor. Creates an empty universe.
    universe() { }

    //! Destructor. Deletes the stored objects.
    ~universe() { }

    /**
     * Acquires an object. For convenience, this class accepts a pointer to
     * any object that is a subclas of managed, and returns the pointer back.
     */
    template <typename Obj>
    typename std::enable_if<std::is_base_of<managed, Obj>::value, Obj*>::type
    acquire(Obj* obj) {
      objs_.emplace_back(obj);
      return obj;
    }

  private:
    //! The acquired objects..
    std::vector<std::unique_ptr<managed> > objs_;

  }; // class universe

} // namespace libgm

#endif
