#pragma once

#include <libgm/datastructure/subrange.hpp>

#include <boost/stl_interfaces/iterator_interface.hpp>

#include <cereal/cereal.hpp>

#include <cassert>
#include <memory>
#include <vector>

namespace libgm {

template <typename T>
struct IntrusiveList {
  struct Hook;

  struct Entry {
    T* item = nullptr;
    Hook* hook = nullptr;

    friend bool operator==(const Entry& a, const Entry& b) {
      return a.item == b.item && a.hook == b.hook;
    }

    friend bool operator!=(const Entry& a, const Entry& b) {
      return !(a == b);
    }
  };

  struct Hook {
    Entry prev;
    Entry next;

    // Hook(Entry prev, Entry next) : prev(prev), next(next) {}

    operator bool() const {
      return prev.hook || next.hook;
    }

    ~Hook() {
      if (prev.hook && next.hook) {
        next.hook->prev = prev;
        prev.hook->next = next;
      } else {
        assert(!*this);
      }
    }
  };

  struct HookArray : std::unique_ptr<Hook[]> {
    HookArray() = default;
    explicit HookArray(size_t count)
      : std::unique_ptr<Hook[]>(new Hook[count]) {}

    void reset(size_t count) {
      std::unique_ptr<Hook[]>::reset(new Hook[count]);
    }
  };

  class entry_iterator
    : public boost::stl_interfaces::iterator_interface<
        entry_iterator,
        std::bidirectional_iterator_tag,
        const Entry
      > {
  public:
    using base = boost::stl_interfaces::iterator_interface<
      entry_iterator,
      std::bidirectional_iterator_tag,
      const Entry
    >;

    entry_iterator() = default;
    entry_iterator(const Entry& entry) : entry_(entry) {}
    entry_iterator(const Hook& hook) : entry_{nullptr, const_cast<Hook*>(&hook)} {}

    const Entry& operator*() const { return entry_; }
    bool operator==(const entry_iterator& other) const { return entry_.item == other.entry_.item; }
    entry_iterator& operator++() { entry_ = entry_.hook->next; return *this; }
    entry_iterator& operator--() { entry_ = entry_.hook->prev; return *this; }

    using base::operator++;
    using base::operator--;

  private:
    friend boost::stl_interfaces::access;

    Entry entry_;
  };

  class iterator
    : public boost::stl_interfaces::proxy_iterator_interface<
        iterator,
        std::bidirectional_iterator_tag,
        T*
      > {
  public:
    iterator() = default;
    iterator(entry_iterator it) : it_(it) { }

    T* operator*() const {
      return it_->item;
    }

  private:
    friend boost::stl_interfaces::access;

    entry_iterator& base_reference() noexcept { return it_; }
    const entry_iterator& base_reference() const noexcept { return it_; }

    entry_iterator it_;
  };

  IntrusiveList()
    : root_hook_{{nullptr, &root_hook_}, {nullptr, &root_hook_}} {}

  bool empty() const {
    return !root_hook_.next.item;
  }

  T* front() const {
    return root_hook_.next.item;
  }

  T* back() const {
    return root_hook_.prev.item;
  }

  iterator begin() const {
    return entry_iterator(root_hook_.next);
  }

  iterator end() const {
    return entry_iterator(root_hook_);
  }

  SubRange<entry_iterator> entries() const {
    return {entry_iterator(root_hook_.next), entry_iterator(root_hook_)};
  }

  void push_front(T* item, Hook& hook) {
    assert(item);
    assert(!hook && "The specified hook is already taken");
    Hook* next_hook = root_hook_.next.hook;
    hook.prev = {nullptr, &root_hook_};
    hook.next = root_hook_.next;
    next_hook->prev = {item, &hook};
    root_hook_.next = {item, &hook};
  }

  void push_back(T* item, Hook& hook) {
    assert(item);
    assert(!hook && "The specified hook is already taken");
    Hook* prev_hook = root_hook_.prev.hook;
    hook.prev = root_hook_.prev;
    hook.next = {nullptr, &root_hook_};
    root_hook_.prev = {item, &hook};
    prev_hook->next = {item, &hook};
  }

  void erase(T* item, Hook& hook) {
    assert(hook && "The specified hook is not taken");
    Hook* prev_hook = hook->prev.hook;
    Hook* next_hook = hook->next.hook;
    next_hook->prev = hook->prev;
    prev_hook->next = hook->next;
    hook->reset();
  }

private:
  Hook root_hook_;
};

} // namespace libgm
