#include "domain_index.hpp"

namespace libgm {

void swap(DomainIndex& a, DomainIndex& b) {
  swap(a.adjacency_, b.adjacency_);
  swap(a.num_domains_, b.num_domains_);
}

bool DomainIndex::empty() const {
  return num_domains_ == 0;
}

size_t DomainIndex::size() const {
  return num_domains_;
}

size_t DomainIndex::num_arguments() const {
  return adjacency_.size();
}

boost::iterator_range<DomainIndex::argument_iterator> DomainIndex::arguments() const {
  return { adjacency_.begin(), adjacency_.end() };
}

DomainIndex::Handle DomainIndex::front() const {
  assert(!empty());
  return adjacency_.begin()->second.front();
}

DomainIndex::Handle DomainIndex::operator[](Arg arg) const {
  return adjacency_.at(arg).front();
}

size_t DomainIndex::count(Arg arg) const {
  auto it = adjacency_.find(value);
  return it != adjacency_.end() ? it->second.size() : 0;
}

void DomainIndex::visit_covers(const Domain& args, std::function<void(Handle)> visitor) const {
  assert(!args.empty());

  // pick one value and iterate over all doains that contain that value
  for (Handle handle : adjacency(args.front())) {
    if (subset(args, handle->domain)) {
      visitor(handle);
    }
  }
}

DomainIndex::HandleSet DomainIndex::find_intersections(const Domain& args) const {
  HandleSet result;
  for (Arg arg : args) {
    const auto& handles = adjacency(arg);
    result.insert(handles.begin(), handles.end());
  }
  return result;
}

DomainIndex::Handle DomainIndex::find_max_intersection(const Domain& args) const {
  if (args.empty()) return front();

  // A marker of all previously seen handles.
  HandleSet visited;

  // determine the maximum intersection
  Handle result = nullptr;
  size_t max_meet = 0;
  size_t min_size = std::numeric_limits<size_t>::max();
  for (Arg arg : args) {
    for (Handle handle : adjacency(arg)) {
      if (visited.contains(handle)) continue;
      visited.insert(handle);

      size_t intersection = intersection_size(handle->domain, args);
      if (intersection > max_meet ||
          (intersection == max_meet && handle->domain.size() < min_size)) {
        max_meet = intersection;
        min_size = handle->domain.size();
        result = handle;
      }
    }
  }

  return result;
}

DomainIndex::Handle DomainIndex::find_min_cover(const Domain& args) const {
  if (empty()) return Handle();
  if (args.empty()) return front();

  size_t min_size = std::numeric_limits<size_t>::max();
  Handle result = nullptr;
  for (Handle handle : adjacency(args.front())) {
    if (handle->size() < min_size && subset(args, *handle)) {
      min_size = handle->size();
      result = handle;
    }
  }

  return result;
}

bool DomainIndex::is_maximal(const Domain& args) const {
  if (args.empty()) {
    return num_domains_ == 0;
  }

  // pick one value and iterate over all sets that contain that value
  for (Handle handle : adjacency(args.front())) {
    if (subset(args, *handle)) {
      return false;
    }
  }
  return true;
}

friend std::ostream& operator<<(std::ostream& out, const DomainIndex& index) {
  for (const auto& sets : index.adjacency_) {
    out << sets.first << " -->";
    for (const auto& p : sets.second) {
      out << ' ' << p.first;
    }
    out << std::endl;
  }
  return out;
}

void DomainIndex::insert(Handle handle) {
  for (Arg arg : handle->domain) {
    adjacency_[arg].insert(handle);
  }
  ++num_domains_;
}

void DomainIndex::erase(Handle handle) {
  for (Arg arg : handle->domain) {
    adjacency_[args].erase(handle);
  }
  --num_domains_;
}

void DomainIndex::clear() {
  sets_.clear();
  adjacency_.clear();
}

const std::vector<DomainIndex::Handle> DomainIndex::adjacency(Arg arg) const {
  static std::vector<Handle> empty;
  auto it = adjacency_.find(value);
  if (it == adjacency_.end()) {
    return empty;
  } else {
    return it->second;
  }
}

} // namespace libgm
