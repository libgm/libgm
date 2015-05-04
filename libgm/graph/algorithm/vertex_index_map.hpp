#ifndef LIBGM_VERTEX_INDEX_MAP_HPP
#define LIBGM_VERTEX_INDEX_MAP_HPP

#include <libgm/global.hpp>

#include <boost/property_map/property_map.hpp>

#include <memory>
#include <unordered_map>

namespace libgm {

  /**
   * A readable property map that maps the indices of a graph to
   * the range [0, ..., n).
   * \see Boost.PropertyMap
   */
  template <typename Graph>
  class vertex_index_map {
  public:
    typedef boost::readable_property_map_tag category;
    typedef typename Graph::vertex_type      key_type;
    typedef size_t                           reference;
    typedef size_t                           value_type;

    //! Constructs the property map for a graph.
    vertex_index_map(const Graph& graph)
      : map_(new std::unordered_map<key_type, size_t>()) {
      map_->reserve(graph.num_vertices());
      size_t i = 0;
      for (typename Graph::vertex_type v : graph.vertices()) {
        (*map_)[v] = i++;
      }
    }

    //! Returns the index for the given vertex.
    value_type operator[](key_type vertex) const {
      return map_->at(vertex);
    }

    //! Returns the index for the given vertex.
    friend value_type get(const vertex_index_map& map, key_type vertex) {
      return map.map_->at(vertex);
    }

  private:
    std::shared_ptr<std::unordered_map<key_type, size_t> > map_;

  }; // class vertex_index_map

} // namespace libgm

#endif
