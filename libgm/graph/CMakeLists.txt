subdirs(algorithm)

target_sources(
  libgm
  PRIVATE
  bipartite_graph.cpp
  directed_graph.cpp
  undirected_graph.cpp
  vector_graph.cpp
)
