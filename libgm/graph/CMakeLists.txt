subdirs(algorithm)

target_sources(
  libgm
  PRIVATE
  bipartite_graph.cpp
  directed_graph.cpp
  markov_network.cpp
  undirected_graph.cpp
  special/grid_graph.cpp
)
