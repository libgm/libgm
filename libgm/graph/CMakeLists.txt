subdirs(algorithm)

target_sources(
  libgm
  PRIVATE
  bayesian_network.cpp
  cluster_graph.cpp
  factor_graph.cpp
  markov_network.cpp
)
