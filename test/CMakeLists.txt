project(test)

if(CYGWIN)
add_definitions(-DBOOST_TEST_DYN_LINK)
endif(CYGWIN)

# Disable unused variable warnings in tests
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-variable")
endif(CMAKE_COMPILER_IS_GNUCXX)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -frepo")

subdirs(
  argument
  datastructure
  factor
  graph
  inference
  learning
  model
  math
  optimization
  parser
  range
  serialization
  )
