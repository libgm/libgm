project(LibGM)

cmake_minimum_required(VERSION 2.8.3)

enable_testing()

subdirs(test timing)

if (MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS) # disable checked_iterator warnings
  add_definitions(-D_CRT_SECURE_NO_WARNINGS) # disable fopen warnings
  add_definitions(-D_SECURE_SCL=0)  # necesary for release builds of MSVC
endif(MSVC)

if(MSVC)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  add_definitions(-DBOOST_ALL_DYN_LINK)
#  add_definitions(-DBOOST_ALL_NO_LIB=1)
#  add_definitions(-DBOOST_TEST_DYN_LINK=1)
else(MSVC)
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
endif(MSVC)

find_package(Boost 1.40 COMPONENTS
             program_options
             filesystem
             unit_test_framework
	     system REQUIRED)

# Hacky Eigen3 configuration until there is a find module
include_directories(
  /usr/local/include/eigen3
  /usr/include/eigen3
  $ENV{EIGEN3_INCLUDE_DIR}
)

include_directories(${LibGM_SOURCE_DIR} ${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})
link_libraries(${Boost_LIBRARIES})

# Flags for the debug and release mode
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  link_libraries(pthread)
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()
