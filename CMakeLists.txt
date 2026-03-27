cmake_minimum_required(VERSION 3.30.0)

project(
  LibGM
  VERSION 0.1.0
  DESCRIPTION "A C++20 library for probabilistic graphical models"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

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

find_package(
  Boost 1.70 REQUIRED COMPONENTS
  program_options
  unit_test_framework
)

find_package(Eigen3 5.0 REQUIRED)
# add_executable(example example.cpp)
link_libraries(Eigen3::Eigen)
find_package(unordered_dense CONFIG REQUIRED)

include_directories(${LibGM_SOURCE_DIR} ${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})
link_libraries(${Boost_LIBRARIES} unordered_dense::unordered_dense)

subdirs(libgm)
add_library(libgm)

link_libraries(libgm)
subdirs(tests timing)
