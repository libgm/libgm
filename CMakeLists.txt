## Main LibGM project file The cmake documentation is available online
## or in your installation cmake e.g. "C:\Program Files\CMake
## 2.6\doc\cmake-2.6"

project(LibGM)

cmake_minimum_required(VERSION 2.8.3)

set(CMAKE_CXX_STANDARD 11)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

enable_testing()

# Specify the subdirectories for this root directory This will be
# used for the class branch, not the lab branch: subdirs(src examples
# EXCLUDE_FROM_ALL tests timings projects) (Remove EXCLUDE_FROM_ALL
# for lab branch.)
subdirs(test)

# This would make the if statements easier to read
# set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)
# set(CMAKE_VERBOSE_MAKEFILE ON) -- supposedly useful for EclipseCDT
# http://www.vtk.org/Wiki/CMake:Eclipse_UNIX_Tutorial#
# Automatic_Discovery_of_Include_directories_.28Optional.2C_but_handy.29

# add random environment variable definitions
# deal with Boost time shtuff: we want nanoseconds!
# add_definitions(-DBOOST_DATE_TIME_POSIX_TIME_STD_CONFIG)

if (MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS) # disable checked_iterator warnings
  add_definitions(-D_CRT_SECURE_NO_WARNINGS) # disable fopen warnings
  add_definitions(-D_SECURE_SCL=0)  # necesary for release builds of MSVC
endif(MSVC)

#MESSAGE(STATUS "CMAKE_SYSTEM: ${CMAKE_SYSTEM}")

if (${CMAKE_SYSTEM} MATCHES Linux)
  link_directories(/usr/local/lib64)
  set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)
endif (${CMAKE_SYSTEM} MATCHES Linux)

# Remove tracing by default
# add_definitions(-DTRACE_OFF)

#set(BUILD_SHARED_LIBS ON)

if(MSVC)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
#  add_definitions(-DBOOST_ALL_NO_LIB=1)
#  add_definitions(-DBOOST_TEST_DYN_LINK=1)
  add_definitions(-DBOOST_ALL_DYN_LINK)
else(MSVC)
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
endif(MSVC)

find_package(Boost 1.40 COMPONENTS
              program_options
              filesystem
              iostreams  # deprecate together with MPI
              date_time
              thread
              unit_test_framework
	      system REQUIRED)

#find_package(Eigen3 3.1.2 REQUIRED)

# set include path for this and all subdirs
include_directories(${LibGM_SOURCE_DIR} ${Boost_INCLUDE_DIR})

link_directories(${Boost_LIBRARY_DIRS})

link_libraries(${Boost_LIBRARIES})

if(!MINGW)
  link_libraries(bz2)
endif(!MINGW)

if (${CMAKE_SYSTEM} MATCHES Darwin)
#  link_libraries(blas)
  set(CMAKE_EXE_LINKER_FLAGS "-framework Accelerate")
endif (${CMAKE_SYSTEM} MATCHES Darwin)

include_directories(/usr/local/include/eigen3 "$ENV{EIGEN3_INCLUDE_DIR}")

# Flags for the debug and release mode
if(CMAKE_COMPILER_IS_GNUCXX)
  # Note: -fkeep-inline-functions significantly slows down the compilation
  set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb -O0 -std=c++11 -static-libstdc++")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -std=c++11 -static-libstdc++")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "/EHsc /bigobj")
else()
  set(CMAKE_CXX_FLAGS "-g")
endif()

# uncomment the following line to get CMake variables to print to screen
# include(CMakePrintSystemInformation)
