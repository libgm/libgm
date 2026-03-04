add_executable(simple_config simple_config.cpp)
add_executable(string_functions string_functions.cpp)

add_test(simple_config simple_config ${CMAKE_CURRENT_LIST_DIR}/test.cfg)
add_test(string_functions string_functions)
