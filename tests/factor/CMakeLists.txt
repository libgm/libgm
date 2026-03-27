add_executable(probability_table probability_table_test.cpp)
add_executable(logarithmic_table logarithmic_table_test.cpp)
#
add_executable(probability_vector probability_vector_test.cpp)
add_executable(logarithmic_vector logarithmic_vector_test.cpp)

add_executable(probability_matrix probability_matrix_test.cpp)
add_executable(logarithmic_matrix logarithmic_matrix_test.cpp)
#
add_executable(moment_gaussian moment_gaussian_test.cpp)
add_executable(canonical_gaussian canonical_gaussian_test.cpp)
add_executable(mixture mixture_test.cpp)
#
#add_executable(softmax softmax.cpp)
#
add_test(probability_table probability_table)
add_test(logarithmic_table logarithmic_table)
#
add_test(probability_vector probability_vector)
add_test(logarithmic_vector logarithmic_vector)
#
add_test(probability_matrix probability_matrix)
add_test(logarithmic_matrix logarithmic_matrix)
#
add_test(moment_gaussian moment_gaussian)
add_test(canonical_gaussian canonical_gaussian)
add_test(mixture mixture)
#
#add_test(softmax softmax)
