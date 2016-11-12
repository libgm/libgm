add_executable(table_timing table.cpp)
add_executable(vector_timing vector.cpp)
add_executable(matrix_timing matrix.cpp)
add_executable(moment_gaussian_timing moment_gaussian.cpp)
add_executable(canonical_gaussian_timing canonical_gaussian.cpp)

subdirs(experimental)
