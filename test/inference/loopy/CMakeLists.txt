add_executable(pairwise_mn_bp pairwise_mn_bp.cpp)
add_executable(generalized_bp generalized_bp.cpp)

add_test(pairwise_mn_bp pairwise_mn_bp)
add_test(generalized_bp generalized_bp) # this test is flaky
