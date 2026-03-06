subdirs(eigen
#        likelihood
#        random
)

add_executable(exp exp_test.cpp)

add_test(exp exp)
