add_executable(conjugate_gradient conjugate_gradient.cpp)
add_executable(gradient_descent gradient_descent.cpp)
add_executable(lbfgs lbfgs.cpp)

add_test(conjugate_gradient conjugate_gradient)
add_test(gradient_descent gradient_descent)
add_test(lbfgs lbfgs)
