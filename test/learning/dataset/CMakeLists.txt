set(DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)

# Supporting classes
add_executable(symbolic_format symbolic_format.cpp)
add_test(symbolic_format symbolic_format ${DATA_DIR})

# Static datasets
add_executable(finite_dataset finite_dataset.cpp)
add_executable(vector_dataset vector_dataset.cpp)
add_executable(hybrid_dataset hybrid_dataset.cpp)
add_test(finite_dataset finite_dataset ${DATA_DIR})
add_test(vector_dataset vector_dataset ${DATA_DIR})
add_test(hybrid_dataset hybrid_dataset ${DATA_DIR})

# Sequence datasets
add_executable(finite_sequence_dataset finite_sequence_dataset.cpp)
add_executable(vector_sequence_dataset vector_sequence_dataset.cpp)
# add_executable(hybrid_sequence_dataset hybrid_sequence_dataset.cpp)
add_test(finite_sequence_dataset finite_sequence_dataset ${DATA_DIR})
add_test(vector_sequence_dataset vector_sequence_dataset ${DATA_DIR})
# add_test(hybrid_sequence_dataset hybrid_sequence_dataset ${DATA_DIR})

# Views
add_executable(slice_view slice_view.cpp)
add_executable(fixed_view fixed_view.cpp)
add_executable(sliding_view sliding_view.cpp)
add_test(slice_view slice_view)
add_test(fixed_view fixed_view)
add_test(sliding_view sliding_view)
