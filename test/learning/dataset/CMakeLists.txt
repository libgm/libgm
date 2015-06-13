set(DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)

# Supporting classes
add_executable(text_dataset_format text_dataset_format.cpp)
add_test(text_dataset_format text_dataset_format ${DATA_DIR})

# Static datasets
add_executable(uint_dataset uint_dataset.cpp)
add_executable(real_dataset real_dataset.cpp)
add_executable(hybrid_dataset hybrid_dataset.cpp)
add_test(uint_dataset uint_dataset ${DATA_DIR})
add_test(real_dataset real_dataset ${DATA_DIR})
add_test(hybrid_dataset hybrid_dataset ${DATA_DIR})

# Sequence datasets
add_executable(uint_sequence_dataset uint_sequence_dataset.cpp)
add_executable(real_sequence_dataset real_sequence_dataset.cpp)
# add_executable(hybrid_sequence_dataset hybrid_sequence_dataset.cpp)
add_test(uint_sequence_dataset uint_sequence_dataset ${DATA_DIR})
add_test(real_sequence_dataset real_sequence_dataset ${DATA_DIR})
# add_test(hybrid_sequence_dataset hybrid_sequence_dataset ${DATA_DIR})

# Views
add_executable(slice_view slice_view.cpp)
add_executable(fixed_view fixed_view.cpp)
add_executable(sliding_view sliding_view.cpp)
add_test(slice_view slice_view)
add_test(fixed_view fixed_view)
add_test(sliding_view sliding_view)
