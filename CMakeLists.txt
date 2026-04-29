cmake_minimum_required(VERSION 3.16)

project(ml_foundations_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(-Wall -Wextra -Werror)

add_library(ml_core
    src/common/vector_ops.cpp
    src/common/csv_loader.cpp
    src/common/data_split.cpp
    src/common/feature_scaling.cpp
    src/common/activation_functions.cpp
    src/common/loss_functions.cpp
    src/common/evaluation_metrics.cpp
    src/common/distance_metrics.cpp
    src/linear_regression/linear_regression.cpp
    src/logistic_regression/logistic_regression.cpp
    src/knn/knn.cpp
    src/knn/knn_utils.cpp
    src/decision_tree/split_utils.cpp
    src/decision_tree/impurity_utils.cpp
)

target_include_directories(ml_core PUBLIC include)

add_executable(ml_app
    app/main.cpp
)

target_link_libraries(ml_app PRIVATE ml_core)