cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Toy train-inference engine
project(ttie VERSION 1.0 LANGUAGES CXX)

# ------------------------------------------- Library

add_library(ttie INTERFACE)
target_compile_features(ttie INTERFACE cxx_std_17)
target_include_directories(ttie INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ------------------------------------------- Example

add_executable(example example/main.cpp)
target_link_libraries(example PRIVATE ttie)

# ------------------------------------------- Tests

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tests tests/test_main.cpp)
target_link_libraries(tests PRIVATE gtest gtest_main ttie)

add_test(NAME tests COMMAND tests)

# ------------------------------------------- Formatting

find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    set(FORMAT_SOURCE_FILES
        include/ttie/ttie.h
        tests/test_main.cpp
        example/main.cpp
    )
    
    foreach(file ${FORMAT_SOURCE_FILES})
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
            message(WARNING "Formatting: File not found - ${file}")
        endif()
    endforeach()
    
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i --style=file ${FORMAT_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Applying clang-format..."
        VERBATIM
    )
    
    add_custom_target(check-format
        COMMAND ${CLANG_FORMAT} --dry-run --Werror --style=file ${FORMAT_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Checking code style..."
        VERBATIM
    )
endif()