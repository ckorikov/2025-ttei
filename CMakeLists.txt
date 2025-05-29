cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

add_test(NAME tests COMMAND tests --gtest_brief=1)

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

# ------------------------------------------- Code analysis

find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
    set(TIDY_SOURCE_FILES
        include/ttie/ttie.h
        tests/test_main.cpp
        example/main.cpp
    )

    foreach(file ${TIDY_SOURCE_FILES})
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
            message(WARNING "Clang-Tidy: File not found - ${file}")
        endif()
    endforeach()

    string(REPLACE ";" " " TIDY_SOURCE_FILES_STRING "${TIDY_SOURCE_FILES}")

    if(WIN32)
        add_custom_target(tidy
            COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Running clang-tidy and saving report in clang-tidy-report.log..."
            COMMAND cmd /C "\"\"${CLANG_TIDY}\" -p ${CMAKE_BINARY_DIR} ${TIDY_SOURCE_FILES_STRING} > clang-tidy-report.log 2>&1\" || exit 0"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running clang-tidy (Windows, errors suppressed)..."
        )
    else()
        add_custom_target(tidy
            COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Running clang-tidy and saving report in clang-tidy-report.log..."
            COMMAND sh -c "\"${CLANG_TIDY}\" -p ${CMAKE_BINARY_DIR} ${TIDY_SOURCE_FILES_STRING} > clang-tidy-report.log 2>&1 || true"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running clang-tidy (Unix, errors suppressed)..."
            VERBATIM
        )
    endif()
endif()
