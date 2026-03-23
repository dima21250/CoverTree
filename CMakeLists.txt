# CMake build system for CoverTree
# Modern, cross-platform build with automatic macOS optimization
cmake_minimum_required(VERSION 3.15)
project(CoverTree VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

#-----------------------------------------------------------------------------
# Platform-specific optimizations
#-----------------------------------------------------------------------------
if(APPLE)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE MACOS_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "macOS Architecture: ${MACOS_ARCH}")

    if(MACOS_ARCH STREQUAL "arm64")
        # Apple Silicon (M1/M2/M3/M4)
        message(STATUS "Configuring for Apple Silicon")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
    else()
        # Intel Mac
        message(STATUS "Configuring for Intel Mac")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()

    # Use libc++ on macOS
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

elseif(UNIX AND NOT APPLE)
    # Linux
    message(STATUS "Configuring for Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

elseif(WIN32)
    # Windows
    message(STATUS "Configuring for Windows")
    # Add Windows-specific flags if needed
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Suppress warnings from bundled Eigen (old version uses deprecated std::result_of)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

#-----------------------------------------------------------------------------
# Options
#-----------------------------------------------------------------------------
option(USE_GPU "Use GPU for FAISS acceleration (affects sample_events.py)" OFF)
option(USE_OPENAI_EMBEDDINGS "Enable OpenAI embedding support in sample_events.py" OFF)

#-----------------------------------------------------------------------------
# Find dependencies
#-----------------------------------------------------------------------------

# Eigen (required)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
else()
    # Use bundled Eigen
    message(STATUS "Using bundled Eigen from lib/")
    set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
endif()

# Threads (required)
find_package(Threads REQUIRED)

#-----------------------------------------------------------------------------
# Include directories
#-----------------------------------------------------------------------------
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/src/commons
)

#-----------------------------------------------------------------------------
# Common source files
#-----------------------------------------------------------------------------
file(GLOB COMMON_SOURCES src/commons/*.cpp)

#-----------------------------------------------------------------------------
# Cover Tree Library (shared)
#-----------------------------------------------------------------------------
add_library(covertree_core SHARED
    src/cover_tree/cover_tree.cpp
    ${COMMON_SOURCES}
)

target_include_directories(covertree_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cover_tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/commons
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(covertree_core
    PUBLIC
        Threads::Threads
)

# Set library properties
set_target_properties(covertree_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME covertree
)

#-----------------------------------------------------------------------------
# Cover Tree executable
#-----------------------------------------------------------------------------
add_executable(cover_tree_exe
    src/cover_tree/main.cpp
)

target_link_libraries(cover_tree_exe
    PRIVATE
        covertree_core
)

set_target_properties(cover_tree_exe PROPERTIES
    OUTPUT_NAME cover_tree
)

#-----------------------------------------------------------------------------
# SG Tree executable
#-----------------------------------------------------------------------------
add_executable(sg_tree_exe
    src/sg_tree/main.cpp
    src/sg_tree/sgtree.cpp
    ${COMMON_SOURCES}
)

target_include_directories(sg_tree_exe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sg_tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/commons
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(sg_tree_exe
    PRIVATE
        Threads::Threads
)

set_target_properties(sg_tree_exe PROPERTIES
    OUTPUT_NAME sg_tree
)

#-----------------------------------------------------------------------------
# Installation
#-----------------------------------------------------------------------------
install(TARGETS covertree_core cover_tree_exe sg_tree_exe
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(FILES
    src/cover_tree/cover_tree.h
    src/cover_tree/utils.h
    DESTINATION include/covertree
)

#-----------------------------------------------------------------------------
# Testing (optional)
#-----------------------------------------------------------------------------
option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
    enable_testing()

    # Add test executable
    add_executable(test_tree
        tests/test_basic.cpp
    )

    target_link_libraries(test_tree
        PRIVATE
            covertree_core
    )

    add_test(NAME BasicTests COMMAND test_tree)
endif()

#-----------------------------------------------------------------------------
# Generate Python config for GPU option
#-----------------------------------------------------------------------------
# Convert CMake booleans to Python booleans
if(USE_GPU)
    set(PYTHON_USE_GPU "True")
else()
    set(PYTHON_USE_GPU "False")
endif()

if(USE_OPENAI_EMBEDDINGS)
    set(PYTHON_USE_OPENAI_EMBEDDINGS "True")
else()
    set(PYTHON_USE_OPENAI_EMBEDDINGS "False")
endif()

set(CONFIG_FILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/covertree_config.py.in")
set(CONFIG_FILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/covertree_config.py")
configure_file(${CONFIG_FILE_IN} ${CONFIG_FILE_OUT} @ONLY)

# Copy the config file to the source directory so that sample_events.py can import it
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/covertree_config.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CONFIG_FILE_OUT}
            ${CMAKE_CURRENT_SOURCE_DIR}/covertree_config.py
    DEPENDS ${CONFIG_FILE_OUT}
    COMMENT "Copying covertree_config.py to source directory"
)

#-----------------------------------------------------------------------------
# Print configuration summary
#-----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "CoverTree Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "C++ Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Eigen include:     ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Use GPU:           ${USE_GPU}")
if(APPLE)
message(STATUS "macOS arch:        ${MACOS_ARCH}")
endif()
message(STATUS "========================================")
message(STATUS "")
