cmake_minimum_required(VERSION 3.16)

project(VBLAST_C_Extensions C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ============================================================
# Compiler options
# ============================================================
add_compile_options(
    -O3
    -Wall
    -Wextra
    -fPIC
    -march=native
)

# ============================================================
# Find system MKL (Explicit Sequential Components)
# ============================================================
find_path(
    MKL_INCLUDE_DIR
    NAMES mkl.h mkl_cblas.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/intel/oneapi/mkl/latest/include
        /opt/intel/oneapi/mkl/*/include
)

find_library(MKL_INTERFACE_LIB NAMES mkl_intel_lp64
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
          /opt/intel/oneapi/mkl/latest/lib/intel64 /opt/intel/oneapi/mkl/*/lib/intel64)

find_library(MKL_SEQUENTIAL_LIB NAMES mkl_sequential
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
          /opt/intel/oneapi/mkl/latest/lib/intel64 /opt/intel/oneapi/mkl/*/lib/intel64)

find_library(MKL_CORE_LIB NAMES mkl_core
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
          /opt/intel/oneapi/mkl/latest/lib/intel64 /opt/intel/oneapi/mkl/*/lib/intel64)

if(NOT MKL_INCLUDE_DIR OR NOT MKL_INTERFACE_LIB OR NOT MKL_SEQUENTIAL_LIB OR NOT MKL_CORE_LIB)
    message(FATAL_ERROR "Could not find explicit sequential MKL components (lp64, sequential, core).")
endif()

message(STATUS "MKL include:    ${MKL_INCLUDE_DIR}")
message(STATUS "MKL interface:  ${MKL_INTERFACE_LIB}")
message(STATUS "MKL sequential: ${MKL_SEQUENTIAL_LIB}")
message(STATUS "MKL core:       ${MKL_CORE_LIB}")

# ============================================================
# Math and Threading libraries
# ============================================================
find_library(MATH_LIBRARY NAMES m REQUIRED)

# ============================================================
# Common function for C extensions
# ============================================================
function(add_python_c_extension TARGET_NAME SOURCE_FILE OUTPUT_DIR)
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILE})

    target_include_directories(${TARGET_NAME} PRIVATE ${MKL_INCLUDE_DIR})

    # Hard-link the sequential stack directly + standard linux libs (pthread, dl, m)
    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
        ${MKL_INTERFACE_LIB}
        ${MKL_SEQUENTIAL_LIB}
        ${MKL_CORE_LIB}
        -lpthread
        -ldl
        ${MATH_LIBRARY}
    )

    get_filename_component(MKL_LIB_DIR ${MKL_CORE_LIB} DIRECTORY)

    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_${TARGET_NAME}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${OUTPUT_DIR}
        BUILD_RPATH "${MKL_LIB_DIR}"
        INSTALL_RPATH "${MKL_LIB_DIR}"
    )
endfunction()

# ============================================================
# Exact V-BLAST (Best MKL Variants)
# ============================================================

# Householder
add_python_c_extension(
    householder
    exact_vblast/c/householder.c
    exact_vblast
)

# Chang-Paige
add_python_c_extension(
    chang_paige
    exact_vblast/c/chang_paige.c
    exact_vblast
)

# Computing Inverse
add_python_c_extension(
    computing_inverse
    exact_vblast/c/computing_inverse.c
    exact_vblast
)

# ============================================================
# Approximate V-BLAST (Best MKL Variants)
# ============================================================

# Repeated Random Sampling
add_python_c_extension(
    repeated_random_sampling
    approximate_vblast/c/repeated_random_sampling.c
    approximate_vblast
)

# SQRD
add_python_c_extension(
    sqrd
    approximate_vblast/c/sqrd.c
    approximate_vblast
)

# QR baelines
add_python_c_extension(
    qr_baselines
    performence_comparison_qr/c/qr_baselines.c
    performence_comparison_qr
)


# ils
add_python_c_extension(
    ils_search
    ../schnorr_euchner/c/ils_search.c
    ../schnorr_euchner/
)

# ============================================================
# Summary
# ============================================================
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "VBLAST Native C Extensions (Sequential MKL)")
message(STATUS "==========================================")