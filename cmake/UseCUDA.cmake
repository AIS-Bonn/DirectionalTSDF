#################
# UseCUDA.cmake #
#################

FIND_PACKAGE(CUDA QUIET)

OPTION(WITH_CUDA "Build with CUDA support?" ${CUDA_FOUND})

IF (WITH_CUDA)
  ENABLE_LANGUAGE(CUDA)

  # Auto-detect the CUDA compute capability.
  SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  IF (NOT DEFINED CUDA_COMPUTE_CAPABILITY)
    INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
  ENDIF ()

  SET(ITM_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

  # Set the compute capability flags.
  FOREACH (compute_capability ${CUDA_COMPUTE_CAPABILITY})
    LIST(APPEND ITM_CUDA_NVCC_FLAGS --generate-code arch=compute_${compute_capability},code=sm_${compute_capability})
  ENDFOREACH ()

  # Enable fast math.
  LIST(APPEND ITM_CUDA_NVCC_FLAGS --use_fast_math)

  # Fix max number of registers per kernel. May increase throughput (e.g. GTX 960M) but needs to be tailored to GPU
  LIST(APPEND ITM_CUDA_NVCC_FLAGS --maxrregcount=64)

  # Enable debugging
#    LIST(APPEND ITM_CUDA_NVCC_FLAGS --device-debug -lineinfo)
#    LIST(APPEND ITM_CUDA_NVCC_FLAGS -O0 ; "SHELL:-Xptxas -O0")

  # If on Windows, make it possible to enable GPU debug information.
  IF (MSVC_IDE)
    OPTION(ENABLE_CUDA_DEBUGGING "Enable CUDA debugging?" OFF)
    IF (ENABLE_CUDA_DEBUGGING)
      SET(ITM_CUDA_NVCC_FLAGS -G; ${ITM_CUDA_NVCC_FLAGS})
    ENDIF ()
  ENDIF ()

  # If on Linux:
  IF (${CMAKE_SYSTEM} MATCHES "Linux")
    # Make sure that C++14 support is enabled when compiling with nvcc. From CMake 3.5 onwards,
    # the host flag -std=c++11 is automatically propagated to nvcc. Manually setting it prevents
    # the project from building.
    IF (${CMAKE_VERSION} VERSION_LESS 3.5)
      SET(ITM_CUDA_NVCC_FLAGS -std=c++14; ${ITM_CUDA_NVCC_FLAGS})
    ENDIF ()
  ENDIF ()

  # suppress nvcc warnings.
  IF (NOT MSVC_IDE)
    #  LIST(APPEND ITM_CUDA_NVCC_FLAGS "SHELL:-Xcudafe --display_error_number")
    LIST(APPEND ITM_CUDA_NVCC_FLAGS "SHELL:-Xcudafe --diag_suppress=3111")
    LIST(APPEND ITM_CUDA_NVCC_FLAGS "SHELL:-Xcudafe --diag_suppress=cc_clobber_ignored")
  ENDIF ()
ELSE ()
  ADD_DEFINITIONS(-DCOMPILE_WITHOUT_CUDA)
ENDIF ()