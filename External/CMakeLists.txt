include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(ExternalProject)


set(YAML_CPP_BUILD_TESTS OFF CACHE INTERNAL "")
set(YAML_CPP_BUILD_TOOLS OFF CACHE INTERNAL "")
add_subdirectory(yaml-cpp)

# Exclude the examples from the build
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")
# Exclude the tests from the build
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")
add_subdirectory(stdgpu)

set(-DUSE_BASIC_LOGGING ON CACHE INTERNAL "")
set(-DBUILD_SOPHUS_TESTS OFF CACHE INTERNAL "")
set(-DBUILD_SOPHUS_EXAMPLES OFF CACHE INTERNAL "")
add_subdirectory(Sophus)