include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(ExternalProject)
ExternalProject_Add(
	yaml-cpp-project
	SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/"
	INSTALL_DIR "${CMAKE_BINARY_DIR}/usr"
	CMAKE_ARGS -DYAML_CPP_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
add_library(yaml-cpp UNKNOWN IMPORTED)
include_directories(SYSTEM "${PROJECT_BINARY_DIR}/usr/include")
