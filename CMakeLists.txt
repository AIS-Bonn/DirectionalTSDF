##########################################
# Top-level CMakeLists.txt for InfiniTAM #
##########################################

cmake_minimum_required(VERSION 3.13)

PROJECT(InfiniTAM)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(CMAKE_CXX_STANDARD 17)

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
ENDIF()

SET(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
IF (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  SET(CFLAGS_WARN -Wno-misleading-indentation ${CFLAGS_WARN})
ENDIF()
SET(CMAKE_CXX_FLAGS "-fPIC -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
  SET(CMAKE_CXX_FLAGS "-O0 -g ${CMAKE_CXX_FLAGS}")
ELSE()
  SET(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
ENDIF()

INCLUDE_DIRECTORIES(./)
INCLUDE_DIRECTORIES(./External)

ADD_SUBDIRECTORY(External)
ADD_SUBDIRECTORY(ORUtils)
ADD_SUBDIRECTORY(FernRelocLib)
ADD_SUBDIRECTORY(MiniSlamGraphLib)
ADD_SUBDIRECTORY(ITMLib)
ADD_SUBDIRECTORY(Apps)
