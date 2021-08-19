###########################
# CMakeLists.txt for Apps #
###########################

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/External)

add_subdirectory(InfiniTAM)
add_subdirectory(InfiniTAM_cli)

add_executable(hash hash.cpp)
TARGET_LINK_LIBRARIES(hash ITMLib)
