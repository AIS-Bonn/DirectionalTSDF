set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch2)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_block_traversal.cpp
  )
#add_executable(unit_tests ${TEST_SOURCES})
cuda_add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests Catch ITMLib::ITMLib)

include(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UsePNG.cmake)