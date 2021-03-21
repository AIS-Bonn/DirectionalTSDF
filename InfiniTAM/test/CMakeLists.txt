set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch2)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

set(TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_block_traversal.cpp
  )
add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests Catch ITMLib)

include(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UsePNG.cmake)

set(targetname stdgpu_benchmark)
cuda_add_executable(${targetname} stdgpu_benchmark.cu)
target_link_libraries(${targetname} ITMLib stdgpu::stdgpu )
target_link_libraries(${targetname} InputSource ITMLib MiniSlamGraphLib ORUtils FernRelocLib stdc++fs)
include(${PROJECT_SOURCE_DIR}/cmake/LinkPNG.cmake)
