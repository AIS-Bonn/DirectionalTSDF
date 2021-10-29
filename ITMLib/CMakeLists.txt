#############################
# CMakeLists.txt for ITMLib #
#############################

SET(targetname ITMLib)

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenMP.cmake)

file(GLOB_RECURSE sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

# Find headers (required for clion)
file(GLOB_RECURSE public_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
  )
file(GLOB_RECURSE headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

##########################################
# Specify the target and where to put it #
##########################################

SOURCE_GROUP("" FILES ${sources} ${headers})

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)
IF (WITH_CUDA)
  CUDA_ADD_LIBRARY(${targetname})
ELSE ()
  ADD_LIBRARY(${targetname})
ENDIF ()
ADD_LIBRARY(ITMLib::${targetname} ALIAS ${targetname})

TARGET_SOURCES(${targetname} PRIVATE ${sources})
TARGET_LINK_LIBRARIES(${targetname} ITMLib::ORUtils ITMLib::FernRelocLib yaml::yaml stdgpu::stdgpu stdc++fs)

TARGET_COMPILE_OPTIONS(${targetname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  ${ITM_CUDA_NVCC_FLAGS}
  >)

TARGET_INCLUDE_DIRECTORIES(${targetname}
  PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  )

# install the target and create export-set
INSTALL(TARGETS ${targetname} ORUtils FernRelocLib yaml-cpp stdgpu
  EXPORT ${targetname}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

# install header file
INSTALL(FILES ${public_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# generate and install export file
INSTALL(EXPORT ${targetname}Targets
  FILE ${targetname}Targets.cmake
  NAMESPACE ITMLib::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ITMLib
  )


ADD_SUBDIRECTORY(test)