##############################
# CMakeLists.txt for ORUtils #
##############################

SET(targetname ORUtils)

################################
# Specify the libraries to use #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenMP.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UsePNG.cmake)

##########################################
# Specify the target and where to put it #
##########################################

file(GLOB headers include/ORUtils/*.h)
file(GLOB sources src/*.cpp)

SOURCE_GROUP("" FILES ${sources} ${headers})

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)
IF(WITH_CUDA)
  CUDA_ADD_LIBRARY(${targetname})
ELSE()
  ADD_LIBRARY(${targetname})
ENDIF()
ADD_LIBRARY(ITMLib::${targetname} ALIAS ${targetname})
TARGET_SOURCES(${targetname} PRIVATE ${sources})
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkPNG.cmake)

TARGET_INCLUDE_DIRECTORIES(${targetname}
  PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/External/eigen>
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  )

# install the target and create export-set
INSTALL(TARGETS ${targetname}
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