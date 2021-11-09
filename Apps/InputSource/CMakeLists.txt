##################################
# CMakeLists.txt for InputSource #
##################################

###########################
# Specify the target name #
###########################

SET(targetname InputSource)

#####################################
# Offer support for input mirroring #
#####################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/OfferInputMirroring.cmake)

################################
# Specify the libraries to use #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseFFmpeg.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseGLUT.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseKinect2.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseLibRoyale.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenGL.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenMP.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenNI.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseRealSense.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseRealSense2.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseUVC.cmake)

#############################
# Specify the project files #
#############################

SET(sources
CompositeImageSourceEngine.cpp
FFMPEGReader.cpp
FFMPEGWriter.cpp
IdleImageSourceEngine.cpp
ImageSourceEngine.cpp
IMUSourceEngine.cpp
Kinect2Engine.cpp
LibUVCEngine.cpp
OpenNIEngine.cpp
PicoFlexxEngine.cpp
RealSenseEngine.cpp
RealSense2Engine.cpp
TrajectorySourceEngine.cpp
PointCloudSourceEngine.cpp
)

SET(headers
CompositeImageSourceEngine.h
FFMPEGReader.h
FFMPEGWriter.h
IdleImageSourceEngine.h
ImageSourceEngine.h
IMUSourceEngine.h
Kinect2Engine.h
LibUVCEngine.h
OpenNIEngine.h
PicoFlexxEngine.h
RealSenseEngine.h
RealSense2Engine.h
TrajectorySourceEngine.h
PointCloudSourceEngine.h
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${sources} ${headers})

##########################################
# Specify the target and where to put it #
##########################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)
IF(WITH_CUDA)
  CUDA_ADD_LIBRARY(${targetname} STATIC ${sources} ${headers} ${templates})
ELSE()
  ADD_LIBRARY(${targetname} STATIC ${sources} ${headers} ${templates})
ENDIF()
TARGET_LINK_LIBRARIES(${targetname} ITMLib::ITMLib tinyply)

TARGET_INCLUDE_DIRECTORIES(${targetname}
  PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_INSTALL_INCLUDEDIR}>
  )
