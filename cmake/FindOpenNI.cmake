# - Find OpenNI
# This module defines
#  OpenNI_INCLUDE_DIR, where to find OpenNI include files
#  OpenNI_LIBRARIES, the libraries needed to use OpenNI
#  OpenNI_FOUND, If false, do not try to use OpenNI.
# also defined, but not for general use are
#  OpenNI_LIBRARY, where to find the OpenNI library.

set(OPENNI_ROOT "/usr/local" CACHE FILEPATH "Root directory of OpenNI2")

# Finally the library itself
find_library(OpenNI_LIBRARY
NAMES OpenNI
PATHS "${OPENNI_ROOT}/Lib" "C:/Program Files (x86)/OpenNI/Lib" "C:/Program Files/OpenNI/Lib" ${CMAKE_LIB_PATH}
)

find_path(OPENNI_INCLUDE_DIR OpenNI.h PATH "${OPENNI_ROOT}/Include" "${OPENNI_ROOT}/include/openni2")

find_library(OPENNI_LIBRARY OpenNI2 PATH "${OPENNI_ROOT}/Bin/x64-Release/")

# handle the QUIETLY and REQUIRED arguments and set JPEG_FOUND to TRUE if
# all listed variables are TRUE
#include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
#include(${CMAKE_MODULE_PATH}/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(OpenNI DEFAULT_MSG OPENNI_LIBRARY OPENNI_INCLUDE_DIR)

if(OPENNI_FOUND)
  set(OPENNI_LIBRARIES ${OPENNI_LIBRARY})
endif()

mark_as_advanced(OPENNI_LIBRARY OPENNI_INCLUDE_DIR)

