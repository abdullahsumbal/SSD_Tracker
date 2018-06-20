cmake_minimum_required(VERSION 3.5)

# ----------------------------------------------------------------------------
# Set Project Name
# ----------------------------------------------------------------------------
set (PROJECT_NAME  SSD)


# ----------------------------------------------------------------------------
# Find Libraries
# ----------------------------------------------------------------------------
FIND_PACKAGE(Caffe REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

# ----------------------------------------------------------------------------
# Add Libraries and peoject files
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${Caffe_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

INCLUDE(CheckIncludeFileCXX)
check_include_file_cxx(opencv2/bgsegm.hpp HAVE_OPENCV_CONTRIB)

INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/include
)
# ----------------------------------------------------------------------------
# Add Definition and Options
# ----------------------------------------------------------------------------
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
option(USE_OPENCV "Build with OpenCV support" ON)
if(HAVE_OPENCV_CONTRIB)
    add_definitions(-DHAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" ON)
    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" ON)
    option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" ON)

else(HAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" OFF)
    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" OFF)
    option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" OFF)
endif(HAVE_OPENCV_CONTRIB)

if(USE_OCV_UKF)
    add_definitions(-DUSE_OCV_UKF)
endif(USE_OCV_UKF)

if(USE_OCV_KCF)
    add_definitions(-DUSE_OCV_KCF)
endif(USE_OCV_KCF)

if(USE_OCV_BGFG)
    add_definitions(-DUSE_OCV_BGFG)
endif(USE_OCV_BGFG)
# ----------------------------------------------------------------------------
# SELECT C++ standard
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)

# ----------------------------------------------------------------------------
# Link source and header files
# ----------------------------------------------------------------------------
add_executable(${PROJECT_NAME} src/main.cpp include/Pipeline.h src/ssd_detect.cpp include/ssd_detect.h)

# ----------------------------------------------------------------------------
# Link libraries to project
# ----------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} )