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
        ${PROJECT_SOURCE_DIR}/Tracker
        ${PROJECT_SOURCE_DIR}/Tracker/HungarianAlg
        ${PROJECT_SOURCE_DIR}/Tracker/graph
        ${PROJECT_SOURCE_DIR}/Tracker/graph/GTL/include
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
set(CMAKE_CXX_STANDARD 14)

# ----------------------------------------------------------------------------
# Link source and header files
# ----------------------------------------------------------------------------
set(graph_source
        Tracker/graph/tokenise.cpp
        #             Tracker/graph/script.cpp
        #             Tracker/graph/rings.cpp
        Tracker/graph/mytree.cpp
        Tracker/graph/mygraph.cpp
        Tracker/graph/mwbmatching.cpp
        Tracker/graph/mincut.cpp
        Tracker/graph/gport.cpp
        #             Tracker/graph/gml2nestedsql.cpp
        #             Tracker/graph/gml2dot.cpp
        #             Tracker/graph/components.cpp
        Tracker/graph/fheap.c
        )

set(graph_header
        Tracker/graph/tokenise.h
        Tracker/graph/mytree.h
        Tracker/graph/mygraph.h
        Tracker/graph/mwbmatching.h
        Tracker/graph/mincut.h
        Tracker/graph/gport.h
        Tracker/graph/gdefs.h
        Tracker/graph/fheap.h
        )

set(gtl_source
        Tracker/graph/GTL/src/bellman_ford.cpp
        Tracker/graph/GTL/src/bfs.cpp
        Tracker/graph/GTL/src/biconnectivity.cpp
        Tracker/graph/GTL/src/bid_dijkstra.cpp
        Tracker/graph/GTL/src/components.cpp
        Tracker/graph/GTL/src/debug.cpp
        Tracker/graph/GTL/src/dfs.cpp
        Tracker/graph/GTL/src/dijkstra.cpp
        Tracker/graph/GTL/src/edge.cpp
        Tracker/graph/GTL/src/embedding.cpp
        Tracker/graph/GTL/src/fm_partition.cpp
        Tracker/graph/GTL/src/gml_parser.cpp
        Tracker/graph/GTL/src/gml_scanner.cpp
        Tracker/graph/GTL/src/graph.cpp
        Tracker/graph/GTL/src/maxflow_ff.cpp
        Tracker/graph/GTL/src/maxflow_pp.cpp
        Tracker/graph/GTL/src/maxflow_sap.cpp
        Tracker/graph/GTL/src/min_tree.cpp
        Tracker/graph/GTL/src/node.cpp
        Tracker/graph/GTL/src/planarity.cpp
        Tracker/graph/GTL/src/pq_node.cpp
        Tracker/graph/GTL/src/pq_tree.cpp
        Tracker/graph/GTL/src/ratio_cut_partition.cpp
        Tracker/graph/GTL/src/st_number.cpp
        Tracker/graph/GTL/src/topsort.cpp
        )

set(gtl_header
        Tracker/graph/GTL/include/GTL/algorithm.h
        Tracker/graph/GTL/include/GTL/bellman_ford.h
        Tracker/graph/GTL/include/GTL/bfs.h
        Tracker/graph/GTL/include/GTL/biconnectivity.h
        Tracker/graph/GTL/include/GTL/bid_dijkstra.h
        Tracker/graph/GTL/include/GTL/bin_heap.h
        Tracker/graph/GTL/include/GTL/components.h
        Tracker/graph/GTL/include/GTL/debug.h
        Tracker/graph/GTL/include/GTL/dfs.h
        Tracker/graph/GTL/include/GTL/dijkstra.h
        Tracker/graph/GTL/include/GTL/edge.h
        Tracker/graph/GTL/include/GTL/edge_data.h
        Tracker/graph/GTL/include/GTL/edge_map.h
        Tracker/graph/GTL/include/GTL/embedding.h
        Tracker/graph/GTL/include/GTL/fm_partition.h
        Tracker/graph/GTL/include/GTL/gml_parser.h
        Tracker/graph/GTL/include/GTL/gml_scanner.h
        Tracker/graph/GTL/include/GTL/graph.h
        Tracker/graph/GTL/include/GTL/maxflow_ff.h
        Tracker/graph/GTL/include/GTL/maxflow_pp.h
        Tracker/graph/GTL/include/GTL/maxflow_sap.h
        Tracker/graph/GTL/include/GTL/min_tree.h
        Tracker/graph/GTL/include/GTL/ne_map.h
        Tracker/graph/GTL/include/GTL/node.h
        Tracker/graph/GTL/include/GTL/node_data.h
        Tracker/graph/GTL/include/GTL/node_map.h
        Tracker/graph/GTL/include/GTL/planarity.h
        Tracker/graph/GTL/include/GTL/pq_node.h
        Tracker/graph/GTL/include/GTL/pq_tree.h
        Tracker/graph/GTL/include/GTL/ratio_cut_partition.h
        Tracker/graph/GTL/include/GTL/st_number.h
        Tracker/graph/GTL/include/GTL/symlist.h
        Tracker/graph/GTL/include/GTL/topsort.h
        Tracker/graph/GTL/include/GTL/version.h
        Tracker/graph/GTL/include/GTL/GTL.h
        )

set(folder_headers
        Tracker/Ctracker.h
        Tracker/track.h
        Tracker/HungarianAlg/HungarianAlg.h
        Tracker/LocalTracker.h
        Tracker/Kalman.h
        )
set(folder_source
        Tracker/Ctracker.cpp
        Tracker/track.cpp
        Tracker/HungarianAlg/HungarianAlg.cpp
        Tracker/LocalTracker.cpp
        Tracker/Kalman.cpp
        )

add_executable(${PROJECT_NAME}
        src/main.cpp include/Pipeline.h
        src/ssd_detect.cpp
        include/ssd_detect.h
        include/defines.h
        ${folder_source} ${folder_headers} ${graph_source} ${graph_header} ${gtl_source} ${gtl_header}
)

SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_headers})

SOURCE_GROUP("graph" FILES ${graph_source} ${graph_header})
SOURCE_GROUP("GTL" FILES ${gtl_source} ${gtl_header})

# ----------------------------------------------------------------------------
# Link libraries to project
# ----------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} )