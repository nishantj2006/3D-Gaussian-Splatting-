# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


project(sibr_video)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

# The remote Gaussian viewer only needs FFmpegVideoEncoder for optional capture
# export. VideoUtils additionally requires OpenCV contrib's optical-flow and
# edge-filter modules, which are not shipped with Jetson's NVIDIA OpenCV build.
# Excluding that unrelated utility keeps the interactive viewer build portable.
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/VideoUtils.cpp")


## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
	${FFMPEG_LIBRARIES}
	OpenMP::OpenMP_CXX
	sibr_graphics
)



add_definitions( -DSIBR_VIDEO_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${SIBR_FOLDER})


## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
)
