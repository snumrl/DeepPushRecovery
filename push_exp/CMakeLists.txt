cmake_minimum_required(VERSION 2.8.6)
project(pushexp)
add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Boost 1.65.0 COMPONENTS filesystem python3 numpy3)
find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)
find_package(PythonLibs REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

link_directories(../core/)
include_directories(../core/)

file(GLOB pypushexp_srcs "PushSim.h" "PushSim.cpp" "pypushexp.cpp")
file(GLOB pushrender_srcs "main.cpp" "PushSim.h" "PushSim.cpp" "PushWindow.h" "PushWindow.cpp" "PushWindowDraw.cpp")

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${DART_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})


if(APPLE)
    add_definitions(-DGL_SILENCE_DEPRECATION)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

add_library(pypushexp SHARED ${pypushexp_srcs})

target_link_libraries(pypushexp ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} mss)
set_target_properties(pypushexp PROPERTIES PREFIX "" )

add_executable(push_render ${pushrender_srcs})
target_link_libraries(push_render ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenGL_gl_LIBRARY} ${OpenGL_glu_LIBRARY} ${GLUT_LIBRARIES} mss)
