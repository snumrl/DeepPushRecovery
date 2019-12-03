cmake_minimum_required(VERSION 2.8.6)
project(render)

link_directories(../core/)
include_directories(../core/)

find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(PythonLibs REQUIRED)
find_package(Boost 1.65.0 COMPONENTS filesystem python3 numpy3)

include_directories(${DART_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
if(APPLE)
    add_definitions(-DGL_SILENCE_DEPRECATION)
endif(APPLE)

file(GLOB srcs "*.h" "*.cpp")
add_executable(render ${srcs})
target_link_libraries(render ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenGL_gl_LIBRARY} ${OpenGL_glu_LIBRARY} ${GLUT_LIBRARIES} mss)
