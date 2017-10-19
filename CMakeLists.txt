cmake_minimum_required(VERSION 3.8)
project(demo/ransac)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
else()
    message( FATAL_ERROR "Boost is missing." )
endif()


set(SOURCE_FILES main.cpp ransac.h ransac.cpp)
add_executable(demo/ransac ${SOURCE_FILES})

target_link_libraries(demo/ransac LINK_PUBLIC ${Boost_LIBRARIES} boost_program_options)