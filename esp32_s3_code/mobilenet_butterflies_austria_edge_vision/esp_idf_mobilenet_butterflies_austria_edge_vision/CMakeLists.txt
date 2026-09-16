cmake_minimum_required(VERSION 3.16.0)

list(APPEND EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../../common_components_butterflies_austria")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(butterflies_austria_mobilenetv2_esp32s3)
