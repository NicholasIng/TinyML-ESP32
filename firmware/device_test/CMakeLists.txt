cmake_minimum_required(VERSION 3.16.0)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
# Build only our code and what it needs, not every ESP-IDF component.
set(COMPONENTS src __pio_env)   # __pio_env: needed by PlatformIO
project(device_test)
