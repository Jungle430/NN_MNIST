cmake_minimum_required(VERSION 3.21)
include_directories(../include)
aux_source_directory(. SRC_SOURCE)
add_library(SRC ${SRC_SOURCE})