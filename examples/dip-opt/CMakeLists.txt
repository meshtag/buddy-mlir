add_custom_command(OUTPUT corr2d.o
COMMAND ${BUDDY_BINARY_DIR}/dip-opt ${BUDDY_EXAMPLES_DIR}/dip-opt/corr2d.mlir -lower-dip -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_CONV_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/dip-opt/corr2d.o
DEPENDS dip-opt)

add_library(Corr2D STATIC corr2d.o)

message(${BUDDY_EXAMPLES_DIR})

SET_TARGET_PROPERTIES(
Corr2D
PROPERTIES
LINKER_LANGUAGE C)

find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(correlation2D correlation2D.cpp)
add_dependencies(correlation2D dip-opt)
target_link_libraries(correlation2D ${OpenCV_LIBS} Corr2D)


include(ExternalProject)

# NOTE for Ninja generator:
# Ninja generator will try to find external project library at the expected location.
# BUILD_BYPRODUCTS option is added to specify build output and satisfy dependencies.
ExternalProject_Add(project_googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG "v1.5.0"
  GIT_SHALLOW 1
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
  TIMEOUT 10
  BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBENCHMARK_ENABLE_TESTING=OFF
  UPDATE_COMMAND ""
  TEST_COMMAND "")

ExternalProject_Get_Property(project_googlebenchmark INSTALL_DIR)

file(MAKE_DIRECTORY ${INSTALL_DIR}/include)
add_library(dip_googlebenchmark STATIC IMPORTED)
target_include_directories(dip_googlebenchmark INTERFACE ${INSTALL_DIR}/include)
set_property(TARGET dip_googlebenchmark PROPERTY IMPORTED_LOCATION
  "${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}")

add_dependencies(dip_googlebenchmark project_googlebenchmark)








find_package(Threads)
target_link_libraries(dip_googlebenchmark INTERFACE Threads::Threads)

add_executable(correlation2DBenchmark correlation2DBenchmark.cpp)
add_dependencies(correlation2DBenchmark dip-opt)
target_link_libraries(correlation2DBenchmark ${OpenCV_LIBS} Corr2D dip_googlebenchmark Conv2D)
