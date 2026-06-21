#!/bin/bash
# A-Network CUDA 构建脚本
set -e

SRC=src_gpu
BUILD=build_gpu
mkdir -p "$BUILD"

nvcc -std=c++20 -O3 \
     -arch=sm_86 \
     -I"$SRC" \
     "$SRC/app/train.cu" \
     "$SRC/a-network/field.cu" \
     -o "$BUILD/train"

echo "Build OK: $BUILD/train"
