#!/bin/bash
set -e

SRC=src_gpu
CPU_SRC=src
BUILD=build_gpu
mkdir -p "$BUILD"

nvcc -std=c++20 -O3 \
     -arch=sm_86 \
     -I"$SRC" -I"$CPU_SRC" \
     "$SRC/app/train.cu" \
     "$SRC/a-network/field.cu" \
     "$CPU_SRC/a-network/convert.cpp" \
     "$CPU_SRC/a-network/field.cpp" \
     "$CPU_SRC/a-network/readout.cpp" \
     "$CPU_SRC/a-network/backward.cpp" \
     "$CPU_SRC/a-network/model.cpp" \
     "$CPU_SRC/tokenizer/bpe.cpp" \
     "$CPU_SRC/io/data.cpp" \
     "$CPU_SRC/io/checkpoint.cpp" \
     "$CPU_SRC/io/progress.cpp" \
     "$CPU_SRC/train/optimizer.cpp" \
     -o "$BUILD/train" \
     -Xcompiler -fopenmp

echo "Build OK: $BUILD/train"
