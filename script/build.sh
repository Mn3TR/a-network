#!/bin/bash
SRC=src
TMPDIR=tmp
mkdir -p "$TMPDIR"

COMMON="$SRC/a-network/convert.cpp $SRC/a-network/field.cpp $SRC/a-network/readout.cpp $SRC/a-network/backward.cpp $SRC/a-network/model.cpp"
INFRA="$SRC/tokenizer/bpe.cpp $SRC/io/data.cpp $SRC/io/checkpoint.cpp $SRC/io/progress.cpp $SRC/train/optimizer.cpp"

g++ -std=c++20 -O3 -ffast-math -march=native -funroll-loops -fopenmp \
    -I"$SRC" "$SRC/app/train.cpp" $COMMON $INFRA -o train && \
    echo "Build OK: train"
