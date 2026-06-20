@echo off
set SRC=src
set TMPDIR=tmp
if not exist %TMPDIR% mkdir %TMPDIR%
set TMP=%CD%\%TMPDIR%
set TEMP=%CD%\%TMPDIR%

set COMMON=%SRC%\a-network\convert.cpp %SRC%\a-network\field.cpp %SRC%\a-network\readout.cpp %SRC%\a-network\backward.cpp %SRC%\a-network\model.cpp
set INFRA=%SRC%\tokenizer\bpe.cpp %SRC%\io\data.cpp %SRC%\io\checkpoint.cpp %SRC%\io\progress.cpp %SRC%\train\optimizer.cpp

clang++ -std=c++23 -target x86_64-pc-windows-msvc -fopenmp -I%SRC% %SRC%\app\train.cpp %COMMON% %INFRA% -o train.exe
if %errorlevel% == 0 ( echo Build OK: train.exe ) else ( echo Build FAILED )
