@echo off
set SRC=src
set TMPDIR=tmp
if not exist %TMPDIR% mkdir %TMPDIR%
set TMP=%CD%\%TMPDIR%
set TEMP=%CD%\%TMPDIR%

set COMMON=%SRC%\framework\a-network\a_network.cpp %SRC%\framework\a-network\convert.cpp %SRC%\framework\a-network\field.cpp %SRC%\framework\a-network\readout.cpp %SRC%\framework\a-network\backward.cpp %SRC%\framework\a-network\model.cpp
set INFRA=%SRC%\framework\tokenizer.cpp %SRC%\framework\data.cpp %SRC%\framework\progress.cpp %SRC%\framework\optimizer.cpp %SRC%\framework\trainer.cpp %SRC%\framework\generator.cpp %SRC%\framework\composite_network.cpp %SRC%\framework\logger.cpp

clang++ -std=c++23 -target x86_64-pc-windows-msvc -O3 -ffast-math -ffp-contract=fast -march=native -funroll-loops -fopenmp -I%SRC% %SRC%\app\train.cpp %COMMON% %INFRA% -o train.exe
if %errorlevel% == 0 ( echo Build OK: train.exe ) else ( echo Build FAILED )
