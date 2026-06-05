@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM DSA Recommendation Pipeline Runner
REM
REM Usage:
REM   run_dsa_pipeline.bat [input_file]
REM
REM Optional environment overrides:
REM   MODEL=qwen2.5-coder:14b
REM   LIMIT=100
REM   START_INDEX=0
REM   OLLAMA_URL=http://localhost:11434
REM ============================================================

pushd "%~dp0"

set "ROOT=%CD%"
set "PY_EXE=python"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=py -3"
)

if not exist "%ROOT%\data\raw" mkdir "%ROOT%\data\raw"
if not exist "%ROOT%\data\intermediate" mkdir "%ROOT%\data\intermediate"
if not exist "%ROOT%\data\processed" mkdir "%ROOT%\data\processed"
if not exist "%ROOT%\data\failures" mkdir "%ROOT%\data\failures"

set "MODEL=%MODEL%"
if "%MODEL%"=="" set "MODEL=qwen2.5-coder:14b"

set "LIMIT=%LIMIT%"
if "%LIMIT%"=="" set "LIMIT="
set "LIMIT_ARG="
if not "%LIMIT%"=="" set "LIMIT_ARG=--limit %LIMIT%"

set "START_INDEX=%START_INDEX%"
if "%START_INDEX%"=="" set "START_INDEX=0"

set "OLLAMA_URL=%OLLAMA_URL%"
if "%OLLAMA_URL%"=="" set "OLLAMA_URL=http://localhost:11434"

set "RESUME_FLAG=--resume"
set "FORCE_FLAG="
if /I "%FORCE%"=="1" (
    set "RESUME_FLAG="
    set "FORCE_FLAG=--force"
)
if /I "%FORCE%"=="true" (
    set "RESUME_FLAG="
    set "FORCE_FLAG=--force"
)
if /I "%FORCE%"=="yes" (
    set "RESUME_FLAG="
    set "FORCE_FLAG=--force"
)

set "INPUT_FILE=%~1"
if "%INPUT_FILE%"=="" set "INPUT_FILE=%ROOT%\data\raw\problems.json"

set "PASS1_OUT=%ROOT%\data\intermediate\pass1_enriched.jsonl"
set "PASS1_FAIL=%ROOT%\data\failures\pass1_failures.jsonl"
set "TAXONOMY_OUT=%ROOT%\data\processed\taxonomy.json"
set "MAPPING_OUT=%ROOT%\data\processed\label_mapping.json"
set "REPORT1_OUT=%ROOT%\data\processed\taxonomy_report.md"
set "PASS3_OUT=%ROOT%\data\processed\problems_enriched_final.jsonl"
set "PASS3_FAIL=%ROOT%\data\failures\pass3_failures.jsonl"
set "REPORT2_OUT=%ROOT%\data\processed\final_validation_report.md"

echo ============================================================
echo DSA Recommendation Pipeline
echo Root: %ROOT%
echo Input: %INPUT_FILE%
echo Model: %MODEL%
echo Ollama: %OLLAMA_URL%
echo ============================================================

if not exist "%INPUT_FILE%" (
    echo Input file not found: "%INPUT_FILE%"
    echo Pass the dataset path as the first argument, for example:
    echo   run_dsa_pipeline.bat data\raw\problems.json
    goto :fail
)

echo.
echo [1/3] Running pass1 enrichment...
%PY_EXE% scripts\pass1_enrich.py ^
    --input "%INPUT_FILE%" ^
    --output "%PASS1_OUT%" ^
    --failures "%PASS1_FAIL%" ^
    --model "%MODEL%" ^
    --ollama-url "%OLLAMA_URL%" ^
    --start-index %START_INDEX% ^
    %RESUME_FLAG% ^
    %FORCE_FLAG% ^
    %LIMIT_ARG%
if errorlevel 1 goto :fail

echo.
echo [2/3] Building taxonomy...
%PY_EXE% scripts\build_taxonomy.py ^
    --input "%PASS1_OUT%" ^
    --taxonomy-output "%TAXONOMY_OUT%" ^
    --mapping-output "%MAPPING_OUT%" ^
    --report-output "%REPORT1_OUT%"
if errorlevel 1 goto :fail

echo.
echo [3/3] Running constrained final enrichment...
%PY_EXE% scripts\pass3_constrained_enrich.py ^
    --input "%PASS1_OUT%" ^
    --taxonomy "%TAXONOMY_OUT%" ^
    --label-mapping "%MAPPING_OUT%" ^
    --output "%PASS3_OUT%" ^
    --failures "%PASS3_FAIL%" ^
    --report "%REPORT2_OUT%" ^
    --model "%MODEL%" ^
    --ollama-url "%OLLAMA_URL%" ^
    --start-index %START_INDEX% ^
    %RESUME_FLAG% ^
    %FORCE_FLAG% ^
    %LIMIT_ARG%
if errorlevel 1 goto :fail

echo.
echo Pipeline complete.
echo Final output: "%PASS3_OUT%"
echo Report: "%REPORT2_OUT%"
goto :end

:fail
echo.
echo Pipeline failed. Check the logs and failure JSONL files for details.
exit /b 1

:end
popd
endlocal
