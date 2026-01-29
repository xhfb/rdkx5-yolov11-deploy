@echo off
setlocal enabledelayedexpansion

:: ==========================================
:: RDK X5 YOLOv11 Deploy Tool (Clean Workspace)
:: ==========================================

:: Initialize
set "MODEL="
set "CALIB="
set "CONTAINER_NAME=rdk_converter_v1"
set "IMAGE_NAME=openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8"

:: ------------------------------------------
:: 1. CLI Mode Check
:: ------------------------------------------
if "%~1"=="" goto interactive_mode

:parse_loop
if "%~1"=="" goto check_params
if "%~1"=="--model" (
    set "MODEL=%~2"
    shift
    shift
    goto parse_loop
)
if "%~1"=="--calibrate_images" (
    set "CALIB=%~2"
    shift
    shift
    goto parse_loop
)
shift
goto parse_loop

:: ------------------------------------------
:: 2. Interactive Mode
:: ------------------------------------------
:check_params
if "%MODEL%"=="" goto interactive_mode
if "%CALIB%"=="" goto interactive_mode
goto validate_files

:interactive_mode
cls
echo ========================================================
echo        RDK X5 YOLOv11 Deploy Tool (Windows)
echo ========================================================
echo.
echo [INFO] No parameters detected. Entering Input Mode.
echo.

:ask_model
echo [1/2] Please input .pt MODEL path:
set /p "MODEL=> "
set "MODEL=!MODEL:"=!"
if "!MODEL!"=="" goto ask_model
if not exist "!MODEL!" (
    echo [ERROR] File not found. Try again.
    goto ask_model
)

echo.
:ask_calib
echo [2/2] Please input IMAGES folder path:
set /p "CALIB=> "
set "CALIB=!CALIB:"=!"
if "!CALIB!"=="" goto ask_calib
if not exist "!CALIB!" (
    echo [ERROR] Folder not found. Try again.
    goto ask_calib
)

:: ------------------------------------------
:: 3. File Preparation & Timestamp
:: ------------------------------------------
:validate_files
echo.
echo [Host] Preparing workspace...

for %%F in ("!MODEL!") do set "MODEL_NAME=%%~nxF"
for %%F in ("!CALIB!") do set "CALIB_NAME=%%~nxF"

:: --- 生成时间戳 ---
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set "TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%"

:: --- 创建运行目录 runs/run_xxx ---
set "RUN_DIR_NAME=run_!TIMESTAMP!"
set "RUN_DIR_PATH=%cd%\runs\!RUN_DIR_NAME!"
if not exist "runs" mkdir runs
mkdir "!RUN_DIR_PATH!"

echo [Host] Output directory created: runs\!RUN_DIR_NAME!

:: --- 【修改点】将模型复制到 runs 目录 ---
echo [Host] Copying model to run directory...
copy /y "!MODEL!" "!RUN_DIR_PATH!\!MODEL_NAME!" >nul
:: Docker 路径指向 runs 内部
set "MODEL_PATH_FOR_DOCKER=/data/runs/!RUN_DIR_NAME!/!MODEL_NAME!"

:: --- 【修改点】将图片同步到 runs 目录 ---
echo [Host] Copying images to run directory...
:: 创建一个专门放原图的子文件夹，保持整洁
set "IMAGES_DEST_DIR=!RUN_DIR_PATH!\source_images"
robocopy "!CALIB!" "!IMAGES_DEST_DIR!" /E /NFL /NDL /NJH /NJS >nul
:: Docker 路径指向 runs 内部
set "CALIB_PATH_FOR_DOCKER=/data/runs/!RUN_DIR_NAME!/source_images"

:: ------------------------------------------
:: 4. Docker Execution
:: ------------------------------------------
:check_docker
echo [Host] Checking Docker environment...

set "CONTAINER_STATUS="
for /f "tokens=*" %%i in ('docker ps -f name^=%CONTAINER_NAME% --format "{{.Status}}"') do set CONTAINER_STATUS=%%i

if not "%CONTAINER_STATUS%"=="" (
    echo [Host] Found running container. Reusing...
) else (
    echo [Host] Container not running. Recreating...
    docker rm -f %CONTAINER_NAME% >nul 2>&1
    goto create_container
)
goto run_script

:create_container
docker run -d --name %CONTAINER_NAME% --entrypoint /bin/bash -v "%cd%":/data %IMAGE_NAME% -c "while true; do sleep 3600; done"
if %errorlevel% neq 0 (
    echo [FATAL ERROR] Failed to start Docker container.
    pause
    exit /b
)

:run_script
echo.
echo [Host] Running internal script...
echo --------------------------------------------------------

docker exec %CONTAINER_NAME% /bin/bash /data/scripts/internal_runner.sh "!MODEL_PATH_FOR_DOCKER!" "!CALIB_PATH_FOR_DOCKER!" "!RUN_DIR_NAME!"

echo.
if %errorlevel%==0 (
    echo [Success] All Done!
    echo [Result] Files are located in: runs\!RUN_DIR_NAME!
    start explorer "!RUN_DIR_PATH!"
) else (
    echo [Error] Conversion failed.
    docker logs %CONTAINER_NAME% --tail 20
)

echo.
pause