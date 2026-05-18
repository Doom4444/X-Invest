@echo off

SET ENV_NAME=xinvest_env

echo ================================
echo Initializing Conda...
echo ================================

CALL "D:\Programs\anaconda\Scripts\activate.bat"

echo ================================
echo Checking Environment...
echo ================================

conda env list | findstr %ENV_NAME% >nul

IF %ERRORLEVEL% NEQ 0 (
    echo Environment not found. Creating...
    conda create -n %ENV_NAME% python=3.10 -y

    echo Activating new environment...
    call conda activate %ENV_NAME%
) ELSE (
    echo Environment found. Activating...
    call conda activate %ENV_NAME%
)

echo ================================
echo Installing Requirements...
echo ================================

IF EXIST requirements.txt (
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found!
)

echo ================================
echo Setup Completed
echo ================================

pause