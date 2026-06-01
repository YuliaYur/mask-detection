@echo off
setlocal

set PYTHON=python
set SOURCES=src tests

if "%~1"=="" goto help
if /I "%~1"=="install" goto install
if /I "%~1"=="format" goto format
if /I "%~1"=="format-check" goto format_check
if /I "%~1"=="lint" goto lint
if /I "%~1"=="test" goto test
if /I "%~1"=="check" goto check

goto help

:install
%PYTHON% -m pip install -r docker/requirements.txt -r requirements-dev.txt -r requirements-test.txt
goto end

:format
%PYTHON% -m black %SOURCES%
goto end

:format_check
%PYTHON% -m black --check %SOURCES%
goto end

:lint
%PYTHON% -m flake8 %SOURCES%
%PYTHON% -m pylint %SOURCES%
goto end

:test
%PYTHON% -m pytest
goto end

:check
call %~f0 format-check
if errorlevel 1 goto end
call %~f0 lint
if errorlevel 1 goto end
call %~f0 test
goto end

:help
echo Usage: make.bat ^<install^|format^|format-check^|lint^|test^|check^>

:end
endlocal
