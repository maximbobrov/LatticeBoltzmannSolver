@echo off
cd /d "%~dp0"
rem Override via environment variables if your tools live elsewhere:
if not defined QTDIR set "QTDIR=C:\Qt\5.15.2\msvc2019_64"
if not defined VCPKG_ROOT set "VCPKG_ROOT=%USERPROFILE%\dev\vcpkg"
rem vcvars64 overwrites VCPKG_ROOT with the VS-bundled vcpkg - restore ours.
set "VCPKG_SAVED=%VCPKG_ROOT%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "VCPKG_ROOT=%VCPKG_SAVED%"
"%QTDIR%\bin\qmake.exe"
nmake
if not exist freeglut.dll copy "%VCPKG_ROOT%\installed\x64-windows\bin\freeglut.dll" .
