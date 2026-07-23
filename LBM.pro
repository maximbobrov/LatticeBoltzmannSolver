TEMPLATE = app
TARGET = LBM
DESTDIR = $$PWD
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    main_app.cpp \
    lbm_core.cpp \
    scenarios.cpp \
    validation.cpp \
    viz.cpp

HEADERS += \
    lbm.h \
    lattice.h \
    cuda_kernels.h

unix{
LIBS += -lGL -lGLU -lglut -lm
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
}
win32{
# freeglut via vcpkg: set the VCPKG_ROOT environment variable, or keep the
# default location %USERPROFILE%/dev/vcpkg. The env var is only trusted if
# freeglut is actually installed there (VS developer prompts point VCPKG_ROOT
# at the VS-bundled vcpkg, which does not have it).
VCPKG_DIR = $$(VCPKG_ROOT)
!exists($$VCPKG_DIR/installed/x64-windows/lib/freeglut.lib): VCPKG_DIR = $$(USERPROFILE)/dev/vcpkg
VCPKG_INST = $$VCPKG_DIR/installed/x64-windows
INCLUDEPATH += $$VCPKG_INST/include
LIBS += -L$$VCPKG_INST/lib -lfreeglut -lopengl32 -lglu32
win32-msvc*: QMAKE_CXXFLAGS += /openmp

# Optional CUDA core (cuda_kernels.cu). Uses the CUDA_PATH environment
# variable (set by the NVIDIA installer). Guarded by the presence of nvcc so
# the project still builds without CUDA installed (HAVE_CUDA undefined ->
# the gpu*() stubs in main_app.cpp report "not available" and the app runs on
# the CPU only).
CUDA_DIR = $$(CUDA_PATH)
exists($$CUDA_DIR/bin/nvcc.exe) {
    DEFINES += HAVE_CUDA
    INCLUDEPATH += "$$CUDA_DIR/include"
    QMAKE_LIBDIR += "$$CUDA_DIR/lib/x64"
    LIBS += -lcudart_static
    CUDA_SOURCES += cuda_kernels.cu
    cuda.name = CUDA compiler
    cuda.input = CUDA_SOURCES
    cuda.output = ${QMAKE_FILE_BASE}.obj
    cuda.commands = $$shell_quote($$CUDA_DIR/bin/nvcc.exe) -arch=native -m64 -std=c++14 \
        -Xcompiler -MD -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_IN} \
        -I$$shell_quote($$CUDA_DIR/include)
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
}
