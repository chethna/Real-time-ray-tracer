#-------------------------------------------------
#
# Project created by QtCreator 2015-03-28T16:41:05
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = sample5
TEMPLATE = app

#Whatever libs you want in your program
#CONFIG += console
CONFIG -= app_bundle

SOURCES += main.cpp\
        mainwindow.cpp \
    glwidget.cpp \
    sample1.cpp \
    sample5.cpp \
    sample2.cpp \
    util/mouse.cpp \
    util/utility.c \
    util/DeviceMemoryLogger.cpp \
    util/ImageLoader.cpp \
    util/HDRLoader.cpp \
    util/PPMLoader.cpp \
    util/SampleScene.cpp \
    util/MeshScene.cpp \
    util/OptixMesh.cpp \
    util/OptixMeshImpl.cpp \
    util/MeshBase.cpp \
    util/sutil.c \
    util/ImageDisplay.cpp \
    util/rply-1.01/rply.c \
    thread.cpp \

HEADERS  += mainwindow.h \
    glwidget.h \
    sample1.h \
    sample5.h \
    commonStructs.h \  
    sample2.h \
    src/paint_camera.h \
    src/phong.h \
    src/random.h \
    src/helpers.h \
    util/SampleScene.h \
    util/MeshScene.h \
    util/NsightHelper.h \
    util/mouse.h \
    util/utility.h \
    util/DeviceMemoryLogger.h \
    util/ImageLoader.h \
    util/HDRLoader.h \
    util/PPMLoader.h \
    util/OptixMesh.h \
    util/OptixMeshImpl.h \
    util/MeshBase.h \
    util/OptixMeshClasses.h \
    util/MeshException.h \
    util/sutil.h \
    util/sampleConfig.h \
    util/ImageDisplay.h \
    util/rply-1.01/rply.h \
    thread.h \
    src/diffuse.h

FORMS    += mainwindow.ui

MOC_DIR = moc
OBJECTS_DIR = obj

mac: LIBS += -framework GLUT -framework Python
#else:unix|win32: LIBS += -lGLUT

# Added stuff
INCLUDEPATH +=./util ./src
#/opt/local/include

#Whatever libs you want in your program
#DESTDIR=./

# use this to suppress some warning from boost
QMAKE_CXXFLAGS_WARN_ON += "-Wno-unused-parameter"
QMAKE_CXXFLAGS += -msse -msse2 -msse3
QMAKE_CXXFLAGS_WARN_ON += "-Wno-ignored-qualifiers"
macx:QMAKE_CXXFLAGS += -arch x86_64
#macx:INCLUDEPATH += /usr/local/include/
# define the _DEBUG flag for the graphics lib

#unix:LIBS += -L/usr/local/lib

#Optix Stuff, so any optix program that we wish to turn into PTX code
CUDA_SOURCES += src/normal_shader.cu \
                src/pinhole_camera.cu \
                src/adaptive_pinhole_camera.cu \
                src/constantbg.cu \
                src/envmap.cu \
                src/sphere.cu \
                src/draw_color.cu \
                src/checker.cu \
                src/glass.cu \
                src/parallelogram.cu \
                src/sphere_shell.cu \
                src/phong.cu \
                src/box.cu \
                src/ambocc.cu \
                src/accum_camera.cu \
                src/diffuse.cu \
                src/texturemap.cu \
                src/bodyShader1.cu


#This will change for you, just set it to wherever you have installed cuda
# Path to cuda SDK install
macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5
linux:CUDA_DIR = /usr/local/cuda-7.0
# Path to cuda toolkit install
#macx:CUDA_SDK = /Developer/NVIDIA/CUDA-6.5/samples
#linux:CUDA_SDK = /usr/local/cuda-6.5/samples

# include paths, change this to wherever you have installed OptiX
#macx:INCLUDEPATH += sutil
#macx:INCLUDEPATH += /Developer/OptiX/SDK
#linux:INCLUDEPATH += /usr/local/OptiX/SDK/sutil
#linux:INCLUDEPATH += /usr/local/OptiX/SDK
INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += $$CUDA_SDK/common/inc/
#INCLUDEPATH += $$CUDA_DIR/../shared/inc/
macx:INCLUDEPATH += /Developer/OptiX/include
macx:INCLUDEPATH += /Developer/OptiX/include/optixu
linux:INCLUDEPATH += /usr/local/OptiX/include

# lib dirs
#QMAKE_LIBDIR += $$CUDA_DIR/lib64
macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
#QMAKE_LIBDIR += $$CUDA_SDK/common/lib
macx:QMAKE_LIBDIR += /Developer/OptiX/lib64
linux:QMAKE_LIBDIR += /usr/local/OptiX/lib64

#Add our cuda and optix libraries
LIBS += -lcudart
LIBS += -loptix -loptixu -loptix_prime

# nvcc flags (ptxas option verbose is always useful)
# add the PTX flags to compile optix files
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -ptx

#set our ptx directory so that our ptx files are put somewhere else
PTX_DIR = ptx

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
optix.input = CUDA_SOURCES

#Change our output name to something suitable
optix.output = $$PTX_DIR/${TARGET}_generated_${QMAKE_FILE_BASE}.cu.ptx

# Tweak arch according to your GPU's compute capability
# Either run your device query in cuda/samples or look in section 6 here #http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz3OzHV3KTV
#for optix you can only have one architechture when using the PTX flags when using the -ptx flag you dont want to have the -c flag for compiling
optix.commands = $$CUDA_DIR/bin/nvcc -m64 -gencode arch=compute_30,code=sm_30 $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#use this line for debug code
#optix.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=compute_30,code=sm_30 $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#Declare that we wnat to do this before compiling the C++ code
optix.CONFIG = target_predeps
#now declare that we don't want to link these files with gcc, otherwise it will treat them as object #files
optix.CONFIG += no_link
optix.dependency_type = TYPE_C
# Tell Qt that we want add our optix compiler
QMAKE_EXTRA_UNIX_COMPILERS += optix

DISTFILES += \
    src/diffuse.cu
