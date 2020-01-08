TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    main.cpp



INCLUDEPATH += /usr/local/include/opencv4 \


INCLUDEPATH += /usr/include/eigen3/


LIBS += /usr/local/lib/libopencv_*.so \


INCLUDEPATH += /usr/include/openni2

LIBS += -L/usr/lib -lOpenNI2
