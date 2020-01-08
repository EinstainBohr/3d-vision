TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    main.cpp


INCLUDEPATH += /usr/include/eigen3/


INCLUDEPATH += /usr/include/vtk-6.2/


LIBS += /usr/lib/x86_64-linux-gnu/libvtk*.so

INCLUDEPATH += /usr/include/boost/

LIBS += /usr/lib/x86_64-linux-gnu/libboost_*.so


INCLUDEPATH += /usr/local/include/pcl-1.9/


LIBS += /usr/local/lib/libpcl_*.so


INCLUDEPATH += /usr/include/ni

LIBS += -L/usr/lib -lOpenNI
