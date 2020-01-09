TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp


INCLUDEPATH += /usr/local/include/opencv4 \
               /root/anaconda3/envs/pytorch-gpu/include/python3.6m \
        -fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security  -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes

INCLUDEPATH += /usr/include/eigen3/


LIBS += /usr/local/lib/libopencv_*.so \
        -L/root/anaconda3/envs/pytorch-gpu/lib/python3.6/config-3.6m-x86_64-linux-gnu -lpython3.6m\
        -lpthread -ldl  -lutil -lm  -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions

DISTFILES += \
    test_module.py

