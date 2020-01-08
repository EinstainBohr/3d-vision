#include <iostream>
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h>
#include <string.h>
#include <sstream>
#include <Python.h>
#include <thread>
#include <time.h>

using namespace cv;
using namespace std;


int main()
{

    Py_Initialize();
    import_array();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\'/root/桌面/work_space/cp-opencv/')");

    PyObject *pModule = PyImport_ImportModule("test_module");
    PyObject *pDict = PyModule_GetDict(pModule);
    PyObject *pFunc = PyDict_GetItemString(pDict, "run");


    VideoCapture cap(0);
    Mat img;

    for(;;){
        cap.read(img);

        auto sz = img.size();
        int x = sz.width;
        int y = sz.height;
        int z = img.channels();
        uchar *CArrays = new uchar[x*y*z];//这一行申请的内存需要释放指针，否则存在内存泄漏的问题
        int iChannels = img.channels();
        int iRows = img.rows;
        int iCols = img.cols * iChannels;

        uchar* p;
        int id = -1;
        for (int i = 0; i < iRows; i++)
        {
            // get the pointer to the ith row
            p = img.ptr<uchar>(i);
            // operates on each pixel
            for (int j = 0; j < iCols; j++)
            {
                CArrays[++id] = p[j];//连续空间
            }
        }

        npy_intp Dims[3] = { y, x, z}; //注意这个维度数据！
        PyObject *PyArray = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, CArrays);
        PyObject *ArgArray = PyTuple_New(1);
        PyTuple_SetItem(ArgArray, 0, PyArray);
        PyObject_CallObject(pFunc, ArgArray);
        delete []CArrays ;
        CArrays =nullptr;
    }
    return 0;
}
