#include <iostream>
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h>
#include <string.h>
#include <sstream>
#include <Python.h>
#include <thread>
#include <cstring>
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


    VideoCapture cap(CAP_OPENNI2);
    Mat img ,depth;
    viz::Viz3d window("window");
    //显示坐标系
    window.showWidget("Coordinate", viz::WCoordinateSystem());

    for(;;){
        cap.grab();
        cap.retrieve( img, CAP_OPENNI_BGR_IMAGE);
        cap.retrieve( depth, CAP_OPENNI_DEPTH_MAP);

        resize(img,img,Size(img.cols*0.5,img.rows*0.5));

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
        char *point;
        //cahr *point_add = ' ';
        string data;
        npy_intp Dims[3] = { y, x, z}; //注意这个维度数据！
        PyObject *PyArray = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, CArrays);
        PyObject *ArgArray = PyTuple_New(1);
        PyTuple_SetItem(ArgArray, 0, PyArray);
        PyObject *string = PyObject_CallObject(pFunc, ArgArray);
        PyArg_Parse(string, "s", &point);

        vector<int> boxes;
        for(int i = 0;i < strlen(point)+1 ;i++)
        {
            //cout<< point[i] ;
            data = data + point[i];
            if(point[i] == ' '){
                int a = atoi(data.c_str());
                boxes.push_back(a);
                //cout << pointa[j] << " ";
                data.clear();
            }
        }
        //cout <<pointa << endl;


        int height=depth.rows;
        int width = depth.cols;
        //创建一个储存point cloud的图片
        Mat point_cloud = Mat::zeros(height, width, CV_32FC3);
        //point cloud 赋值，其中 fx,fy,cx,cy 为Kinect2 的内参
        double fx = 368.096588, fy = 368.096588, cx = 261.696594, cy = 202.522202;
        for(int row=0; row<depth.rows;row++)
            for (int col = 0; col < depth.cols; col++)
            {
                point_cloud.at<Vec3f>(row, col)[2] = depth.at<unsigned short>(row,col);
                point_cloud.at<Vec3f>(row, col)[0] = depth.at<unsigned short>(row, col)*(col - cx) / fx;
                point_cloud.at<Vec3f>(row, col)[1] = depth.at<unsigned short>(row, col)*(row - cy) / fy;
            }

        for(int i = 1;i < int(boxes.size())/4 + 1;i++){
            //cout << boxes[4*i-4] << " " << boxes[4*i-3] << " "<< boxes[4*i-2] << " "<< boxes[4*i-1] << " " ;
            rectangle(img,Point(boxes[4*i-4],boxes[4*i-3]),Point(boxes[4*i-2],boxes[4*i-1]),
                      Scalar(255, 0, 0),10, LINE_8,0);
            viz::WCube cube_widget(Point3f(depth.at<unsigned short>(boxes[4*i-4],boxes[4*i-3])*(boxes[4*i-3] - cx) / fx,
                                           depth.at<unsigned short>(boxes[4*i-4],boxes[4*i-3])*(boxes[4*i-4] - cy) / fy,
                                           depth.at<unsigned short>(boxes[4*i-4],boxes[4*i-3])),
                                   Point3f(depth.at<unsigned short>(boxes[4*i-2],boxes[4*i-1])*(boxes[4*i-1] - cx) / fx,
                                           depth.at<unsigned short>(boxes[4*i-2],boxes[4*i-1])*(boxes[4*i-2] - cy) / fy,
                                           depth.at<unsigned short>(boxes[4*i-2],boxes[4*i-1])),
                                   true, viz::Color::green());
            cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
            window.showWidget("Cube Widget", cube_widget);
        }
        cv::viz::WCloud cloud(point_cloud);
        window.showWidget("cloud",cloud);

        window.spinOnce(1, true);

        imshow("cv_show",img);
        waitKey(1);


        cout << endl << point <<endl;
        delete []CArrays ;
        CArrays =nullptr;
    }
    return 0;
}
