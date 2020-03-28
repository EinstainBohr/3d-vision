#include <iostream>
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h>
#include <string.h>
#include <sstream>
#include <Python.h>
#include <thread>
#include <cstring>
#include <time.h>
#include <OpenNI.h>


using namespace std;
using namespace openni;
using namespace cv;

viz::Viz3d window("window");


double fx = 365.23, fy = 365.23, cx = 249.452, cy = 210.152;


int main(){


    Py_Initialize();

    if (!Py_IsInitialized()) {
        printf("初始化失败！");
        return 0;
    }
    import_array();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\'/root/桌面/work_space/pose_3d')");
    PyRun_SimpleString("print(sys.path)");

    PyObject *pModule = PyImport_ImportModule("pose_handle");
    if (pModule == NULL) {
        cout << "没找到" << endl;
    }
    PyObject *pDict = PyModule_GetDict(pModule);
    PyObject *pFunc = PyDict_GetItemString(pDict, "run_image");


    // 1. Initial OpenNI
    OpenNI::initialize();
    // 2. Open Device
    Device mDevice;
    mDevice.open( ANY_DEVICE );
    openni::Status status;

    // 3. Create depth stream
    VideoStream mDepthStream;
    mDepthStream.create( mDevice, SENSOR_DEPTH );
    VideoMode mMode1;
    mMode1.setResolution( 640, 480 );
    mMode1.setFps( 30 );
    mMode1.setPixelFormat( PIXEL_FORMAT_DEPTH_1_MM );
    mDepthStream.setVideoMode( mMode1);


    // 4. Create color stream
    VideoStream mColorStream;
    mColorStream.create( mDevice, SENSOR_COLOR );
    VideoMode mMode;
    mMode.setResolution( 640, 480 );
    mMode.setFps( 30 );
    mMode.setPixelFormat( PIXEL_FORMAT_RGB888 );
    mColorStream.setVideoMode( mMode);

     if( mDevice.isImageRegistrationModeSupported( IMAGE_REGISTRATION_DEPTH_TO_COLOR ) )
   {
       status=mDevice.setImageRegistrationMode( IMAGE_REGISTRATION_DEPTH_TO_COLOR );
   }


    // 6. start
    VideoFrameRef  mColorFrame;
    VideoFrameRef  mDepthFrame;
    mDepthStream.start();
    mColorStream.start();
    int iMaxDepth = mDepthStream.getMaxPixelValue();
    cv::Mat cImageBGR;
    cv::Mat mScaledDepth;
    for(;;)
    {
        // 7. check is color stream is available
        mColorStream.readFrame( &mColorFrame );
        mDepthStream.readFrame( &mDepthFrame);

        // 7b. convert data to OpenCV format
        const cv::Mat mImageRGB(
                mColorFrame.getHeight(), mColorFrame.getWidth(),
                CV_8UC3, (void*)mColorFrame.getData() );

        const cv::Mat mImageDepth(
                  mDepthFrame.getHeight(), mDepthFrame.getWidth(),
                  CV_16UC1, (void*)mDepthFrame.getData() );


        mImageDepth.convertTo( mScaledDepth, CV_8U, 255.0 / iMaxDepth );
        cv::cvtColor( mImageRGB, cImageBGR, cv::COLOR_RGB2BGR );




        Mat image(Size(640, 480),CV_8UC3 , Scalar(0, 0, 0));
        for (int j=0; j<cImageBGR.rows; j++) {
          for (int i=0; i<cImageBGR.cols; i++) {
              image.at<cv::Vec3b>(j,i)[0]= cImageBGR.at<cv::Vec3b>(j,i)[0];
              image.at<cv::Vec3b>(j,i)[1]= cImageBGR.at<cv::Vec3b>(j,i)[1];
              image.at<cv::Vec3b>(j,i)[2]= cImageBGR.at<cv::Vec3b>(j,i)[2];
          } // 一行结束
        }


        auto sz = cImageBGR.size();
        int x = sz.width;
        int y = sz.height;
        int z = cImageBGR.channels();
        uchar *CArrays = new uchar[x*y*z];//这一行申请的内存需要释放指针，否则存在内存泄漏的问题
        int iChannels = cImageBGR.channels();
        int iRows = cImageBGR.rows;
        int iCols = cImageBGR.cols * iChannels;

        uchar* p;
        int id = -1;
        for (int i = 0; i < iRows; i++)
        {
            // get the pointer to the ith row
            p = cImageBGR.ptr<uchar>(i);
            // operates on each pixel
            for (int j = 0; j < iCols; j++)
            {
                CArrays[++id] = p[j];//连续空间
            }
        }



        char *point ,*test_string;
        string data, data2;
        npy_intp Dims[3] = { y, x, z}; //注意这个维度数据！
        PyObject *PyArray = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, CArrays);
        PyObject *ArgArray = PyTuple_New(1);
        PyTuple_SetItem(ArgArray, 0, PyArray);
        PyObject *string = PyObject_CallObject(pFunc, ArgArray);
        PyArg_ParseTuple(string, "s|s", &point,&test_string);


        vector<int> boxes, lines;
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
        //int j = 0;
        for(int i = 0;i < strlen(test_string)+1 ;i++)
        {
            //cout<< point[i] ;
            data2 = data2 + test_string[i];
            if(test_string[i] == ' '){
                int a = atoi(data2.c_str());
                lines.push_back(a);
                data2.clear();
            }
        }


        viz::WSphere Sphere_widget(Point3f(rand()%500,rand()%500,rand()%500),
        60,10,viz::Color::blue());
        window.showWidget("Cube Widget_boll", Sphere_widget);



        int flags = 0;
        for(int i = 1;i < int(lines.size())/4 + 1;i++){
            int x = lines[4*i-4];
            int y = lines[4*i-3];
            int x1 = lines[4*i-2];
            int y1 = lines[4*i-1];
            int mid_x = mImageDepth.at<unsigned short>(y,x)*(x - cx) / fx;
            int mid_y = mImageDepth.at<unsigned short>(y,x)*(y - cy) / fy;
            int mid_z = mImageDepth.at<unsigned short>(y,x);
            int mid_x1 = mImageDepth.at<unsigned short>(y1,x1)*(x1 - cx) / fx;
            int mid_y1 = mImageDepth.at<unsigned short>(y1,x1)*(y1 - cy) / fy;
            int mid_z1 = mImageDepth.at<unsigned short>(y1,x1);
            line(cImageBGR, Point(x,y),Point(x1,y1),Scalar(255,0,0),5, LINE_8, 0);


            if(mid_x == 0|| mid_y == 0|| mid_z ==0 ||
               mid_x1 == 0|| mid_y1 == 0|| mid_z1 ==0)
                continue;



            if(flags == 0){
                viz::WLine axis(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::cherry());
                axis.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget", axis);
            }
            if(flags == 1){
                viz::WLine axis1(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::brown());
                axis1.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget1", axis1);
            }
            if(flags == 2){
                viz::WLine axis2(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::amethyst());
                axis2.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget2", axis2);
            }
            if(flags == 3){
                viz::WLine axis3(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::azure());
                axis3.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget3", axis3);
            }
            if(flags == 4){
                viz::WLine axis4(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::apricot());
                axis4.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget4", axis4);
            }
            if(flags == 5){
                viz::WLine axis5(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::bluberry());
                axis5.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget5", axis5);
            }
            if(flags == 6){
                viz::WLine axis6(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::gold());
                axis6.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget6", axis6);
            }
            if(flags == 7){
                viz::WLine axis7(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::pink());
                axis7.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget7", axis7);
            }
            if(flags == 8){
                viz::WLine axis8(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::blue());
                axis8.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget8", axis8);
            }
            if(flags == 9){
                viz::WLine axis9(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::red());
                axis9.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget9", axis9);
            }
            if(flags == 10){
                viz::WLine axis10(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::green());
                axis10.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget10", axis10);
            }
            if(flags == 11){
                viz::WLine axis11(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::red());
                axis11.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget9", axis11);
            }
            if(flags == 12){
                viz::WLine axis12(Point3f(mid_x,mid_y,mid_z), Point3f(mid_x1,mid_y1,mid_z1),viz::Color::green());
                axis12.setRenderingProperty(viz::LINE_WIDTH, 10.0);
                window.showWidget("Line Widget12", axis12);
            }
            flags++;
        }


        int height=mImageDepth.rows;
        int width = mImageDepth.cols;
        cout <<height << " " <<width <<endl;
        //创建一个储存point cloud的图片
        Mat point_cloud = Mat::zeros(height, width, CV_32FC3);
        for(int row=0; row<mImageDepth.rows;row++)
            for (int col = 0; col < mImageDepth.cols; col++)
            {
                point_cloud.at<Vec3f>(row, col)[2] = mImageDepth.at<unsigned short>(row,col);
                point_cloud.at<Vec3f>(row, col)[0] = mImageDepth.at<unsigned short>(row, col)*(col - cx) / fx;
                point_cloud.at<Vec3f>(row, col)[1] = mImageDepth.at<unsigned short>(row, col)*(row - cy) / fy;
            }
        cv::viz::WCloud cloud(point_cloud,image);
        window.showWidget("cloud",cloud);

        int flag = 0;
        for(int i = 1;i < int(boxes.size())/2 + 1;i++){
            int x = boxes[2*i-2];
            int y = boxes[2*i-1];
            int mid_x = mImageDepth.at<unsigned short>(y,x)*(x - cx) / fx;
            int mid_y = mImageDepth.at<unsigned short>(y,x)*(y - cy) / fy;
            int mid_z = mImageDepth.at<unsigned short>(y,x);
            //cout << mid_x  << " " << mid_y << " " <<mid_z <<endl;

            circle(cImageBGR,Point(x,y),5,Scalar(0, 255, 0),4);



            if(mid_x == 0|| mid_y == 0|| mid_z ==0)
                continue;



            if(flag ==0){
                viz::WSphere Sphere_widget(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::blue());
                window.showWidget("Cube Widget", Sphere_widget);
            }
            if(flag ==1){
                viz::WSphere Sphere_widget1(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::red());
                window.showWidget("Cube Widget1", Sphere_widget1);
            }
            if(flag ==2){
                viz::WSphere Sphere_widget2(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::black());
                window.showWidget("Cube Widget2", Sphere_widget2);
            }
            if(flag ==3){
                viz::WSphere Sphere_widget3(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::green());
                window.showWidget("Cube Widget3", Sphere_widget3);
            }
            if(flag ==4){
                viz::WSphere Sphere_widget4(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::yellow());
                window.showWidget("Cube Widget4", Sphere_widget4);
            }
            if(flag ==5){
                viz::WSphere Sphere_widget5(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::gray());
                window.showWidget("Cube Widget5", Sphere_widget5);
            }
            if(flag ==6){
                viz::WSphere Sphere_widget6(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::white());
                window.showWidget("Cube Widget6", Sphere_widget6);
            }
            if(flag ==7){
                viz::WSphere Sphere_widget7(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::blue());
                window.showWidget("Cube Widget7", Sphere_widget7);
            }
            if(flag ==8){
                viz::WSphere Sphere_widget8(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::red());
                window.showWidget("Cube Widget8", Sphere_widget8);
            }
            if(flag ==9){
                viz::WSphere Sphere_widget9(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::black());
                window.showWidget("Cube Widget9", Sphere_widget9);
            }
            if(flag ==10){
                viz::WSphere Sphere_widget10(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::green());
                window.showWidget("Cube Widget10", Sphere_widget10);
            }
            if(flag ==11){
                viz::WSphere Sphere_widget11(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::yellow());
                window.showWidget("Cube Widget11", Sphere_widget11);
            }
            if(flag ==12){
                viz::WSphere Sphere_widget12(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::gray());
                window.showWidget("Cube Widget12", Sphere_widget12);
            }
            if(flag ==13){
                viz::WSphere Sphere_widget13(Point3f(mid_x,mid_y,mid_z),
                25,10,viz::Color::white());
                window.showWidget("Cube Widget13", Sphere_widget13);
            }
            flag++;
        }




        window.showWidget("Coordinate", viz::WCoordinateSystem());
        window.spinOnce(1, true);
        imshow("cv_show",cImageBGR);
        int key = cv::waitKey(5);
        if(key == 27){
            break;
        }
    }

    // 9. stop
    mDepthStream.destroy();
    mColorStream.destroy();
    mDevice.close();
    OpenNI::shutdown();
    system("pause");
    return 0;
}

