/*
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <thread>

using namespace cv;
using namespace std;


int main(){

    VideoCapture capture(CAP_OPENNI2); // or CAP_OPENNI
    Mat depth;
    Mat Image;
    Mat depth2;

    Eigen::Matrix3f K_ir;           // ir内参矩阵
    K_ir <<
        365.23, 0, 249.452,
        0, 365.23, 210.152,
        0, 0, 1;
    Eigen::Matrix3f K_rgb;          // rgb内参矩阵
    K_rgb <<
        1081.37, 0, 959.5,
        0, 1081.37, 539.5,
        0, 0, 1;

    Eigen::Matrix3f R_ir2rgb;
    Eigen::Matrix3f R;
    Eigen::Vector3f T_temp;
    Eigen::Vector3f T;
    R_ir2rgb <<
        0.9996, 0.0023, -0.0269,
        -0.0018, 0.9998, 0.0162,
        0.0269, -0.0162, 0.9995;
    T_temp <<
        65.9080,
        -4.1045,
        -13.9045;
    R = K_rgb*R_ir2rgb*K_ir.inverse();
    T = K_rgb*T_temp;


    while (1)
    {
        capture.grab();
        capture.retrieve( depth, CAP_OPENNI_DEPTH_MAP);
        capture.retrieve( Image, CAP_OPENNI_BGR_IMAGE);
        capture.retrieve( depth2, CAP_OPENNI_DISPARITY_MAP);


        int height=depth.rows;
        int width = depth.cols;
        //创建一个储存point cloud的图片
        cout << Image.cols <<endl;
        //投影计算部分
        Mat result(480, 640, CV_8UC3);
        int i = 0;
        for (int row = 0; row < 480; row++)
        {
            for (int col = 0; col < 640; col++)
            {
                unsigned short* p = (unsigned short*)depth.data;
                unsigned short depthValue = p[row * 640 + col];
                //cout << "depthValue       " << depthValue << endl;
                if (depthValue != -std::numeric_limits<unsigned short>::infinity()  && depthValue != 0 && depthValue != 65535)
                {
                    // 投影到彩色图上的坐标
                    Eigen::Vector3f uv_depth(col, row, 1.0f);
                    Eigen::Vector3f uv_color = depthValue / 1000.f*R*uv_depth + T / 1000;   //用于计算映射，核心式子

                    int X = static_cast<int>(uv_color[0] / uv_color[2]);         //计算X，即对应的X值
                    int Y = static_cast<int>(uv_color[1] / uv_color[2]);         //计算Y，即对应的Y值

                    if ((X >= 0 && X < 1920) && (Y >= 0 && Y < 1080))
                    {

                        result.data[i * 3] = Image.data[3 * (Y * 1920 + X)];
                        result.data[i * 3 + 1] = Image.data[3 * (Y * 1920 + X) + 1];
                        result.data[i * 3 + 2] = Image.data[3 * (Y * 1920 + X) + 2];
                    }
                }
                else
                {
                    result.data[i * 3] = 0;
                    result.data[i * 3 + 1] = 0;
                    result.data[i * 3 + 2] = 0;
                }
                i++;
            }
        }
        resize(Image,Image,Size(Image.cols*0.5,Image.rows*0.5));
        imshow("rgb",Image);
        imshow("结果图", result);
        imshow("depth2",depth2);
        waitKey(1);
    }

}
*/

/*
#include <stdlib.h>
#include <iostream>
#include <string>
#include "OpenNI.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
using namespace cv;
using namespace openni;

void CheckOpenNIError( Status result, string status )
{
    if( result != STATUS_OK )
        cerr << status << " Error: " << OpenNI::getExtendedError() << endl;
}

int main( int argc, char** argv )
{
    Status result = STATUS_OK;

    //OpenNI2 image
    VideoFrameRef oniDepthImg;
    VideoFrameRef oniColorImg;

    //OpenCV image
    cv::Mat cvDepthImg;
    cv::Mat cvBGRImg;
    cv::Mat cvFusionImg;

    cv::namedWindow("depth");
    cv::namedWindow("image");
    cv::namedWindow("fusion");
    char key=0;

    //【1】
    // initialize OpenNI2
    result = OpenNI::initialize();
    CheckOpenNIError( result, "initialize context" );

    // open device
    Device device;
    result = device.open( openni::ANY_DEVICE );

    //【2】
    // create depth stream
    VideoStream oniDepthStream;
    result = oniDepthStream.create( device, openni::SENSOR_DEPTH );

    //【3】
    // set depth video mode
    VideoMode modeDepth;
    modeDepth.setResolution( 640, 480 );
    modeDepth.setFps( 30 );
    modeDepth.setPixelFormat( PIXEL_FORMAT_DEPTH_1_MM );
    oniDepthStream.setVideoMode(modeDepth);
    // start depth stream
    result = oniDepthStream.start();

    // create color stream
    VideoStream oniColorStream;
    result = oniColorStream.create( device, openni::SENSOR_COLOR );
    // set color video mode
    VideoMode modeColor;
    modeColor.setResolution( 640, 480 );
    modeColor.setFps( 30 );
    modeColor.setPixelFormat( PIXEL_FORMAT_RGB888 );
    oniColorStream.setVideoMode( modeColor);

//【4】
    // set depth and color imge registration mode
    if( device.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR ) )
    {
        device.setImageRegistrationMode( IMAGE_REGISTRATION_DEPTH_TO_COLOR );
    }
    // start color stream
    result = oniColorStream.start();

    while( key!=27 )
    {
        // read frame
        if( oniColorStream.readFrame( &oniColorImg ) == STATUS_OK )
        {
            // convert data into OpenCV type
            cv::Mat cvRGBImg( oniColorImg.getHeight(), oniColorImg.getWidth(), CV_8UC3, (void*)oniColorImg.getData() );
            cv::cvtColor( cvRGBImg, cvBGRImg, COLOR_RGB2BGR );
            cv::imshow( "image", cvBGRImg );
        }

        if( oniDepthStream.readFrame( &oniDepthImg ) == STATUS_OK )
        {
            cv::Mat cvRawImg16U( oniDepthImg.getHeight(), oniDepthImg.getWidth(), CV_16UC1, (void*)oniDepthImg.getData() );
            cvRawImg16U.convertTo( cvDepthImg, CV_8U, 255.0/(oniDepthStream.getMaxPixelValue()));
            //【5】
            // convert depth image GRAY to BGR
            cv::cvtColor(cvDepthImg,cvFusionImg,COLOR_GRAY2BGR);
            cv::imshow( "depth", cvDepthImg );
        }
        //【6】
        cv::addWeighted(cvBGRImg,0.5,cvFusionImg,0.5,0,cvFusionImg);
        cv::imshow( "fusion", cvFusionImg );
        key = cv::waitKey(20);
    }

    //cv destroy
    cv::destroyWindow("depth");
    cv::destroyWindow("image");
    cv::destroyWindow("fusion");

    //OpenNI2 destroy
    oniDepthStream.destroy();
    oniColorStream.destroy();
    device.close();
    OpenNI::shutdown();

    return 0;
}
*/


#include <iostream>
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h>
#include <string.h>
#include <sstream>
#include <Python.h>
#include <thread>
#include <cstring>
#include <time.h>
// OpenNI Header
#include <OpenNI.h>

// namespace
using namespace std;
using namespace openni;
using namespace cv;

int main( int argc, char **argv ){


  // 1. Initial OpenNI
  OpenNI::initialize();
  // 2. Open Device
  Device mDevice;
  mDevice.open( ANY_DEVICE );
  openni::Status status;

  Py_Initialize();
  import_array();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\'/root/桌面/work_space/cp-opencv/')");

  PyObject *pModule = PyImport_ImportModule("test_module");
  PyObject *pDict = PyModule_GetDict(pModule);
  PyObject *pFunc = PyDict_GetItemString(pDict, "run");


  // 3. Create depth stream
  VideoStream mDepthStream;
  mDepthStream.create( mDevice, SENSOR_DEPTH );
  VideoMode mMode1;
  mMode1.setResolution( 640, 480 );
  mMode1.setFps( 30 );
  mMode1.setPixelFormat( PIXEL_FORMAT_DEPTH_1_MM );


  // 4. Create color stream
  VideoStream mColorStream;
  mColorStream.create( mDevice, SENSOR_COLOR );
  VideoMode mMode;
  mMode.setResolution( 640, 480 );
  mMode.setFps( 30 );
  mMode.setPixelFormat( PIXEL_FORMAT_RGB888 );

   if( mDevice.isImageRegistrationModeSupported( IMAGE_REGISTRATION_DEPTH_TO_COLOR ) )
 {
     status=mDevice.setImageRegistrationMode( IMAGE_REGISTRATION_DEPTH_TO_COLOR );
 }
  // 5. create OpenCV Window
  //cv::namedWindow( "Depth Image");
  //cv::namedWindow( "Color Image");
  viz::Viz3d window("window");
  //显示坐标系
  window.showWidget("Coordinate", viz::WCoordinateSystem());

  // 6. start
  VideoFrameRef  mColorFrame;
  VideoFrameRef  mDepthFrame;
  mDepthStream.start();
  mColorStream.start();
  int iMaxDepth = mDepthStream.getMaxPixelValue();
  cv::Mat cImageBGR;
  cv::Mat mScaledDepth;
  while( true )
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

      int height=mImageDepth.rows;
      int width = mImageDepth.cols;
      //创建一个储存point cloud的图片
      Mat point_cloud = Mat::zeros(height, width, CV_32FC3);
      //point cloud 赋值，其中 fx,fy,cx,cy 为Kinect2 的内参
      double fx = 368.096588, fy = 368.096588, cx = 261.696594, cy = 202.522202;
      for(int row=0; row<mImageDepth.rows;row++)
          for (int col = 0; col < mImageDepth.cols; col++)
          {
              point_cloud.at<Vec3f>(row, col)[2] = mImageDepth.at<unsigned short>(row,col);
              point_cloud.at<Vec3f>(row, col)[0] = mImageDepth.at<unsigned short>(row, col)*(col - cx) / fx;
              point_cloud.at<Vec3f>(row, col)[1] = mImageDepth.at<unsigned short>(row, col)*(row - cy) / fy;
          }
      cv::viz::WCloud cloud(point_cloud,image);
      window.showWidget("cloud",cloud);

      for(int i = 1;i < int(boxes.size())/4 + 1;i++){
          //cout << boxes[4*i-4] << " " << boxes[4*i-3] << " "<< boxes[4*i-2] << " "<< boxes[4*i-1] << " " ;
          rectangle(cImageBGR,Point(boxes[4*i-4],boxes[4*i-3]),Point(boxes[4*i-2],boxes[4*i-1]),
                    Scalar(255, 0, 0),10, LINE_8,0);
          viz::WCube cube_widget(Point3f(mImageDepth.at<unsigned short>(boxes[4*i-4],boxes[4*i-3])*(boxes[4*i-3] - cx) / fx,
                                         mImageDepth.at<unsigned short>(boxes[4*i-4],boxes[4*i-3])*(boxes[4*i-4] - cy) / fy,
                                         mImageDepth.at<unsigned short>(boxes[4*i-4],boxes[4*i-3])),
                                 Point3f(mImageDepth.at<unsigned short>(boxes[4*i-2],boxes[4*i-1])*(boxes[4*i-1] - cx) / fx,
                                         mImageDepth.at<unsigned short>(boxes[4*i-2],boxes[4*i-1])*(boxes[4*i-2] - cy) / fy,
                                         mImageDepth.at<unsigned short>(boxes[4*i-2],boxes[4*i-1])),
                                 true, viz::Color::green());
          cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
          window.showWidget("Cube Widget", cube_widget);
      }

      window.spinOnce(1, true);
      imshow("cv_show",cImageBGR);
      // 6a. check keyboard
      cv::waitKey( 1 );
  }

  // 9. stop
  mDepthStream.destroy();
  mColorStream.destroy();
  mDevice.close();
  OpenNI::shutdown();
  system("pause");
  return 0;
}


