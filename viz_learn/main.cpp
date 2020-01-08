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
    Mat Image,cut;
    Mat depth2;
    viz::Viz3d window("window");
    //显示坐标系
    window.showWidget("Coordinate", viz::WCoordinateSystem());


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




    while (!window.wasStopped())
    {
        capture.grab();
        capture.retrieve( depth, CAP_OPENNI_DEPTH_MAP);
        capture.retrieve( depth2, CAP_OPENNI_DISPARITY_MAP);
        capture.retrieve( Image, CAP_OPENNI_BGR_IMAGE);


        Mat result(480, 640, CV_8UC3);
        int i = 0;
        for (int row = 0; row < depth.rows; row++)
        {
            for (int col = 0; col < depth.cols; col++)
            {
                unsigned short* p = (unsigned short*)depth.data;
                unsigned short depthValue = p[row * depth.cols + col];
                //cout << "depthValue       " << depthValue << endl;
                if (depthValue != -std::numeric_limits<unsigned short>::infinity()  && depthValue != 0 && depthValue != 65535)
                {
                    // 投影到彩色图上的坐标
                    Eigen::Vector3f uv_depth(col, row, 1.0f);
                    Eigen::Vector3f uv_color = depthValue / 1000.f*R*uv_depth + T / 1000;   //用于计算映射，核心式子

                    int X = static_cast<int>(uv_color[0] / uv_color[2]);         //计算X，即对应的X值
                    int Y = static_cast<int>(uv_color[1] / uv_color[2]);         //计算Y，即对应的Y值

                    if ((X >= 0 && X < Image.cols) && (Y >= 0 && Y < Image.rows))
                    {

                        result.data[i * 3] = Image.data[3 * (Y * Image.cols + X)];
                        result.data[i * 3 + 1] = Image.data[3 * (Y * Image.cols + X) + 1];
                        result.data[i * 3 + 2] = Image.data[3 * (Y * Image.cols + X) + 2];
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

        imshow("结果图", result);
        waitKey(1);

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
        cv::viz::WCloud cloud(point_cloud,result);
        window.showWidget("cloud",cloud);

        int x = point_cloud.at<Vec3f>(320, 240)[0];
        int y = point_cloud.at<Vec3f>(320, 240)[1];
        int z = point_cloud.at<Vec3f>(320, 240)[2];

        viz::WCube cube_widget(Point3f(200+x,200+y,0.0+z), Point3f(0.0+x,0.0+y,-200+z), true, viz::Color::green());
        cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
        window.showWidget("Cube Widget", cube_widget);


        imshow("dep",depth2);

        resize(Image,cut,Size(Image.cols*0.5,Image.rows*0.5));
        imshow("rgb",cut);
        waitKey( 3 );
        window.spinOnce(1, true);
    }


}
