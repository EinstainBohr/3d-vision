#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd/kinfu.hpp>

using namespace cv;
using namespace cv::kinfu;
using namespace std;

#include <opencv2/viz.hpp>

namespace Kinect2Params
{
    static const Size frameSize = Size(512, 424);
    // approximate values, no guarantee to be correct
    static const float focal = 366.1f;
    static const float cx = 258.2f;
    static const float cy = 204.f;
    static const float k1 =  0.12f;
    static const float k2 = -0.34f;
    static const float k3 =  0.12f;
};

struct DepthSource
{
public:
    DepthSource(int cam) :
        DepthSource("", cam)
    { }

    DepthSource(String fileListName) :
        DepthSource(fileListName, -1)
    { }

    DepthSource(String fileListName, int cam) :
        frameIdx(0),
        vc( cam >= 0 ? VideoCapture(VideoCaptureAPIs::CAP_OPENNI2 + cam) : VideoCapture()),
        undistortMap1(),
        undistortMap2(),
        useKinect2Workarounds(true)
    { }

    UMat getDepth()
    {
        UMat out;

        vc.grab();
        vc.retrieve(out, CAP_OPENNI_DEPTH_MAP);

        // workaround for Kinect 2
        if(useKinect2Workarounds)
        {
            out = out(Rect(Point(), Kinect2Params::frameSize));

            UMat outCopy;
            // linear remap adds gradient between valid and invalid pixels
            // which causes garbage, use nearest instead
            remap(out, outCopy, undistortMap1, undistortMap2, cv::INTER_NEAREST);

            cv::flip(outCopy, out, 1);
        }

        if (out.empty())
            throw std::runtime_error("Matrix is empty");
        return out;
    }

    bool empty()
    {
        return depthFileList.empty() && !(vc.isOpened());
    }

    void updateParams(Params& params)
    {
        if (vc.isOpened())
        {
            // this should be set in according to user's depth sensor
            int w = (int)vc.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
            int h = (int)vc.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);

            float focal = (float)vc.get(CAP_OPENNI_DEPTH_GENERATOR | CAP_PROP_OPENNI_FOCAL_LENGTH);

            // it's recommended to calibrate sensor to obtain its intrinsics
            float fx, fy, cx, cy;
            Size frameSize;
            if(useKinect2Workarounds)
            {
                fx = fy = Kinect2Params::focal;
                cx = Kinect2Params::cx;
                cy = Kinect2Params::cy;

                frameSize = Kinect2Params::frameSize;
            }
            else
            {
                fx = fy = focal;
                cx = w/2 - 0.5f;
                cy = h/2 - 0.5f;

                frameSize = Size(w, h);
            }

            Matx33f camMatrix = Matx33f(fx,  0, cx,
                                        0,  fy, cy,
                                        0,   0,  1);

            params.frameSize = frameSize;
            params.intr = camMatrix;
            params.depthFactor = 1000.f;

            Matx<float, 1, 5> distCoeffs;
            distCoeffs(0) = Kinect2Params::k1;
            distCoeffs(1) = Kinect2Params::k2;
            distCoeffs(4) = Kinect2Params::k3;
            if(useKinect2Workarounds)
                initUndistortRectifyMap(camMatrix, distCoeffs, cv::noArray(),
                                        camMatrix, frameSize, CV_16SC2,
                                        undistortMap1, undistortMap2);
        }
    }

    vector<string> depthFileList;
    size_t frameIdx;
    VideoCapture vc;
    UMat undistortMap1, undistortMap2;
    bool useKinect2Workarounds;
};


const std::string vizWindowName = "cloud";


int main(int argc, char **argv)
{
    bool coarse = false;
    bool idle = false;

    //coarse = true;

    //recordPath = "record";

    //idle = true;


    Ptr<DepthSource> ds;

    ds = makePtr<DepthSource>(0);

    Ptr<Params> params;
    Ptr<KinFu> kf;

    if(coarse)
        params = Params::coarseParams();
    else
        params = Params::defaultParams();

    // These params can be different for each depth sensor
    ds->updateParams(*params);

    // Enables OpenCL explicitly (by default can be switched-off)
    cv::setUseOptimized(true);

    // Scene-specific params should be tuned for each scene individually
    //params->volumePose = params->volumePose.translate(Vec3f(0.f, 0.f, 0.5f));
    //params->tsdf_max_weight = 16;

    if(!idle)
        kf = KinFu::create(params);

    cv::viz::Viz3d window(vizWindowName);
    window.setViewerPose(Affine3f::Identity());
    bool pause = false;


    UMat rendered;
    UMat points;
    UMat normals;

    for(UMat frame = ds->getDepth(); !frame.empty(); frame = ds->getDepth())
    {
        UMat cvt8;
        float depthFactor = params->depthFactor;
        convertScaleAbs(frame, cvt8, 0.25*256. / depthFactor);
        if(!idle)
        {
            imshow("depth", cvt8);

            if(!kf->update(frame))
            {
                kf->reset();
                std::cout << "reset" << std::endl;
            }
            else
            {
                if(1)
                {
                    kf->getCloud(points, normals);
                    if(!points.empty() && !normals.empty())
                    {
                        viz::WCloud cloudWidget(points, viz::Color::red());
                        viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::blue());
                        window.showWidget("cloud", cloudWidget);
                        //window.showWidget("normals", cloudNormals);
                    }
                }

                //window.showWidget("worldAxes", viz::WCoordinateSystem());
                Vec3d volSize = kf->getParams().voxelSize*kf->getParams().volumeDims;
                window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                     volSize),
                                  kf->getParams().volumePose);
                window.setViewerPose(kf->getPose());
                window.spinOnce(1, false);
            }

            kf->render(rendered);
        }
        else
        {
            rendered = cvt8;
        }



        imshow("render", rendered);

        waitKey(1);
    }

    return 0;
}
