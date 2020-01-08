/*
 *
 *
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>

#include <mutex>


pcl::visualization::PCLVisualizer viewer("PCL Viewer");

typedef pcl::PointXYZRGBA  PointT;
typedef pcl::PointCloud<PointT>  PointCloudT;

std::mutex cloud_mutex;


void cloud_cb_ (const PointCloudT::ConstPtr &callback_cloud, PointCloudT::Ptr& cloud)
{
  cloud_mutex.lock ();    // for not overwriting the point cloud from another thread
  *cloud = *callback_cloud;
  cloud_mutex.unlock ();
}


int main()
{
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    PointCloudT::Ptr cloud (new PointCloudT);
    pcl::Grabber* interface = new pcl::OpenNIGrabber();
    boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
        [&] (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& callback_cloud) { cloud_cb_ (callback_cloud, cloud); };
    interface->registerCallback(f);
    interface->start ();

    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
    viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
    viewer.setCameraPosition(0,0,-2,0,-1,0,0);
    viewer.spin();

    while (!viewer.wasStopped())
    {
        boost::this_thread::sleep (boost::posix_time::seconds (1));
    }

    interface->stop();

}

*/

/*
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/common/time.h>


#include <chrono>
#include <mutex>
#include <thread>

//using namespace std::literals::chrono_literals;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// PCL viewer //
pcl::visualization::PCLVisualizer viewer("PCL Viewer");

// Mutex: //
std::mutex cloud_mutex;

enum { COLS = 640, ROWS = 480 };

void cloud_cb_ (const PointCloudT::ConstPtr &callback_cloud, PointCloudT::Ptr& cloud,
    bool* new_cloud_available_flag)
{
  cloud_mutex.lock ();    // for not overwriting the point cloud from another thread
  *cloud = *callback_cloud;
  *new_cloud_available_flag = true;
  cloud_mutex.unlock ();
}

struct callback_args{
  // structure used to pass arguments to the callback function
  PointCloudT::Ptr clicked_points_3d;
  pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

void pp_callback (const pcl::visualization::PointPickingEvent& event, void* args)
{
  struct callback_args* data = (struct callback_args *)args;
  if (event.getPointIndex () == -1)
    return;
  PointT current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);
  data->clicked_points_3d->points.push_back(current_point);
  // Draw clicked points in red:
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red (data->clicked_points_3d, 200, 155, 0);
  data->viewerPtr->removePointCloud("clicked_points");
  data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
  data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
  std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}



int main (int argc, char** argv)
{

  // Algorithm parameters:
  std::string svm_filename = "/root/桌面/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
  float min_confidence = -1.5;
  float min_height = 1.3;
  float max_height = 2.3;
  float voxel_size = 0.06;
  Eigen::Matrix3f rgb_intrinsics_matrix;
  rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

  // Read Kinect live stream:
  PointCloudT::Ptr cloud (new PointCloudT);
  bool new_cloud_available_flag = false;
  pcl::Grabber* interface = new pcl::OpenNIGrabber();
  boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
      [&] (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& callback_cloud) { cloud_cb_ (callback_cloud, cloud, &new_cloud_available_flag); };
  interface->registerCallback(f);
  interface->start ();

  // Wait for the first frame:
  while(!new_cloud_available_flag)
   // std::this_thread::sleep_for(1ms);
  new_cloud_available_flag = false;

  cloud_mutex.lock ();    // for not overwriting the point cloud

  // Display pointcloud:
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Add point picking callback to viewer:
  struct callback_args cb_args;
  PointCloudT::Ptr clicked_points_3d (new PointCloudT);
  cb_args.clicked_points_3d = clicked_points_3d;
  cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(&viewer);
  viewer.registerPointPickingCallback (pp_callback, (void*)&cb_args);
  std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

  // Spin until 'Q' is pressed:
  viewer.spin();
  std::cout << "done." << std::endl;

  cloud_mutex.unlock ();

  // Ground plane estimation:
  Eigen::VectorXf ground_coeffs;
  ground_coeffs.resize(4);
  std::vector<int> clicked_points_indices;
  for (unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
    clicked_points_indices.push_back(i);
  pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
  model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
  std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

  // Initialize new viewer:
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");          // viewer initialization
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Create classifier for people detection:
  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM

  // People detection app initialization:
  pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
  people_detector.setVoxelSize(voxel_size);                        // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);                // set person classifier
  people_detector.setPersonClusterLimits(min_height, max_height, 0.1, 8.0);  // set person classifier
//  people_detector.setSensorPortraitOrientation(true);             // set sensor orientation to vertical

  // For timing:
  static unsigned count = 0;
  static double last = pcl::getTime ();

  // Main loop:
  while (!viewer.wasStopped())
  {
    if (new_cloud_available_flag && cloud_mutex.try_lock ())    // if a new cloud is available
    {
      new_cloud_available_flag = false;

      // Perform people detection on the new cloud:
      std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
      people_detector.setInputCloud(cloud);
      people_detector.setGround(ground_coeffs);                    // set floor coefficients
      people_detector.compute(clusters);                           // perform people detection

      ground_coeffs = people_detector.getGround();                 // get updated floor coefficients

      // Draw cloud and people bounding boxes in the viewer:
      viewer.removeAllPointClouds();
      viewer.removeAllShapes();
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
      viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
      unsigned int k = 0;
      for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
      {
        if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
        {

          // draw theoretical person bounding box in the PCL viewer:
          it->drawTBoundingBox(viewer, k);
          k++;
        }
      }
      std::cout << k << " people found" << std::endl;
      viewer.spinOnce();

      // Display average framerate:
      if (++count == 30)
      {
        double now = pcl::getTime ();
        std::cout << "Average framerate: " << double(count)/double(now - last) << " Hz" <<  std::endl;
        count = 0;
        last = now;
      }
      cloud_mutex.unlock ();
    }
  }

  return 0;
}
*/


#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/exceptions.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/people/people_detector.h>
#include <pcl/gpu/people/colormap.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <boost/filesystem.hpp>

#include <iostream>

namespace pc = pcl::console;
using namespace pcl::visualization;
using namespace pcl::gpu;
using namespace pcl;
using namespace std;

struct SampledScopeTime : public StopWatch
{
  enum { EACH = 33 };
  SampledScopeTime(int& time_ms) : time_ms_(time_ms) {}
  ~SampledScopeTime()
  {
    static int i_ = 0;
    time_ms_ += getTime ();
    if (i_ % EACH == 0 && i_)
    {
      std::cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )" << std::endl;
      time_ms_ = 0;
    }
    ++i_;
  }
  private:
  int& time_ms_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

string
make_name(int counter, const char* suffix)
{
  char buf[4096];
  sprintf (buf, "./people%04d_%s.png", counter, suffix);
  return buf;
}

template<typename T> void
savePNGFile(const std::string& filename, const pcl::gpu::DeviceArray2D<T>& arr)
{
  int c;
  pcl::PointCloud<T> cloud(arr.cols(), arr.rows());
  arr.download(cloud.points, c);
  pcl::io::savePNGFile(filename, cloud);
}

template <typename T> void
savePNGFile (const std::string& filename, const pcl::PointCloud<T>& cloud)
{
  pcl::io::savePNGFile(filename, cloud);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PeoplePCDApp
{
  public:
    typedef pcl::gpu::people::PeopleDetector PeopleDetector;

    enum { COLS = 640, ROWS = 480 };

    PeoplePCDApp (pcl::Grabber& capture) : capture_(capture), exit_(false), time_ms_(0), cloud_cb_(true), counter_(0), final_view_("Final labeling"), depth_view_("Depth")
    {
      final_view_.setSize (COLS, ROWS);
      depth_view_.setSize (COLS, ROWS);

      final_view_.setPosition (0, 0);
      depth_view_.setPosition (650, 0);

      cmap_device_.create(ROWS, COLS);
      cmap_host_.points.resize(COLS * ROWS);
      depth_device_.create(ROWS, COLS);
      image_device_.create(ROWS, COLS);

      depth_host_.points.resize(COLS * ROWS);

      rgba_host_.points.resize(COLS * ROWS);
      rgb_host_.resize(COLS * ROWS * 3);

      people::uploadColorMap(color_map_);

    }

    void
    visualizeAndWrite(bool write = false)
    {
      const PeopleDetector::Labels& labels = people_detector_.rdf_detector_->getLabels();
      people::colorizeLabels(color_map_, labels, cmap_device_);

      int c;
      cmap_host_.width = cmap_device_.cols();
      cmap_host_.height = cmap_device_.rows();
      cmap_host_.points.resize(cmap_host_.width * cmap_host_.height);
      cmap_device_.download(cmap_host_.points, c);

      final_view_.showRGBImage<pcl::RGB>(cmap_host_);
      final_view_.spinOnce(1, true);

      if (cloud_cb_)
      {
        depth_host_.width = people_detector_.depth_device1_.cols();
        depth_host_.height = people_detector_.depth_device1_.rows();
        depth_host_.points.resize(depth_host_.width * depth_host_.height);
        people_detector_.depth_device1_.download(depth_host_.points, c);
      }

      depth_view_.showShortImage(&depth_host_.points[0], depth_host_.width, depth_host_.height, 0, 5000, true);
      depth_view_.spinOnce(1, true);

      if (write)
      {
        if (cloud_cb_)
          savePNGFile(make_name(counter_, "ii"), cloud_host_);
        else
          savePNGFile(make_name(counter_, "ii"), rgba_host_);
        savePNGFile(make_name(counter_, "c2"), cmap_host_);
        savePNGFile(make_name(counter_, "s2"), labels);
        savePNGFile(make_name(counter_, "d1"), people_detector_.depth_device1_);
        savePNGFile(make_name(counter_, "d2"), people_detector_.depth_device2_);
      }
    }

    void source_cb1(const PointCloud<PointXYZRGBA>::ConstPtr& cloud)
    {
      {
        std::lock_guard<std::mutex> lock(data_ready_mutex_);
        if (exit_)
          return;

        pcl::copyPointCloud(*cloud, cloud_host_);
      }
      data_ready_cond_.notify_one();
    }

    void source_cb2(const openni_wrapper::Image::Ptr& image_wrapper, const openni_wrapper::DepthImage::Ptr& depth_wrapper, float)
    {
      {
        std::unique_lock<std::mutex> lock (data_ready_mutex_, std::try_to_lock);

        if (exit_ || !lock)
          return;

        //getting depth
        int w = depth_wrapper->getWidth();
        int h = depth_wrapper->getHeight();
        int s = w * PeopleDetector::Depth::elem_size;
        const unsigned short *data = depth_wrapper->getDepthMetaData().Data();
        depth_device_.upload(data, s, h, w);

        depth_host_.points.resize(w *h);
        depth_host_.width = w;
        depth_host_.height = h;
        std::copy(data, data + w * h, &depth_host_.points[0]);

        //getting image
        w = image_wrapper->getWidth();
        h = image_wrapper->getHeight();
        s = w * PeopleDetector::Image::elem_size;

        //fill rgb array
        rgb_host_.resize(w * h * 3);
        image_wrapper->fillRGB(w, h, (unsigned char*)&rgb_host_[0]);

        // convert to rgba, TODO image_wrapper should be updated to support rgba directly
        rgba_host_.points.resize(w * h);
        rgba_host_.width = w;
        rgba_host_.height = h;
        for(int i = 0; i < rgba_host_.size(); ++i)
        {
          const unsigned char *pixel = &rgb_host_[i * 3];
          RGB& rgba = rgba_host_.points[i];
          rgba.r = pixel[0];
          rgba.g = pixel[1];
          rgba.b = pixel[2];
        }
        image_device_.upload(&rgba_host_.points[0], s, h, w);
      }
      data_ready_cond_.notify_one();
    }

    void
    startMainLoop ()
    {
      cloud_cb_ = false;

      PCDGrabberBase* ispcd = dynamic_cast<pcl::PCDGrabberBase*>(&capture_);
      if (ispcd)
        cloud_cb_= true;

      typedef openni_wrapper::DepthImage::Ptr DepthImagePtr;
      typedef openni_wrapper::Image::Ptr ImagePtr;

      std::function<void (const PointCloud<PointXYZRGBA>::ConstPtr&)> func1 = [this] (const PointCloud<PointXYZRGBA>::ConstPtr& cloud) { source_cb1 (cloud); };
      std::function<void (const ImagePtr&, const DepthImagePtr&, float)> func2 = [this] (const ImagePtr& img, const DepthImagePtr& depth, float constant)
      {
        source_cb2 (img, depth, constant);
      };
      boost::signals2::connection c = cloud_cb_ ? capture_.registerCallback (func1) : capture_.registerCallback (func2);

      {
        std::unique_lock<std::mutex> lock(data_ready_mutex_);

        try
        {
          capture_.start ();
          while (!exit_ && !final_view_.wasStopped())
          {
            //bool has_data = (data_ready_cond_.wait_for(lock, 100ms) == std::cv_status::no_timeout);
            if(1)
            {
              SampledScopeTime fps(time_ms_);

              if (cloud_cb_)
                process_return_ = people_detector_.process(cloud_host_.makeShared());
              else
                process_return_ = people_detector_.process(depth_device_, image_device_);

              ++counter_;
            }

            if(has_data && (process_return_ == 2))
              visualizeAndWrite();
          }
          final_view_.spinOnce (3);
        }
        catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
        catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

        capture_.stop ();
      }
      c.disconnect();
    }

    std::mutex data_ready_mutex_;
    std::condition_variable data_ready_cond_;

    pcl::Grabber& capture_;

    bool cloud_cb_;
    bool exit_;
    int time_ms_;
    int counter_;
    int process_return_;
    PeopleDetector people_detector_;
    PeopleDetector::Image cmap_device_;
    pcl::PointCloud<pcl::RGB> cmap_host_;

    PeopleDetector::Depth depth_device_;
    PeopleDetector::Image image_device_;

    pcl::PointCloud<unsigned short> depth_host_;
    pcl::PointCloud<pcl::RGB> rgba_host_;
    std::vector<unsigned char> rgb_host_;

    PointCloud<PointXYZRGBA> cloud_host_;

    ImageViewer final_view_;
    ImageViewer depth_view_;

    DeviceArray<pcl::RGB> color_map_;
};

int main(int argc, char** argv)
{
  // selecting GPU and prining info
  int device = 0;
  pc::parse_argument (argc, argv, "-gpu", device);
  pcl::gpu::setDevice (device);
  pcl::gpu::printShortCudaDeviceInfo (device);

  // selecting data source
  pcl::Grabber::Ptr capture (new pcl::OpenNIGrabber());

  //selecting tree files
  std::vector<string> tree_files;
  tree_files.push_back("Data/forest1/tree_20.txt");
  tree_files.push_back("Data/forest2/tree_20.txt");
  tree_files.push_back("Data/forest3/tree_20.txt");
  tree_files.push_back("Data/forest4/tree_20.txt");

  pc::parse_argument (argc, argv, "-tree0", tree_files[0]);
  pc::parse_argument (argc, argv, "-tree1", tree_files[1]);
  pc::parse_argument (argc, argv, "-tree2", tree_files[2]);
  pc::parse_argument (argc, argv, "-tree3", tree_files[3]);

  int num_trees = (int)tree_files.size();
  pc::parse_argument (argc, argv, "-numTrees", num_trees);

  tree_files.resize(num_trees);
  if (num_trees == 0 || num_trees > 4)
    return std::cout << "Invalid number of trees" << std::endl, -1;

  try
  {
    // loading trees
    typedef pcl::gpu::people::RDFBodyPartsDetector RDFBodyPartsDetector;
    RDFBodyPartsDetector::Ptr rdf(new RDFBodyPartsDetector(tree_files));
    PCL_INFO("Loaded files into rdf");

    // Create the app
    PeoplePCDApp app(*capture);
    app.people_detector_.rdf_detector_ = rdf;

    // executing
    app.startMainLoop ();
  }
  catch (const pcl::PCLException& e) { std::cout << "PCLException: " << e.detailedMessage() << std::endl; }
  catch (const std::runtime_error& e) { std::cout << e.what() << std::endl; }
  catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
  catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

  return 0;
}

