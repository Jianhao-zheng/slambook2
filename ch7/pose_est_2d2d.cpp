#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

void pose_estimation_2d2d(vector<KeyPoint> keypoints_1,
                          vector<KeyPoint> keypoints_2, vector<DMatch> matches,
                          Mat &R, Mat &t);

Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
  // Load image
  if (argc != 3) {
    cout << " wrong number of input, correct usage: pose_est_2d2d img1 img2"
         << endl;
    return 1;
  }
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr &&
         img_2.data != nullptr);  // check if images loaded successfully

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  // verify E = t^R*alpha
  Mat t_hat = (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
               t.at<double>(2, 0), 0, -t.at<double>(0, 0), -t.at<double>(1, 0),
               t.at<double>(0, 0), 0);
  cout << "t^*R = :" << endl << t_hat * R << endl;

  // verify epipolar constraint
  //   Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0,
  //            1);  // intrinsic matrix
  //   for (DMatch m : matches) {
  //     Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
  //     Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
  //     Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
  //     Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
  //     Mat d = y2.t() * t_hat * R * y1;
  //     cout << "epipolar constraint = " << d << endl;
  //   }

  return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches) {
  Mat descriptor_1, descriptor_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor_generator = ORB::create();
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");

  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  descriptor_generator->compute(img_1, keypoints_1, descriptor_1);
  descriptor_generator->compute(img_2, keypoints_2, descriptor_2);

  vector<DMatch> initial_matches;
  matcher->match(descriptor_1, descriptor_2, initial_matches);

  auto min_max = minmax_element(initial_matches.begin(), initial_matches.end(),
                                [](const DMatch &m1, const DMatch &m2) {
                                  return m1.distance < m2.distance;
                                });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptor_1.rows; i++) {
    if (initial_matches[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(initial_matches[i]);
    }
  }
}

void pose_estimation_2d2d(vector<KeyPoint> keypoints_1,
                          vector<KeyPoint> keypoints_2, vector<DMatch> matches,
                          Mat &R, Mat &t) {
  vector<Point2f> keypoints1_2f;
  vector<Point2f> keypoints2_2f;

  for (int i = 0; i < (int)matches.size(); i++) {
    keypoints1_2f.push_back(keypoints_1[matches[i].queryIdx].pt);
    keypoints2_2f.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  // find fundamental matrix
  Mat F_matrix = findFundamentalMat(keypoints1_2f, keypoints2_2f, CV_FM_8POINT);
  cout << "fundamental matrix is: " << endl << F_matrix << endl;

  // find essential matrix
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0,
           1);  // intrinsic matrix
  Point2d principal_point(325.1, 249.7);
  double focal_length = 521;
  Mat E_matrix = findEssentialMat(keypoints1_2f, keypoints2_2f, K);
  cout << "essential_matrix is: " << endl << E_matrix << endl;

  recoverPose(E_matrix, keypoints1_2f, keypoints2_2f, K, R, t);
  cout << "R is:" << endl << R << endl;
  cout << "t is:" << endl << t << endl;
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                 (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}