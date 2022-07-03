#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout
        << " wrong number of input, correct usage: feature_extraction img1 img2"
        << endl;
    return 1;
  }

  // Load image
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr &&
         img_2.data != nullptr);  // check if images loaded successfully

  // Initialization
  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptor_1, descriptor_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor_generator = ORB::create();
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");

  // Step-1: detect orientaed FAST
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  // Step-2: compute descriptors
  descriptor_generator->compute(img_1, keypoints_1, descriptor_1);
  descriptor_generator->compute(img_2, keypoints_2, descriptor_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB costs: " << time_used.count() << " seconds." << endl;

  Mat img1_with_keypoints;
  drawKeypoints(img_1, keypoints_1, img1_with_keypoints, Scalar::all(-1),
                DrawMatchesFlags::DEFAULT);
  imshow("ORB features", img1_with_keypoints);

  // Step-3: Match the keypoints of two images
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptor_1, descriptor_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB features takes: " << time_used.count() << "seconds."
       << endl;

  // Step-4: Filter bad matches
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) {
                                  return m1.distance < m2.distance;
                                });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  vector<DMatch> good_matches;
  for (int i = 0; i < descriptor_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  // Step-5 draw match result
  Mat img_match;
  Mat img_filtered_match;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches,
              img_filtered_match);
  imshow("all matches", img_match);
  imshow("good matches", img_filtered_match);
  waitKey(0);

  return 0;
}