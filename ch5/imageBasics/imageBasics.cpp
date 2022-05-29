#include <chrono>
#include <iostream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
  // read image from argv[1]
  cv::Mat image;
  image = cv::imread(argv[1]);

  // check if image loaded
  if (image.data == nullptr) {
    cerr << "file \"" << argv[1] << "\" doesn't exist" << endl;
    return 0;
  }

  cout << "width is:" << image.cols << ",height is:" << image.rows
       << ",number of channels is:" << image.channels() << endl;
  cv::imshow("image", image);
  cv::waitKey(0);

  if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
    cout << "wrong image type, please input an rgb or gray image." << endl;
    return 0;
  }

  cout << "*************method given by the author************" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (size_t y = 0; y < image.rows; y++) {
    unsigned char *row_ptr = image.ptr<unsigned char>(y);
    for (size_t x = 0; x < image.cols; x++) {
      unsigned char *data_ptr = &row_ptr[x * image.channels()];

      for (int c = 0; c != image.channels(); c++) {
        unsigned char data = data_ptr[c];
      }
    }
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout << "elapsed time for viewing all pixels is: " << time_used.count()
       << " sec" << endl;

  cout << "*************method using \".at\"************" << endl;
  t1 = chrono::steady_clock::now();
  for (size_t y = 0; y < image.rows; y++) {
    for (size_t x = 0; x < image.cols; x++) {
      unsigned char data = image.at<uchar>(y, x);
    }
  }
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "elapsed time for viewing all pixels is: " << time_used.count()
       << " sec" << endl;

  // following will not copy the image
  // any change in "image_another" will change the value in the original matrix
  cv::Mat image_another = image;
  image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
  cv::imshow("image", image);
  cout << "only change the copied image with \"=\"" << endl;
  cv::waitKey(0);

  // use clone to copy the image
  cv::Mat image_clone = image.clone();
  image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
  cv::imshow("image", image);
  cout << "set the lefttop block to black in copied image by \".clone()\""
       << endl;
  cv::waitKey(0);
  cv::imshow("image_clone", image_clone);
  cout << "print copied image by \".clone()\"" << endl;
  cv::waitKey(0);

  cv::destroyAllWindows();
  return 0;
}
