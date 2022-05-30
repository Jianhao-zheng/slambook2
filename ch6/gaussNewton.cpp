#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;   // real value
  double ae = 2.0, be = -1.0, ce = 5.0;  // estimated value

  double sigma = 1.0;
  double inv_sigma = 1.0 / sigma;

  int N = 100;
  cv::RNG rng;

  vector<double> x_sample, y_sample;
  x_sample.resize(N);
  y_sample.resize(N);
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    // this generator can estimate the exact value
    //  double x = rng.uniform((double)-5, (double)5);
    x_sample[i] = x;
    y_sample[i] = exp(ar * x * x + br * x + cr) + rng.gaussian(sigma * sigma);
  }

  // Gaussian Newton
  int iters = 100;
  double prev_cost = DBL_MAX;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (int iter = 0; iter < iters; iters++) {
    Matrix3d H = Matrix3d::Zero();
    Vector3d g = Vector3d::Zero();
    double cost = 0;

    for (int i = 0; i < N; i++) {
      Vector3d J = Vector3d::Zero();

      double exp_temp =
          exp(ae * x_sample[i] * x_sample[i] + be * x_sample[i] + ce);
      double e = y_sample[i] - exp_temp;

      J[0] = -x_sample[i] * x_sample[i] * exp_temp;
      J[1] = -x_sample[i] * exp_temp;
      J[2] = -exp_temp;

      H += inv_sigma * inv_sigma * J * J.transpose();
      g += -inv_sigma * inv_sigma * e * J;
      cost += e * e;
    }

    // cout << "Current cost: " << cost << endl;

    if (cost >= prev_cost) {
      //   cout << "Current cost: " << cost
      //        << " is greater than previous one. Break!" << endl;
      break;
    }
    prev_cost = cost;

    Vector3d delta_x = H.ldlt().solve(g);
    ae += delta_x[0];
    be += delta_x[1];
    ce += delta_x[2];
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "estimated value, ae= " << ae << " be= " << be << " ce=" << ce
       << endl;
  cout << "total time used: " << time_used.count() << "sec" << endl;

  return 0;
}